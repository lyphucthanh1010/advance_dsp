import os
import json
import glob
import pickle
import numpy as np
import tensorflow as tf
import librosa
import multiprocessing
import argparse
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from essentia.standard import MonoLoader, Chromaprinter
import chromaprint

# ---------- CONFIGURATION ----------
SEGMENT_LENGTH = 10          # Độ dài đoạn âm thanh: 10 giây (dùng trong extract training)
SLIDE_LENGTH = 5             # Cửa sổ trượt: 5 giây (50% overlap)
MIN_SAMPLES_PER_CLASS = 40   # Số mẫu tối thiểu cho mỗi lớp
FINGERPRINT_SEGMENT_LENGTH = 30      # Số giá trị cho mỗi fingerprint chunk
AUDIO_FOLDER = "music_dataset_wav"    # Thư mục chứa file huấn luyện
OUTPUT_FOLDER = "chromaprints_output"   # Thư mục lưu file JSON của tập huấn luyện
TRAINING_EMBEDDINGS_PICKLE = "training_embeddings.pkl"  # Lưu các embedding (sẽ được lưu trong LabelEncoder)
LABEL_ENCODER_PICKLE = "label_encoder.pkl"              # Lưu LabelEncoder (với thuộc tính training_embeddings)
SR = 22050                           # Sampling rate
EMBEDDING_DIM = 128          # Kích thước embedding từ model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Sử dụng CPU

# ---------- EXTRACT CHROMAPRINT ----------
def segment_audio_sliding(file_path, segment_length=SEGMENT_LENGTH, slide_length=SLIDE_LENGTH, sr=SR):
    audio, _ = librosa.load(file_path, sr=sr)
    segments = []
    samples_per_segment = int(segment_length * sr)
    step_samples = int(slide_length * sr)
    total_samples = len(audio)
    for start in range(0, total_samples - samples_per_segment + 1, step_samples):
        end = start + samples_per_segment
        segments.append(audio[start:end])
    return segments

def get_chromaprints(args):
    file_path, output_file = args
    try:
        segments = segment_audio_sliding(file_path, segment_length=SEGMENT_LENGTH, slide_length=SLIDE_LENGTH, sr=SR)
        all_fp_chunks = []
        for segment in segments:
            fp_full_char = Chromaprinter()(segment)
            fp_full = chromaprint.decode_fingerprint(fp_full_char.encode())[0]
            fp_chunks = [fp_full[i:i+FINGERPRINT_SEGMENT_LENGTH] for i in range(0, len(fp_full), FINGERPRINT_SEGMENT_LENGTH)]
            if len(fp_chunks) == 1 and len(fp_chunks[0]) < FINGERPRINT_SEGMENT_LENGTH:
                pad_length = FINGERPRINT_SEGMENT_LENGTH - len(fp_chunks[0])
                fp_chunks[0] += [0] * pad_length
            elif len(fp_chunks) > 1 and len(fp_chunks[-1]) != FINGERPRINT_SEGMENT_LENGTH:
                del fp_chunks[-1]
            all_fp_chunks.extend(fp_chunks)
        label = os.path.basename(file_path).split('_')[0]
        with open(output_file, 'w') as f:
            json.dump({"chromaprints": all_fp_chunks, "label": label}, f, indent=2)
        return output_file
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def get_all_chromaprints(file_paths, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    args_list = []
    for file_path in file_paths:
        base = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_folder, f"{base}.json")
        args_list.append((file_path, output_file))
    with multiprocessing.Pool() as pool:
        results = pool.map(get_chromaprints, args_list)
    return results

def get_audio_file_paths(audio_folder, max_files=None):
    file_paths = []
    for root, dirs, files in os.walk(audio_folder):
        for file in files:
            if file.lower().endswith((".mp3", ".wav", ".flac")):
                file_paths.append(os.path.join(root, file))
    if max_files is not None:
        file_paths = file_paths[:max_files]
    return file_paths

def load_dataset_from_json(json_folder):
    X = []
    y = []
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            label = data["label"]
            for seg in data["chromaprints"]:
                seg_array = np.array(seg, dtype=np.float32).reshape(FINGERPRINT_SEGMENT_LENGTH, 1)
                X.append(seg_array)
                y.append(label)
    X = np.array(X)
    X = np.expand_dims(X, axis=-1)  # (samples, FINGERPRINT_SEGMENT_LENGTH, 1, 1)
    y = np.array(y)
    return X, y

def filter_min_samples(X, y, min_samples=MIN_SAMPLES_PER_CLASS):
    counts = Counter(y)
    valid_labels = {label for label, count in counts.items() if count >= min_samples}
    X_filtered, y_filtered = [], []
    for x, label in zip(X, y):
        if label in valid_labels:
            X_filtered.append(x)
            y_filtered.append(label)
    return np.array(X_filtered), np.array(y_filtered)

# ---------- MODEL (CLASSIFIER) ----------
def build_classifier_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((1,2), padding='same'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((1,2), padding='same'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------- SAVE & LOAD TRAINING DATA (EMBEDDINGS & LABEL_ENCODER) ----------
def save_training_data(json_folder, base_network, output_pickle=TRAINING_EMBEDDINGS_PICKLE):
    # Tính trung bình embedding cho mỗi file audio từ các file JSON.
    training_embeddings = {}
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    labels = []
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            label = data["label"]
            labels.append(label)
            fp_segments = data["chromaprints"]
            seg_arrays = [np.array(seg, dtype=np.float32).reshape(FINGERPRINT_SEGMENT_LENGTH, 1, 1)
                          for seg in fp_segments]
            seg_embeddings = []
            for seg in seg_arrays:
                seg_exp = np.expand_dims(seg, axis=0)
                emb = base_network.predict(seg_exp)
                seg_embeddings.append(emb[0])
            if seg_embeddings:
                mean_emb = np.mean(seg_embeddings, axis=0)
                training_embeddings[jf] = {"label": label, "embedding": mean_emb}
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    # Lưu training embeddings bên trong label_encoder như thuộc tính bổ sung
    label_encoder.training_embeddings = training_embeddings
    with open(LABEL_ENCODER_PICKLE, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Saved training embeddings and LabelEncoder to '{LABEL_ENCODER_PICKLE}'")
    return training_embeddings, label_encoder

# ---------- MAIN TRAINING ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_files", type=int, default=None,
                        help="Số lượng bài hát tối đa cần load từ AUDIO_FOLDER")
    args = parser.parse_args()
    
    print("Collecting training audio file paths...")
    training_files = get_audio_file_paths(AUDIO_FOLDER, max_files=args.max_files)
    print(f"Found {len(training_files)} audio files.")
    
    print("Extracting chromaprints from training files...")
    get_all_chromaprints(training_files, OUTPUT_FOLDER)
    
    print("Loading dataset from JSON files...")
    X, y = load_dataset_from_json(OUTPUT_FOLDER)
    print("Dataset shape:", X.shape, y.shape)
    
    print("Filtering classes with at least", MIN_SAMPLES_PER_CLASS, "samples...")
    X, y = filter_min_samples(X, y, min_samples=MIN_SAMPLES_PER_CLASS)
    print("Filtered dataset shape:", X.shape, y.shape)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print("Number of unique classes:", num_classes)
    print("Selected labels:", label_encoder.classes_)
    
    input_shape = (FINGERPRINT_SEGMENT_LENGTH, 1, 1)
    classifier_model = build_classifier_model(input_shape, num_classes)
    
    classifier_model.fit(X, y_encoded, epochs=50, batch_size=32, validation_split=0.2)
    # Lưu mô hình phân loại ở định dạng native Keras (*.keras)
    classifier_model.save("classifier_model.keras")
    print("Saved classifier model to 'classifier_model.keras'")
    
    # Tạo base_network từ classifier (loại bỏ lớp softmax cuối cùng)
    base_network = tf.keras.Model(inputs=classifier_model.input,
                                  outputs=classifier_model.layers[-2].output)
    base_network.save("base_network.keras")
    print("Saved base_network model to 'base_network.keras'")
    
    training_embeddings, label_encoder = save_training_data(OUTPUT_FOLDER, base_network, output_pickle=TRAINING_EMBEDDINGS_PICKLE)
