import os
import json
import glob
import numpy as np
import multiprocessing
from collections import Counter
from essentia.standard import MonoLoader, Chromaprinter
import chromaprint
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Input, Lambda
from sklearn.preprocessing import LabelEncoder
import librosa
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pytube import YouTube
from pydub import AudioSegment
import tempfile

# ---------- CẤU HÌNH TOÀN CỤC ----------
SEGMENT_LENGTH = 10          # Độ dài đoạn âm thanh: 10 giây
SLIDE_LENGTH = 5             # Cửa sổ trượt: 5 giây (50% overlap)
MIN_SAMPLES_PER_CLASS = 40   # Số mẫu tối thiểu cho mỗi lớp
FINGERPRINT_SEGMENT_LENGTH = 30  # Số giá trị cho mỗi fingerprint chunk
AUDIO_FOLDER = "music_dataset_wav"    # Thư mục chứa file huấn luyện
OUTPUT_FOLDER = "chromaprints_output"   # Thư mục lưu file JSON kết quả
SR = 22050                   # Sampling rate
EMBEDDING_DIM = 128          # Kích thước embedding từ base network
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Sử dụng CPU

# ---------- PHẦN EXTRACT CHROMAPRINT (giống code đã có) ----------
def segment_audio_sliding(file_path, segment_length=SEGMENT_LENGTH, slide_length=SLIDE_LENGTH, sr=SR):
    audio, _ = librosa.load(file_path, sr=sr)
    segments = []
    samples_per_segment = int(segment_length * sr)
    step_samples = int(slide_length * sr)
    total_samples = len(audio)
    for start in range(0, total_samples - samples_per_segment + 1, step_samples):
        end = start + samples_per_segment
        segment = audio[start:end]
        segments.append(segment)
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
            data = {"chromaprints": all_fp_chunks, "label": label}
            json.dump(data, f, indent=2)
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

def get_audio_file_paths(audio_folder):
    file_paths = []
    for root, dirs, files in os.walk(audio_folder):
        for file in files:
            if file.lower().endswith((".mp3", ".wav", ".flac")):
                file_paths.append(os.path.join(root, file))
    return file_paths

def load_dataset_from_json(json_folder):
    X = []
    y = []
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            label = data["label"]
            segments = data["chromaprints"]
            for seg in segments:
                seg_array = np.array(seg, dtype=np.float32).reshape(FINGERPRINT_SEGMENT_LENGTH, 1)
                X.append(seg_array)
                y.append(label)
    X = np.array(X)
    X = np.expand_dims(X, axis=-1)  # shape: (samples, FINGERPRINT_SEGMENT_LENGTH, 1, 1)
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

# ---------- PHẦN SIAMESE NETWORK ----------
def create_pairs(X, y):
    pairs = []
    pair_labels = []
    label_indices = {}
    for i, label in enumerate(y):
        label_indices.setdefault(label, []).append(i)
    n = len(X)
    for i in range(n):
        current_label = y[i]
        # Tạo cặp dương (similar)
        pos_indices = label_indices[current_label]
        if len(pos_indices) > 1:
            j = np.random.choice([idx for idx in pos_indices if idx != i])
            pairs.append([X[i], X[j]])
            pair_labels.append(0)
        # Tạo cặp âm (dissimilar)
        neg_label = np.random.choice([l for l in label_indices.keys() if l != current_label])
        neg_index = np.random.choice(label_indices[neg_label])
        pairs.append([X[i], X[neg_index]])
        pair_labels.append(1)
    return np.array(pairs), np.array(pair_labels)

def create_base_network(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((1,2), padding='same'),
        Dropout(0.25),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((1,2), padding='same'),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu')
    ])
    return model

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean((1 - y_true) * 0.5 * tf.square(y_pred) +
                          y_true * 0.5 * tf.square(tf.maximum(margin - y_pred, 0)))

def build_siamese_model(input_shape):
    base_network = create_base_network(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    model.compile(loss=contrastive_loss, optimizer='adam')
    model.summary()
    return model, base_network

# ---------- PHẦN TÍNH EMBEDDING CHO FILE ----------
def extract_fp_segments_from_audio(file_path):
    try:
        audio = MonoLoader(filename=file_path)()
        fp_full_char = Chromaprinter()(audio)
        fp_full = chromaprint.decode_fingerprint(fp_full_char.encode())[0]
        segments = [fp_full[i:i+FINGERPRINT_SEGMENT_LENGTH] for i in range(0, len(fp_full), FINGERPRINT_SEGMENT_LENGTH)]
        if len(segments) == 1 and len(segments[0]) < FINGERPRINT_SEGMENT_LENGTH:
            pad_length = FINGERPRINT_SEGMENT_LENGTH - len(segments[0])
            segments[0] += [0] * pad_length
        elif len(segments) > 1 and len(segments[-1]) != FINGERPRINT_SEGMENT_LENGTH:
            del segments[-1]
        seg_arrays = [np.array(seg, dtype=np.float32).reshape(FINGERPRINT_SEGMENT_LENGTH, 1) for seg in segments]
        return seg_arrays
    except Exception as e:
        print(f"Error extracting fingerprint segments from {file_path}: {e}")
        return []

def sliding_window_embedding(file_path, base_network):
    try:
        segments = extract_fp_segments_from_audio(file_path)
        if not segments:
            return None
        embeddings = []
        for seg in segments:
            seg_exp = np.expand_dims(seg, axis=0)  # (1, FINGERPRINT_SEGMENT_LENGTH, 1)
            seg_exp = np.expand_dims(seg_exp, axis=-1)  # (1, FINGERPRINT_SEGMENT_LENGTH, 1, 1)
            emb = base_network.predict(seg_exp)
            embeddings.append(emb[0])
        embeddings = np.array(embeddings)
        mean_emb = np.mean(embeddings, axis=0)
        return mean_emb
    except Exception as e:
        print(f"Error in sliding window embedding for {file_path}: {e}")
        return None

# ---------- PHẦN INTEGRATE WITH ELASTICSEARCH ----------
def create_elasticsearch_index(es, index_name, embedding_dim):
    mapping = {
        "mappings": {
            "properties": {
                "file_path": {"type": "keyword"},
                "label": {"type": "keyword"},
                "embedding": {"type": "dense_vector", "dims": embedding_dim}
            }
        }
    }
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=mapping)
    print(f"Created index '{index_name}' with embedding dimension {embedding_dim}")

def index_embeddings(es, index_name, embeddings_dict):
    actions = []
    for file_path, (label, embedding) in embeddings_dict.items():
        actions.append({
            "_index": index_name,
            "_id": file_path,
            "_source": {
                "file_path": file_path,
                "label": label,
                "embedding": embedding.tolist()
            }
        })
    bulk(es, actions)
    print(f"Indexed {len(actions)} documents into '{index_name}'")

def search_similar(es, index_name, query_embedding, top_k=5):
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_embedding.tolist()}
            }
        }
    }
    response = es.search(index=index_name, body={"size": top_k, "query": script_query})
    return response

def download_audio_from_youtube(url, output_path="temp_audio.wav"):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    downloaded_file = audio_stream.download(filename="temp_audio.mp4")
    sound = AudioSegment.from_file(downloaded_file, format="mp4")
    sound.export(output_path, format="wav")
    os.remove(downloaded_file)
    return output_path

# ---------- CHẠY TOÀN BỘ LUỒNG ----------
if __name__ == '__main__':
    # Bước 1: Extract chromaprints từ tập huấn luyện
    print("Collecting training audio file paths...")
    training_files = get_audio_file_paths(AUDIO_FOLDER)
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
    
    input_shape = (FINGERPRINT_SEGMENT_LENGTH, 1, 1)
    # Tạo cặp cho Siamese network
    pairs, pair_labels = create_pairs(X, y)
    print("Pairs shape:", pairs.shape, "Pair labels shape:", pair_labels.shape)
    
    # Huấn luyện Siamese network
    siamese_model, base_network = build_siamese_model(input_shape)
    siamese_model.fit([pairs[:,0], pairs[:,1]], pair_labels, batch_size=32, epochs=50, validation_split=0.2)
    
    # Lưu mô hình sau khi huấn luyện
    siamese_model.save("siamese_model.h5")
    base_network.save("base_network.h5")
    print("Models saved to disk.")
    
    # Tính embedding cho mỗi file huấn luyện và index vào Elasticsearch
    embeddings_by_file = {}
    for file_path in training_files:
        true_label = os.path.basename(file_path).split('_')[0]
        emb = sliding_window_embedding(file_path, base_network)
        if emb is not None:
            embeddings_by_file[file_path] = (true_label, emb)
            print(f"File: {os.path.basename(file_path)} | Label: {true_label}")
    
    es = Elasticsearch("http://localhost:9200")
    index_name = "music_embeddings"
    create_elasticsearch_index(es, index_name, EMBEDDING_DIM)
    index_embeddings(es, index_name, embeddings_by_file)
    
    # Dự đoán với video từ YouTube
    youtube_url = "https://www.youtube.com/watch?v=s6Ff23-0wb8"  # Thay URL nếu cần
    print("Downloading audio from YouTube...")
    test_audio_path = download_audio_from_youtube(youtube_url, output_path="youtube_test.wav")
    query_embedding = sliding_window_embedding(test_audio_path, base_network)
    response = search_similar(es, index_name, query_embedding, top_k=5)
    print("\nSearch results for YouTube video:")
    for hit in response["hits"]["hits"]:
        print(f"File: {hit['_source']['file_path']}, Label: {hit['_source']['label']}, Score: {hit['_score']:.4f}")
