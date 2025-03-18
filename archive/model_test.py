import os
import argparse
import numpy as np
import tensorflow as tf
from pytubefix import YouTube
from pytubefix.cli import on_progress
from pydub import AudioSegment
from essentia.standard import MonoLoader, Chromaprinter
import chromaprint
import librosa
import pickle
import json
import glob

# ---------- CONFIGURATION ----------
FINGERPRINT_SEGMENT_LENGTH = 30      # Số giá trị cho mỗi fingerprint chunk
SR = 22050                           # Sampling rate
LABEL_ENCODER_PICKLE = "label_encoder.pkl"
OUTPUT_FOLDER = "chromaprints_output"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Sử dụng CPU

# ---------- ALGORITHM FOR CHROMAPRINT COMPARISON ----------
def invert(arr):
    mapping = {}
    for i, a in enumerate(arr):
        mapping.setdefault(a, []).append(i)
    return mapping

def chromaprints_comparations(fp_full, fp_frag):
    mask = (1 << 20) - 1
    full_20bit = [x & mask for x in fp_full]
    short_20bit = [x & mask for x in fp_frag]

    common = set(full_20bit) & set(short_20bit)
    i_full_20bit = invert(full_20bit)
    i_short_20bit = invert(short_20bit)

    offsets = {}
    for a in common:
        for i in i_full_20bit[a]:
            for j in i_short_20bit[a]:
                o = i - j
                offsets[o] = offsets.get(o, 0) + 1

    popcnt_table_8bit = [0] * 256
    for i in range(256):
        popcnt_table_8bit[i] = (i & 1) + popcnt_table_8bit[i >> 1]

    def popcnt(x):
        return (popcnt_table_8bit[x & 0xFF] +
                popcnt_table_8bit[(x >> 8) & 0xFF] +
                popcnt_table_8bit[(x >> 16) & 0xFF] +
                popcnt_table_8bit[(x >> 24) & 0xFF])

    def ber(offset):
        errors = 0
        count = 0
        for a, b in zip(fp_full[offset:], fp_frag):
            errors += popcnt(a ^ b)
            count += 1
        return max(0.0, 1.0 - 2.0 * errors / (32.0 * count))

    matches = []
    for count, offset in sorted([(v, k) for k, v in offsets.items()], reverse=True)[:20]:
        matches.append((ber(offset), offset))
    matches.sort(reverse=True)
    score, best_offset = matches[0]
    offset_secs = int(best_offset * 0.1238)
    fp_duration = len(fp_frag) * 0.1238 + 2.6476
    return offset_secs, offset_secs + fp_duration, score

# ---------- DOWNLOAD AUDIO FROM YOUTUBE ----------
def download_audio_from_youtube(url, output_path="temp_audio.wav"):
    yt = YouTube(url, on_progress_callback=on_progress)
    audio_stream = yt.streams.filter(only_audio=True, mime_type="audio/mp4").first()
    if audio_stream is None:
        audio_stream = yt.streams.filter(only_audio=True).first()
    downloaded_file = audio_stream.download(filename="temp_audio")
    if not downloaded_file.endswith(".wav"):
        try:
            sound = AudioSegment.from_file(downloaded_file)
            wav_file = os.path.splitext(downloaded_file)[0] + ".wav"
            sound.export(wav_file, format="wav")
            os.remove(downloaded_file)
            downloaded_file = wav_file
        except Exception as e:
            print("Error converting downloaded file to WAV:", e)
    return downloaded_file

# ---------- EXTRACT FINGERPRINT SEGMENTS ----------
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
        seg_arrays = []
        for seg in segments:
            arr = np.array(seg, dtype=np.float32)
            if not np.all(arr == 0):
                seg_arrays.append(arr.reshape(FINGERPRINT_SEGMENT_LENGTH, 1, 1))
        return seg_arrays
    except Exception as e:
        print(f"Error extracting fingerprint segments from {file_path}: {e}")
        return []

def sliding_window_prediction(file_path, classifier_model):
    segments = extract_fp_segments_from_audio(file_path)
    if not segments:
        return None
    predictions = []
    for seg in segments:
        seg_exp = np.expand_dims(seg, axis=0)  # (1, FINGERPRINT_SEGMENT_LENGTH, 1, 1)
        pred = classifier_model.predict(seg_exp)
        predictions.append(pred[0])
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    return mean_pred

def predict_label(mean_pred, label_encoder):
    pred_index = np.argmax(mean_pred)
    predicted_label = label_encoder.inverse_transform([pred_index])[0]
    return predicted_label

def load_label_encoder(encoder_file=LABEL_ENCODER_PICKLE):
    with open(encoder_file, "rb") as f:
        label_encoder = pickle.load(f)
    print("Loaded LabelEncoder from", encoder_file)
    return label_encoder

def load_classifier_model(model_file="classifier_model.h5"):
    model = tf.keras.models.load_model(model_file)
    print("Loaded classifier model from", model_file)
    return model

def main(audio_source):
    # Load classifier model and LabelEncoder
    classifier_model = load_classifier_model("classifier_model.h5")
    label_encoder = load_label_encoder(LABEL_ENCODER_PICKLE)
    
    # Determine file path for test audio
    if audio_source.startswith("http"):
        print("Downloading audio from YouTube...")
        downloaded_audio = download_audio_from_youtube(audio_source, output_path="temp_audio.wav")
        print("Downloaded audio file:", downloaded_audio)
        file_path = downloaded_audio
    else:
        file_path = audio_source
    
    # Prediction from classifier
    mean_pred = sliding_window_prediction(file_path, classifier_model)
    if mean_pred is None:
        print("Failed to extract prediction from test audio.")
        return
    predicted_label = predict_label(mean_pred, label_encoder)
    print("Predicted label for audio:", predicted_label)
    print("Prediction distribution:", mean_pred)
    
    # Verification: Load a training fingerprint for the predicted label
    candidate_files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.json"))
    target_file = None
    for f in candidate_files:
        with open(f, "r") as jf:
            data = json.load(jf)
            if data["label"] == predicted_label:
                target_file = f
                fp_full = data["chromaprints"]
                break
    if target_file is None:
        print("No training fingerprint found for predicted label.")
    else:
        # Use the first fingerprint segment from the test audio as fp_frag
        fp_frag_list = extract_fp_segments_from_audio(file_path)
        if fp_frag_list:
            fp_frag = fp_frag_list[0].flatten().tolist()
            start_sec, end_sec, score = chromaprints_comparations(fp_full, fp_frag)
            print(f"Verification: Query appears at {start_sec} to {end_sec} seconds with score {score:.4f}")
        else:
            print("No valid fingerprint segment found for verification.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_source", type=str, required=True,
                        help="URL of YouTube video or path to a local audio file")
    args = parser.parse_args()
    main(args.audio_source)
