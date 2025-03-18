import os
import shutil
import datetime
from flask import Flask, request, jsonify, Response
import yt_dlp
import pika
import json
import multiprocessing
import numpy as np
from model import process_real_data_and_copy_files, empty_directory
from repository import savePredictionsResult
from prepare import get_chromaprints_to_test
from prepare_real import get_chromaprints
from task import chromaprints_comparations
from pytube import YouTube 
app = Flask(__name__)

dataset_model_path = 'dataset'
datafull_path = 'datasetreal'
dataset_path = "copied_json_files"
model_path = 'model.pkl'
THRESHOLD_CONFIDENCE = 0.85  
def fallback_search_datasetreal(query_chromaprints):
    fallback_results = []
    for file in os.listdir(datafull_path):
        if file.endswith('.json'):
            file_path = os.path.join(datafull_path, file)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            dataset_chromaprints = json_data.get("chromaprints", [])
            label = json_data.get("label", "")
            res = chromaprints_comparations(fp_full=query_chromaprints, fp_frag=dataset_chromaprints)
            if res is not None:
                fallback_results.append((res[0], res[1], label, res[2]))
    return fallback_results

def process_song(song, source_20bit):
    json_file_path = os.path.join(dataset_path, song)
    with open(json_file_path, 'r') as json_file:
        full_20bit = json.load(json_file)
    chromaprint = full_20bit['chromaprints']
    res = chromaprints_comparations(fp_full=source_20bit, fp_frag=chromaprint)
    if res is not None:
        file_name = os.path.splitext(song)[0]
        start_dur, end_dur, prob = res
        return start_dur, end_dur, file_name, prob
    else:
        return None

def process_video_source(audio_file_path):
    temp_dir = os.path.join("temp", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(temp_dir, exist_ok=True)
    try:
        source_20bit = get_chromaprints_to_test(audio_file_path, is_save_res=False, is_return_len=False)
        
        process_real_data_and_copy_files(dataset_model_path, model_path, source_20bit, data_full=datafull_path)
        
        data_20bit, audio_len = get_chromaprints(audio_file_path)
        
        with multiprocessing.Pool() as pool:
            song_files = [song for song in os.listdir(dataset_path) if song.endswith('.json')]
            song_args = [(song_file, data_20bit) for song_file in song_files]
            result = pool.starmap(process_song, song_args)
        
        filtered_results = [r for r in result if r is not None]
        if filtered_results:
            return filtered_results, audio_len / 44100
        else:
            fallback_results = fallback_search_datasetreal(data_20bit)
            if fallback_results:
                return fallback_results, audio_len / 44100
            else:
                return None, audio_len / 44100
    except Exception as e:
        print("Error:", e)
        return None, None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def group_predictions(predictions):
    grouped = {}
    for pred in predictions:
        start, end, label, conf = pred
        base_name = label.rsplit('_', 1)[0]
        if base_name not in grouped:
            grouped[base_name] = {"start": start, "end": end, "confidence": conf}
        else:
            grouped[base_name]["start"] = min(grouped[base_name]["start"], start)
            grouped[base_name]["end"] = max(grouped[base_name]["end"], end)
            grouped[base_name]["confidence"] = max(grouped[base_name]["confidence"], conf)
    result = []
    for base, info in grouped.items():
        result.append((info["start"], info["end"], base, info["confidence"]))
    return result

def process_audio_youtube(video_id, is_upload_db=False):
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    audio_file_path = os.path.join("uploads", f"{video_id}.mp3")
    print(f"Processing Video Id: {video_id}")
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join("uploads", f"{video_id}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '320',
            }],
            'ffmpeg_location': '/usr/bin'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
        print('Download successfully')
    except Exception as e:
        return {"error": f"Failed to download audio from YouTube: {str(e)}"}
    
    if os.path.exists(audio_file_path):
        results, audio_duration = process_video_source(audio_file_path)
        print(f"Results: {results}")
        if results:
            filtered_results = [r for r in results if r[3] >= THRESHOLD_CONFIDENCE]
            if filtered_results:
                grouped = group_predictions(filtered_results)
                predictions = [
                    {"start": r[0], "end": r[1], "label": r[2], "confidence": round(r[3], 2)}
                    for r in grouped
                ]
                return {"predictions": predictions}
            else:
                return {"msg": "No predictions with confidence >= 0.85"}
        else:
            return {"msg": "Cannot predict this song"}
    else:
        return {"error": "Audio file not found"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav'}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        audio_file_path = os.path.join("uploads", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        file.save(audio_file_path)
        results, audio_duration = process_video_source(audio_file_path)
        print("Result:", results)
        if results:
            filtered_results = [r for r in results if r[3] >= THRESHOLD_CONFIDENCE]
            if filtered_results:
                grouped = group_predictions(filtered_results)
                predictions = [
                    {"start": r[0], "end": r[1], "label": r[2], "confidence": round(r[3], 2)}
                    for r in grouped
                ]
                return jsonify({"predictions": predictions}), 200
            else:
                return jsonify({"msg": "No predictions with confidence >= 0.85"}), 200
    return jsonify({"msg": "Processed"}), 200

@app.route('/predict_from_youtube', methods=['POST'])
def predict_from_youtube():
    if 'video_id' not in request.form:
        return jsonify({"error": "No video ID provided"}), 400
    video_id = request.form['video_id']
    if not video_id:
        return jsonify({"error": "Empty video ID provided"}), 400
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    results = process_audio_youtube(video_id)
    if results:
        if isinstance(results, Response):
            return results
        return jsonify(results), 200
    return jsonify({"msg": "Cannot predict this song"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=8000)
