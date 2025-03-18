import os
import shutil
import datetime
from flask import Flask, request, jsonify, Response
import yt_dlp
import pika
import json
import multiprocessing
import numpy as np
from model_old import process_real_data_and_copy_files, empty_directory
from repository import savePredictionsResult
from prepare import get_chromaprints_to_test
from prepare_real import get_chromaprints
from task import chromaprints_comparations
from pytube import YouTube
app = Flask(__name__)
dataset_model_path ='dataset'
datafull_path ='datasetreal'
dataset_path = "copied_json_files"
model_path ='model.pkl'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        audio_file_path = os.path.join(
            "uploads", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        file.save(audio_file_path)

        results, audio_duration = process_video_source(audio_file_path)

        json_predictions_response = {}

        print("Result:", results)

        if results:
            grouped_results = {}
            for start, end, class_name, confidence in results:
                if round(confidence, 2) >= 0.70 and start >= 0:
                    song_and_artist = class_name.rsplit('_', 1)[0]
                    if song_and_artist not in grouped_results:
                        grouped_results[song_and_artist] = {"start": start, "end": end, "confidence": confidence}
                    else:
                        if start < grouped_results[song_and_artist]["start"]:
                            grouped_results[song_and_artist]["start"] = start
                        if end > grouped_results[song_and_artist]["end"]:
                            grouped_results[song_and_artist]["end"] = end
                        if confidence > grouped_results[song_and_artist]["confidence"]:
                            grouped_results[song_and_artist]["confidence"] = confidence

            for song_and_artist, info in grouped_results.items():
                json_predictions_response[f"{info['start']}-{info['end']}"] = {
                    "label": song_and_artist,
                    "probability": round(info['confidence'], 2),
                }
        # os.remove(audio_file_path)
        print(json_predictions_response)
        if json_predictions_response:
            return jsonify({"data": json_predictions_response}), 200

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
        return jsonify({"data": results}), 200

    return jsonify({"msg": "Cannot predict this song"}), 200


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
        json_predictions_response = {}
        if results:
            grouped_results = {}
            for start, end, class_name, confidence in results:
                if round(confidence, 2) >= 0.70 and start >= 0:
                    song_and_artist = class_name.rsplit('_', 1)[0]
                    if song_and_artist not in grouped_results:
                        grouped_results[song_and_artist] = {"start": start, "end": end, "confidence": confidence}
                    else:
                        if start < grouped_results[song_and_artist]["start"]:
                            grouped_results[song_and_artist]["start"] = start
                        if end > grouped_results[song_and_artist]["end"]:
                            grouped_results[song_and_artist]["end"] = end
                        if confidence > grouped_results[song_and_artist]["confidence"]:
                            grouped_results[song_and_artist]["confidence"] = confidence

            for song_and_artist, info in grouped_results.items():
                json_predictions_response[f"{info['start']}-{info['end']}"] = {
                    "label": song_and_artist,
                    "probability": round(info['confidence'], 2),
                }
        print(f"Processed video {video_id}")
        empty_directory(directory=dataset_path)
        return json_predictions_response
    else:
        empty_directory(directory=dataset_path)
        return {"error": "Audio file not found"}



def process_song(song, source_20bit):
    json_file_path = os.path.join(dataset_path, song)
    with open(json_file_path, 'r') as json_file:
        full_20bit = json.load(json_file)
        chromaprint = full_20bit['chromaprints']
        res = chromaprints_comparations(fp_full=source_20bit, fp_frag= chromaprint )
        if res is not None:
            file_name = os.path.splitext(song)[0]
            start_dur, end_dur, prob = res
            return start_dur, end_dur, file_name, prob
        else:
            return None


def process_video_source(audio_file_path):
    temp_dir = os.path.join(
        "temp", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(temp_dir, exist_ok=True)

    try:
        source_20bit = get_chromaprints_to_test(audio_file_path, is_save_res=False, is_return_len=False)
        result = []
        process_real_data_and_copy_files(dataset_model_path, model_path, source_20bit, data_full=datafull_path)
        data_20bit , audio_len = get_chromaprints(audio_file_path)
        with multiprocessing.Pool() as pool:
            song_files = [song for song in os.listdir(dataset_path) if song.endswith('.json')]
            song_args = [(song_file, data_20bit) for song_file in song_files] 
            result = pool.starmap(process_song, song_args) 

        return result, audio_len / 44100

    except Exception as e:
        print("Error:", e)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)       

def process_message_queue():
    try:
        # Connect to RabbitMQ
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host='localhost', port=5672, credentials=pika.PlainCredentials('admin', 'GyV3UeErhx')))
        channel = connection.channel()

        queue_name = 'videos_queue'

        channel.queue_declare(queue=queue_name, durable=True)

        channel.basic_qos(prefetch_count=10)

        def callback(ch, method, properties, body):
            json_string = body.decode()
            print(json_string)
            json_data = json.loads(json_string)

            video_id = json_data.get("data")

            process_audio_youtube(video_id=video_id, is_upload_db=True)

            ch.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_consume(queue=queue_name,
                              on_message_callback=callback)

        print('Waiting for messages...')
        channel.start_consuming()

    except Exception as e:
        print(f"Error popping Youtube video from message queue: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav'}


if __name__ == '__main__':
    # process_message_queue()
    app.run(debug=True, port= 8000)