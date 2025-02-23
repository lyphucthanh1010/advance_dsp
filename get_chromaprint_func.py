from flask import Flask, request, jsonify
import os
import json
import shutil
from essentia.standard import MonoLoader, Chromaprinter
import acoustid as ai
import multiprocessing

def get_chromaprints(audio_path, is_save_res=True):
    output_directory = "dataset"
    if is_save_res and not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    all_chromaprint_segments = [] 

    for root, dirs, files in os.walk(audio_path):
        for filename in files:
            if filename.endswith(".mp3"):
                audio_file = os.path.join(root, filename)
                audio = MonoLoader(filename=audio_file)()
                fp_full_char = Chromaprinter()(audio)
                fp_full = ai.chromaprint.decode_fingerprint(fp_full_char.encode())[0]

                chromaprint_segments = [fp_full[i:i+30] for i in range(0, len(fp_full), 30)]
                
                if len(chromaprint_segments) == 1 and len(chromaprint_segments[0]) < 30:
                    padding_length = 30 - len(chromaprint_segments[0])
                    chromaprint_segments[0] += [0] * padding_length
                elif len(chromaprint_segments) > 1 and len(chromaprint_segments[-1]) != 30:
                    del chromaprint_segments[-1]

                label = os.path.splitext(os.path.basename(audio_file))[0]

                if is_save_res:
                    output_json_path = os.path.join(output_directory, os.path.basename(audio_file) + '.json')
                    with open(output_json_path, 'w') as json_file:
                        data = {"chromaprints": chromaprint_segments, "label": label}
                        json.dump(data, json_file, indent=2)

                all_chromaprint_segments.extend(chromaprint_segments)
    return all_chromaprint_segments

def get_all_chromaprints(file_paths):
    print("Danh sách thư mục được xử lý:", file_paths)
    with multiprocessing.Pool() as pool:
        pool.map(get_chromaprints, [audio_path for audio_path in file_paths]) 