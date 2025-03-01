from flask import Flask, request, jsonify
import os
import json
import shutil
from essentia.standard import MonoLoader, Chromaprinter
import chromaprint
import multiprocessing

def get_chromaprints(args):

    file_path, destination_directory = args
    try:
        audio = MonoLoader(filename=file_path)()
        fp_full_char = Chromaprinter()(audio)
        fp_full = chromaprint.decode_fingerprint(fp_full_char.encode())[0]
        
        chromaprint_segments = [fp_full[i:i+30] for i in range(0, len(fp_full), 30)]
        
        if len(chromaprint_segments) == 1 and len(chromaprint_segments[0]) < 30:
            padding_length = 30 - len(chromaprint_segments[0])
            chromaprint_segments[0] += [0] * padding_length
        elif len(chromaprint_segments) > 1 and len(chromaprint_segments[-1]) != 30:
            del chromaprint_segments[-1]
        
        label = os.path.splitext(os.path.basename(file_path))[0]
        
        output_file = os.path.join(destination_directory, os.path.basename(file_path) + '.json')
        with open(output_file, 'w') as f:
            data = {
                "chromaprints": chromaprint_segments,
                "label": label
            }
            json.dump(data, f, indent=2)
        return output_file
    except Exception as e:
        print(f"Lỗi khi xử lý {file_path}: {e}")
        return None

def get_all_chromaprints(file_paths):
    print("Danh sách thư mục được xử lý:", file_paths)
    with multiprocessing.Pool() as pool:
        pool.map(get_chromaprints, [audio_path for audio_path in file_paths]) 