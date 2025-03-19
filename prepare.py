import os
import json
import numpy as np
import multiprocessing
import essentia.standard as es
import acoustid as ai

import os
import json
import multiprocessing
import essentia.standard as es
import acoustid as ai

fs = 44100

# def get_chromaprints_to_test(audio_path):
#     audio = es.MonoLoader(filename=audio_path, sampleRate=fs)()
#     volume_adjusted_audio = es.EqualLoudness()(audio) 
#     fp_full_char = es.Chromaprinter()(volume_adjusted_audio)
#     fp_full = ai.chromaprint.decode_fingerprint(fp_full_char.encode())[0]
#     chromaprint_segments = [fp_full[i:i+100] for i in range(0, len(fp_full), 100)]
#     if len(chromaprint_segments[-1]) != 100:
#         del chromaprint_segments[-1]
#     return chromaprint_segments

def get_chromaprints(audio_path, is_save_res=True):
    output_directory = "dataset"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    
    all_chromaprint_segments = []
    
    if os.path.isfile(audio_path):
        files = [audio_path]
    else:
        files = [os.path.join(audio_path, f) for f in os.listdir(audio_path)
                 if f.endswith(".wav") and os.path.isfile(os.path.join(audio_path, f))]
    
    for audio_file in files:
        audio = es.MonoLoader(filename=audio_file)()
        fp_full_char = es.Chromaprinter()(audio)
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



def get_chromaprints_to_test(audio_path, is_save_res=True, is_return_len=True):
    output_directory = "dataset"
    audio = es.MonoLoader(filename=audio_path, sampleRate=fs)()

    fp_full_char = es.Chromaprinter()(audio)
    fp_full = ai.chromaprint.decode_fingerprint(fp_full_char.encode())[0]
   
    chromaprint_segments = [fp_full[i:i+30] for i in range(0, len(fp_full), 30)]
        
    if len(chromaprint_segments) == 1 and len(chromaprint_segments[0]) < 30:
        padding_length = 30 - len(chromaprint_segments[0])
        chromaprint_segments[0] += [0] * padding_length
    elif len(chromaprint_segments) > 1 and len(chromaprint_segments[-1]) != 30:
        del chromaprint_segments[-1]
        
    label = os.path.splitext(os.path.basename(audio_path))[0]
    
    if is_save_res:
        output_json_path = os.path.join(output_directory, os.path.basename(audio_path) + '.json')
        with open(output_json_path, 'w') as json_file:
            data = {"chromaprints": chromaprint_segments, "label": label}
            json.dump(data, json_file, indent=2)
            
    if is_return_len:
        return chromaprint_segments, len(audio)
    else:
        return chromaprint_segments


def get_all_chromaprints(file_paths):
    # print(file_paths)
    with multiprocessing.Pool() as pool:
        pool.map(get_chromaprints, [audio_path for audio_path in file_paths])   


def process_audio(dir_path):

    file_paths = []

    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        file_paths.append(file_path)
    get_all_chromaprints(file_paths)


def main():
    audio_folder = "dataset_segment"
    process_audio(audio_folder)
    print("Dataset prepared successfully.")

if __name__ == "__main__":
    main()

    
