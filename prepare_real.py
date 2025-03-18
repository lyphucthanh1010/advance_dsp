import os
import multiprocessing
import essentia.standard as es
import json
import acoustid as ai
import glob

fs = 44100

def get_chromaprints(audio_path):
    audio = es.MonoLoader(filename=audio_path, sampleRate=fs)()
    fp_full_char = es.Chromaprinter()(audio)
    fp_full = ai.chromaprint.decode_fingerprint(fp_full_char.encode())[0]
    return fp_full, len(audio)

def get_chromaprints_from_subfolders(input_directory,is_save_res=True):
    output_directory = "datasetreal"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if is_save_res:
        audio_files = glob.glob(os.path.join(input_directory, '**/*.wav'), recursive=True)

        for audio_path in audio_files:
            audio = es.MonoLoader(filename=audio_path)()  

            fp_full_char = es.Chromaprinter()(audio)
            fp_full = ai.chromaprint.decode_fingerprint(fp_full_char.encode())[0]

            file_name = os.path.splitext(os.path.basename(audio_path))[0]

            output_json_path = os.path.join(output_directory, file_name + '.json')

            with open(output_json_path, 'w') as json_file:
                result = {"chromaprints": fp_full, "label": file_name}
                json.dump(result, json_file, indent=2)
    return fp_full, len(audio)


def get_all_chromaprints(file_paths):
    with multiprocessing.Pool() as pool:
        pool.map(get_chromaprints_from_subfolders, [audio_path for audio_path in file_paths])

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

