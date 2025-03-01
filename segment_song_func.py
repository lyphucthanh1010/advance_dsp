import os
import librosa
import soundfile as sf
import multiprocessing


audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}

def process_audio_file(args):
 
    source_file, destination_directory, segment_duration, step_duration = args
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    
    try:
        y, sr = librosa.load(source_file, sr=None)
    except Exception as e:
        print(f"Error loading {source_file}: {e}")
        return []
    
    segment_samples = int(sr * segment_duration)
    step_samples = int(sr * step_duration)
    segments_saved = []
    segment_index = 0

    for start in range(0, len(y) - segment_samples + 1, step_samples):
        segment = y[start:start + segment_samples]
        output_filename = f"{base_name}_segment_{segment_index}.wav"
        output_path = os.path.join(destination_directory, output_filename)
        try:
            sf.write(output_path, segment, sr)
            segments_saved.append(output_path)
            segment_index += 1
        except Exception as e:
            print(f"Error saving segment {output_path}: {e}")
            continue

    return segments_saved
