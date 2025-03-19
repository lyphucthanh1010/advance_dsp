import os
import json
import essentia.standard as es

def segment_audio(audio_folder, output_folder, segment_length=5, slide_length=2, min_samples_per_class=50):
    samples_per_class = {}

    for audio_file in os.listdir(audio_folder)[:50]:
        audio_file_path = os.path.join(audio_folder, audio_file)
        if os.path.isfile(audio_file_path):
            try:
                audio = es.MonoLoader(filename=audio_file_path)()
                samples_per_segment = segment_length * 44100
                slide_samples = slide_length * 44100

                start_sample = 0
                segment_index = 0
                while start_sample + samples_per_segment <= len(audio):
                    segment = audio[start_sample:start_sample + samples_per_segment]

                    class_label = os.path.splitext(audio_file)[0]

                    samples_per_class[class_label] = samples_per_class.get(class_label, 0) + 1

                    segmented_folder = os.path.join(output_folder, class_label)
                    os.makedirs(segmented_folder, exist_ok=True)

                    segment_filename = os.path.join(segmented_folder, f"{class_label}_{segment_index}.wav")
                    es.MonoWriter(filename=segment_filename)(segment)

                    print(f"Segment {segment_index} created for {audio_file}")

                    start_sample += slide_samples
                    segment_index += 1

            except Exception as e:
                print(f"Error processing {audio_file}: {e}")


audio_folder = "music_dataset_wav"
output_folder = "dataset_segment"
segment_length = 10  
slide_length = 5
min_samples_per_class = 40
segment_audio(audio_folder, output_folder, segment_length, slide_length, min_samples_per_class)