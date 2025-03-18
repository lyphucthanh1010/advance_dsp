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
                # Calculate the number of samples per segment for the specified segment length
                samples_per_segment = segment_length * 44100
                slide_samples = slide_length * 44100

                # Segment the audio with sliding window
                start_sample = 0
                segment_index = 0
                while start_sample + samples_per_segment <= len(audio):
                    segment = audio[start_sample:start_sample + samples_per_segment]

                    # Get the class label
                    class_label = os.path.splitext(audio_file)[0]

                    # Count the number of segments for this class
                    samples_per_class[class_label] = samples_per_class.get(class_label, 0) + 1

                    # Create a folder for segmented audio if it doesn't exist
                    segmented_folder = os.path.join(output_folder, class_label)
                    os.makedirs(segmented_folder, exist_ok=True)

                    # Save segment into segmented folder
                    segment_filename = os.path.join(segmented_folder, f"{class_label}_{segment_index}.wav")
                    es.MonoWriter(filename=segment_filename)(segment)

                    print(f"Segment {segment_index} created for {audio_file}")

                    # Move to the next segment with sliding window
                    start_sample += slide_samples
                    segment_index += 1

            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

    # Perform data augmentation if necessary
    for class_label, num_samples in samples_per_class.items():
        if num_samples < min_samples_per_class:
            print(f"Class {class_label} has only {num_samples} samples. Performing data augmentation.")


audio_folder = "music_dataset_wav"
output_folder = "dataset_segment"
segment_length = 10  # in seconds
slide_length = 5 # in seconds
min_samples_per_class = 40
segment_audio(audio_folder, output_folder, segment_length, slide_length, min_samples_per_class)