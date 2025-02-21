import os
import shutil

audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}

source_directory = r'Music' 

destination_directory = r'music_dataset'

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

for root, dirs, files in os.walk(source_directory):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in audio_extensions:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_directory, file)
            
            if os.path.exists(destination_file):
                base, extension = os.path.splitext(file)
                count = 1
                new_file = f"{base}_{count}{extension}"
                destination_file = os.path.join(destination_directory, new_file)
                while os.path.exists(destination_file):
                    count += 1
                    new_file = f"{base}_{count}{extension}"
                    destination_file = os.path.join(destination_directory, new_file)
            
            shutil.move(source_file, destination_file)
            print(f"Đã di chuyển: {source_file} -> {destination_file}")
