import os
import argparse
import multiprocessing
import librosa
import soundfile as sf

def convert_mp3_to_wav(source_directory, destination_directory):
    """
    Chuyển đổi tất cả các file MP3 trong source_directory sang WAV,
    lưu vào destination_directory theo cùng cấu trúc thư mục.
    """
    converted_files = []
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if file.lower().endswith(".mp3"):
                source_file_path = os.path.join(root, file)
                # Lấy đường dẫn tương đối so với thư mục nguồn
                relative_path = os.path.relpath(root, source_directory)
                # Tạo thư mục đích tương ứng
                dest_folder = os.path.join(destination_directory, relative_path)
                os.makedirs(dest_folder, exist_ok=True)
                base_name = os.path.splitext(file)[0]
                destination_file_path = os.path.join(dest_folder, base_name + ".wav")
                try:
                    y, sr = librosa.load(source_file_path, sr=None)
                    sf.write(destination_file_path, y, sr)
                    converted_files.append(destination_file_path)
                    print(f"Chuyển đổi thành công: {source_file_path} -> {destination_file_path}")
                except Exception as e:
                    print(f"Lỗi khi chuyển đổi {source_file_path}: {e}")
    return converted_files

def process_audio_file(args):
    """
    Thực hiện segmentation cho một file audio.
    args: (audio_file, dest_folder, segment_duration, step_duration)
    """
    audio_file, dest_folder, segment_duration, step_duration = args
    segments = []
    try:
        y, sr = librosa.load(audio_file, sr=None)
        total_samples = len(y)
        segment_samples = int(segment_duration * sr)
        step_samples = int(step_duration * sr)
        count = 0
        start = 0
        while start + segment_samples <= total_samples:
            segment = y[start:start + segment_samples]
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            segment_file_name = f"{base_name}_segment{count}.wav"
            segment_file_path = os.path.join(dest_folder, segment_file_name)
            sf.write(segment_file_path, segment, sr)
            segments.append(segment_file_path)
            print(f"Segment thành công: {segment_file_path}")
            count += 1
            start += step_samples
    except Exception as e:
        print(f"Lỗi khi segment {audio_file}: {e}")
    return segments

def segment_audio_files(source_directory, destination_directory, segment_duration=10, step_duration=2):
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
    pool_args = []

    for root, dirs, files in os.walk(source_directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in audio_extensions:
                relative_path = os.path.relpath(root, source_directory)
                dest_folder = os.path.join(destination_directory, relative_path)
                os.makedirs(dest_folder, exist_ok=True)
                if ext == ".mp3":
                    # Với file mp3, sử dụng file wav đã chuyển đổi trong thư mục đích
                    base_name = os.path.splitext(file)[0]
                    input_file = os.path.join(dest_folder, base_name + ".wav")
                    if not os.path.exists(input_file):
                        print(f"Không tìm thấy file chuyển đổi cho {file}, bỏ qua segmentation.")
                        continue
                else:
                    # Các file audio khác được segment trực tiếp từ nguồn
                    input_file = os.path.join(root, file)
                pool_args.append((input_file, dest_folder, segment_duration, step_duration))
                print(f"Chuẩn bị segment: {input_file}")

    all_segments = []
    with multiprocessing.Pool() as pool:
        results = pool.map(process_audio_file, pool_args)
        for segs in results:
            all_segments.extend(segs)
    return all_segments

def main():
    parser = argparse.ArgumentParser(
        description="Tool chuyển đổi MP3 sang WAV và segment các file audio, giữ nguyên cấu trúc thư mục."
    )
    parser.add_argument("source_directory", help="Đường dẫn thư mục nguồn")
    parser.add_argument("destination_directory", help="Đường dẫn thư mục đích")
    parser.add_argument("--segment_duration", type=float, default=10,
                        help="Thời gian mỗi segment (giây), mặc định 10 giây")
    parser.add_argument("--step_duration", type=float, default=2,
                        help="Khoảng cách giữa các segment (giây), mặc định 2 giây")
    args = parser.parse_args()

    print("Bắt đầu chuyển đổi các file MP3 sang WAV...")
    converted_files = convert_mp3_to_wav(args.source_directory, args.destination_directory)
    print(f"\nTổng số file chuyển đổi: {len(converted_files)}\n")

    print("Bắt đầu segment các file audio...")
    segments = segment_audio_files(args.source_directory, args.destination_directory,
                                   args.segment_duration, args.step_duration)
    print(f"\nTổng số segment được tạo: {len(segments)}")
    for seg in segments:
        print(seg)

if __name__ == '__main__':
    main()
