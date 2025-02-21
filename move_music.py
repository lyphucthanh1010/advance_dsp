import os
import shutil

# Danh sách các định dạng file âm thanh cần tìm
audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}

# Thư mục gốc để duyệt (có thể thay đổi theo nhu cầu)
source_directory = r'Music'  # Thay đổi đường dẫn cho phù hợp

# Thư mục đích để di chuyển các file âm thanh
destination_directory = r'music_dataset'  # Thay đổi đường dẫn cho phù hợp

# Tạo thư mục đích nếu nó không tồn tại
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Duyệt qua tất cả các thư mục con và file
for root, dirs, files in os.walk(source_directory):
    for file in files:
        # Lấy phần mở rộng file và chuyển về chữ thường để so sánh
        ext = os.path.splitext(file)[1].lower()
        if ext in audio_extensions:
            # Đường dẫn file nguồn
            source_file = os.path.join(root, file)
            # Đường dẫn file đích
            destination_file = os.path.join(destination_directory, file)
            
            # Nếu file đã tồn tại trong thư mục đích, đổi tên file để tránh ghi đè
            if os.path.exists(destination_file):
                base, extension = os.path.splitext(file)
                count = 1
                new_file = f"{base}_{count}{extension}"
                destination_file = os.path.join(destination_directory, new_file)
                while os.path.exists(destination_file):
                    count += 1
                    new_file = f"{base}_{count}{extension}"
                    destination_file = os.path.join(destination_directory, new_file)
            
            # Di chuyển file
            shutil.move(source_file, destination_file)
            print(f"Đã di chuyển: {source_file} -> {destination_file}")
