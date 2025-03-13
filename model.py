import os
import json
import numpy as np
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense
import tensorflow as tf

# -------------------------------
# 1. Load dữ liệu chromaprints từ file JSON sử dụng multiprocessing
chromaprints_dir = "music_dataset_chromaprints"  # Giữ nguyên đường dẫn cũ
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Tắt GPU, chỉ sử dụng CPU

def load_json_file(file_path):
    """
    Load file JSON, trích xuất key "chromaprints" và lấy nhãn từ cấu trúc thư mục.
    Ta giữ nguyên ma trận đặc trưng (không tính trung bình).
    Nếu thành công, trả về tuple (features, label).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chromaprints = data.get("chromaprints", None)
        # Lấy nhãn từ cấu trúc thư mục: folder con đầu tiên của đường dẫn tương đối
        relative_path = os.path.relpath(file_path, chromaprints_dir)
        parts = relative_path.split(os.sep)
        label = parts[0] if parts else None
        if chromaprints is not None and label is not None:
            feat = np.array(chromaprints)
            return (feat, label)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return None

json_files = []
for root, dirs, files in os.walk(chromaprints_dir):
    for file in files:
        if file.lower().endswith(".json"):
            json_files.append(os.path.join(root, file))
print("Total JSON files found:", len(json_files))

with Pool() as pool:
    results = pool.map(load_json_file, json_files)
results = [r for r in results if r is not None]
features_list = [r[0] for r in results]
labels_list = [r[1] for r in results]
print("Total JSON files loaded:", len(features_list))

# -------------------------------
# 2. Điều chỉnh kích thước đặc trưng và lọc đồng thời nhãn:
# Giả sử mỗi file có chromaprints dạng (num_segments, seg_length) với seg_length không đồng nhất.
# Ta đặt target_seg_length = 30; nếu số cột nhỏ hơn, pad; nếu lớn hơn, crop.
target_seg_length = 30
X_list = []
y_list = []
for feat, label in zip(features_list, labels_list):
    if feat.size == 0:
        continue
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    if feat.shape[0] == 0 or feat.shape[1] == 0:
        continue
    num_segments, seg_length = feat.shape
    if seg_length < target_seg_length:
        pad_width = target_seg_length - seg_length
        feat = np.pad(feat, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif seg_length > target_seg_length:
        feat = feat[:, :target_seg_length]
    # Không tính trung bình, giữ nguyên ma trận có shape (num_segments, 30)
    X_list.append(feat)
    y_list.append(label)

if len(X_list) == 0:
    raise ValueError("Không có mẫu dữ liệu nào sau quá trình xử lý! Kiểm tra lại dữ liệu nguồn.")

# Tính số dòng tối đa (max_segments) để pad theo chiều số dòng
max_segments = max(feat.shape[0] for feat in X_list)
padded_features = []
for feat in X_list:
    num_segments, seg_length = feat.shape
    pad_amount = max_segments - num_segments
    feat_padded = np.pad(feat, pad_width=((0, pad_amount), (0, 0)), mode='constant')
    # Thêm dimension channel => shape: (max_segments, target_seg_length, 1)
    feat_padded = np.expand_dims(feat_padded, axis=-1)
    padded_features.append(feat_padded)

X = np.array(padded_features)
print("Padded features shape:", X.shape)  # (num_samples, max_segments, 30, 1)
print("Total samples after filtering:", X.shape[0])

# -------------------------------
# 3. Chuẩn hóa dữ liệu (Min-Max scaling per sample)
X_min = np.min(X, axis=(1,2,3), keepdims=True)
X_max = np.max(X, axis=(1,2,3), keepdims=True)
X = (X - X_min) / (X_max - X_min + 1e-8)

# -------------------------------
# 4. Tạo bảng ánh xạ nhãn và chuyển đổi nhãn sang số
unique_labels = sorted(list(set(y_list)))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
print("Unique labels:", unique_labels)
y_indices = np.array([label_to_idx[label] for label in y_list])

# -------------------------------
# 5. Sử dụng toàn bộ dữ liệu cho huấn luyện và tạo tập test bằng sao chép & xáo trộn.
X_train = X  # 100% dữ liệu dùng cho training
y_train = y_indices
num_classes = len(unique_labels)
y_train_cat = to_categorical(y_train, num_classes=num_classes)

X_test = np.copy(X_train)
y_test = np.copy(y_train)
perm = np.random.permutation(len(X_test))
X_test = X_test[perm]
y_test = y_test[perm]
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# -------------------------------
# 6. Xây dựng mô hình CNN 2D cho chromaprints
# Input shape: (max_segments, target_seg_length, 1)
input_shape = X_train.shape[1:]  # (max_segments, 30, 1)
model = Sequential([
    Input(shape=input_shape),
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1,2), padding='same'),
    Dropout(0.25),
    Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1,2), padding='same'),
    Dropout(0.25),
    Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1,2), padding='same'),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# 7. Huấn luyện mô hình với EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train_cat,
    batch_size=16,
    epochs=100,
    callbacks=[early_stop]
)

train_loss, train_accuracy = model.evaluate(X_train, y_train_cat)
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
print("Train Loss:", train_loss)
print("Train Accuracy:", train_accuracy)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
