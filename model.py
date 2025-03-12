import os
import json
import numpy as np
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization
import tensorflow as tf

# -------------------------------
# 1. Load dữ liệu chromaprints từ file JSON sử dụng multiprocessing
chromaprints_dir = "music_dataset_chromaprints"  # Giữ nguyên đường dẫn cũ

def load_json_file(file_path):
    """
    Load file JSON, trích xuất key "chromaprints" và lấy nhãn từ cấu trúc thư mục.
    Ta tính trung bình các segment để tạo vector đặc trưng cố định.
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
    # Bỏ qua nếu feat là mảng rỗng
    if feat.size == 0:
        continue
    # Nếu feat là mảng 1 chiều, reshape thành (1, len)
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    # Kiểm tra lại kích thước sau reshape
    if feat.shape[0] == 0 or feat.shape[1] == 0:
        continue
    num_segments, seg_length = feat.shape
    if seg_length < target_seg_length:
        pad_width = target_seg_length - seg_length
        feat = np.pad(feat, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif seg_length > target_seg_length:
        feat = feat[:, :target_seg_length]
    # Tính trung bình các segment theo axis=0 => vector có shape (30,)
    avg_feat = np.mean(feat, axis=0)
    X_list.append(avg_feat)
    y_list.append(label)

X = np.array(X_list)
print("Features shape:", X.shape)  # (num_samples, 30)
print("Total samples after filtering:", X.shape[0])
# Đồng thời cập nhật nhãn
y_indices = np.array([label_to_idx[label] for label in y_list]) if 'label_to_idx' in globals() else None

# Nếu bạn chưa tạo bảng ánh xạ nhãn, tạo ở bước 3:
if y_indices is None:
    unique_labels = sorted(list(set(y_list)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    print("Unique labels:", unique_labels)
    y_indices = np.array([label_to_idx[label] for label in y_list])

# -------------------------------
# 3. Chuẩn hóa dữ liệu (Min-Max scaling per sample)
X_min = np.min(X, axis=1, keepdims=True)
X_max = np.max(X, axis=1, keepdims=True)
X = (X - X_min) / (X_max - X_min + 1e-8)

# -------------------------------
# 4. Chia dữ liệu: Sử dụng toàn bộ dữ liệu cho huấn luyện, và tạo tập test từ tập train sao chép & xáo trộn.
X_train = X  # 100% dữ liệu cho training
y_train = y_indices
num_classes = len(set(y_list))
y_train_cat = to_categorical(y_train, num_classes=num_classes)

# Tạo tập test là bản sao của train, sau đó xáo trộn ngẫu nhiên
X_test = np.copy(X_train)
y_test = np.copy(y_train)
perm = np.random.permutation(len(X_test))
X_test = X_test[perm]
y_test = y_test[perm]
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# -------------------------------
# 5. Reshape dữ liệu cho CNN 1D: mỗi vector có 30 chiều -> (30, 1)
X_train = X_train.reshape(-1, target_seg_length, 1)
X_test = X_test.reshape(-1, target_seg_length, 1)
print("Reshaped training data:", X_train.shape)
print("Reshaped test data:", X_test.shape)

# -------------------------------
# 6. Xây dựng mô hình CNN 1D cho chromaprints
model = Sequential([
    Input(shape=(target_seg_length, 1)),
    Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    tf.keras.layers.GlobalAveragePooling1D(),
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
    batch_size=32,
    epochs=100,
    callbacks=[early_stop]
)

train_loss, train_accuracy = model.evaluate(X_train, y_train_cat)
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
print("Train Loss:", train_loss)
print("Train Accuracy:", train_accuracy)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
