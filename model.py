import os
import json
import numpy as np
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Reshape, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend
# -------------------------------
# 1. Load dữ liệu chromaprints từ file JSON sử dụng multiprocessing

chromaprints_dir = "music_dataset_chromaprints"  # Thay bằng đường dẫn thư mục chứa file JSON của bạn

def load_json_file(file_path):
    """
    Hàm này load file JSON, trích xuất key "chromaprints" và "label".
    Nếu thành công, trả về tuple (features, label), trong đó features là numpy array.
    Nếu gặp lỗi hoặc thiếu key, trả về None.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chromaprints = data.get("chromaprints", None)
        label = data.get("label", None)
        if chromaprints is not None and label is not None:
            return (np.array(chromaprints), label)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return None

# Duyệt qua toàn bộ các file JSON trong chromaprints_dir
json_files = []
for root, dirs, files in os.walk(chromaprints_dir):
    for file in files:
        if file.lower().endswith(".json"):
            json_files.append(os.path.join(root, file))

print("Total JSON files found:", len(json_files))

# Sử dụng multiprocessing để load các file song song
with Pool() as pool:
    results = pool.map(load_json_file, json_files)

# Lọc kết quả hợp lệ (không None)
results = [r for r in results if r is not None]
features_list = [r[0] for r in results]
labels_list = [r[1] for r in results]

print("Total JSON files loaded:", len(features_list))

# -------------------------------
# 2. Padding các ma trận chromaprints
# Mỗi file có chromaprints dạng (num_segments, 30), nhưng num_segments có thể khác nhau.
target_seg_length = 30

# Điều chỉnh các mảng đặc trưng sao cho có số cột bằng target_seg_length
adjusted_features = []
for feat in features_list:
    # Nếu mảng 1 chiều, reshape thành (1, len)
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    num_segments, seg_length = feat.shape
    if seg_length < target_seg_length:
        # Pad bên phải để đạt target_seg_length
        pad_width_col = target_seg_length - seg_length
        feat = np.pad(feat, pad_width=((0, 0), (0, pad_width_col)), mode='constant')
    elif seg_length > target_seg_length:
        # Crop lại thành target_seg_length
        feat = feat[:, :target_seg_length]
    adjusted_features.append(feat)

# Cập nhật max_segments dựa trên các mảng đã điều chỉnh
max_segments = max(feat.shape[0] for feat in adjusted_features)

# Padding theo chiều hàng để có cùng số dòng cho tất cả các mảng
padded_features = []
for feat in adjusted_features:
    num_segments, seg_length = feat.shape  # seg_length bây giờ = target_seg_length
    pad_amount = max_segments - num_segments
    feat_padded = np.pad(feat, pad_width=((0, pad_amount), (0, 0)), mode='constant')
    # Thêm dimension channel => shape: (max_segments, target_seg_length, 1)
    feat_padded = np.expand_dims(feat_padded, axis=-1)
    padded_features.append(feat_padded)

X = np.array(padded_features)
print("Features shape:", X.shape) # (num_samples, max_segments, 30, 1)

# -------------------------------
# 3. Tạo bảng ánh xạ nhãn và chuyển đổi nhãn sang số
unique_labels = sorted(list(set(labels_list)))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
print("Unique labels:", unique_labels)

y_indices = np.array([label_to_idx[label] for label in labels_list])

# -------------------------------
# 4. Chia dữ liệu thành train (80%), validation (10%), test (10%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_indices, test_size=0.20, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Train samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
print("Test samples:", X_test.shape[0])

# Chuyển nhãn sang one-hot encoding
num_classes = len(unique_labels)
# y_train_cat = to_categorical(y_train, num_classes=num_classes)
# y_val_cat   = to_categorical(y_val, num_classes=num_classes)
# y_test_cat  = to_categorical(y_test, num_classes=num_classes)

# -------------------------------
# 5. Xây dựng mô hình CNN+LSTM cho chromaprints
def create_cnn_lstm_branch_single(input_shape):
    """
    Xây dựng một nhánh CNN+LSTM cho đặc trưng chromaprints.
    Input shape: (max_segments, 30, 1)
    """
    inp = Input(shape=input_shape)
    
    # Conv Block 1
    x = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    x = Dropout(0.25)(x)
    
    # Conv Block 2
    x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    x = Dropout(0.25)(x)
    
    # Conv Block 3
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    x = Dropout(0.25)(x)
    
    # Conv Block 4
    x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    x = Dropout(0.25)(x)
    
    # Conv Block 5: Pooling lớn để giảm kích thước không gian mạnh mẽ
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(4,4), strides=(4,4), padding='same')(x)
    x = Dropout(0.25)(x)
    
    conv_shape = backend.int_shape(x)
    h = conv_shape[1]
    w = conv_shape[2]
    c = conv_shape[3]
    x = Reshape((w, h * c))(x)
    
    # # LSTM layers: 3 lớp (2 trả về chuỗi, 1 trả về vector cố định)
    # x = LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True)(x)
    # x = LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True)(x)
    # x = LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=False)(x)
    
    return inp, x

input_shape = X_train.shape[1:]  # (max_segments, 30, 1)
inp, branch = create_cnn_lstm_branch_single(input_shape)
x = Dense(units=32, activation='relu')(branch)
output = Dense(units=num_classes, activation='softmax')(x)
model = Model(inputs=inp, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# 6. Huấn luyện và đánh giá mô hình
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=70,
    validation_data=(X_val, y_val)
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
