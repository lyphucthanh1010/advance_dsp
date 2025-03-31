# Music Copyright Detection API

## Giới thiệu

API này được xây dựng bằng Flask nhằm dự đoán vi phạm bản quyền trong âm nhạc thông qua một mô hình Keras đã được huấn luyện sẵn. API hỗ trợ nhận đầu vào là tệp âm thanh hoặc ID video YouTube và trả về kết quả dự đoán.

File model: model.pkl
Dataset: folder dataset
Dataset để test: folder datasetreal
## Cài đặt

1. **Tạo môi trường ảo (tuỳ chọn):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

## Chạy API

1. **Chạy file app.py:**

   ```bash
   python app.py
   ```

2. **Dự đoán file local (API /predict)**

   ```
   curl -X POST -F "file=@đường dẫn file" -w "Total time: %{time_total}\n" http://localhost:8000/predict
   ```

3. **Dự đoán video youtube (API /predict_from_youtube)**
   ```
   curl -X POST -F "video_id=video_id_của_youtube" --insecure -w "Total time: %{time_total}\n" http://localhost:8000/predict_from_youtube
   ```
