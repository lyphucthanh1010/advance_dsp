#!/bin/bash
# Script này tạo commit giả với ngày tháng tùy chỉnh để "lấp đầy" lịch sử commit

# Cấu hình ngày bắt đầu và ngày kết thúc (định dạng YYYY-MM-DD)
START_DATE="2024-01-01"
END_DATE="2025-02-19"  # thay đổi theo khoảng thời gian bạn muốn

# File sẽ được sửa đổi mỗi commit (có thể là file dummy hoặc file có sẵn trong dự án của bạn)
FILE="app.py"

# Kiểm tra nếu file chưa tồn tại thì tạo file
if [ ! -f "$FILE" ]; then
  echo "Khởi tạo file $FILE" > "$FILE"
fi

# Chuyển sang định dạng ngày của GNU date (nếu cần điều chỉnh cho macOS thì có thể dùng gdate)
current_date="$START_DATE"

while [[ "$current_date" < "$END_DATE" ]]; do
  echo "Commit cho ngày: $current_date"
  
  # Thêm một dòng vào file để có thay đổi
  echo print("Dummy commit cho ngày ${current_date}") >> "$FILE"
  git add "$FILE"
  
  # Thiết lập biến môi trường cho ngày commit
  GIT_AUTHOR_DATE="$current_date 12:00:00" \
  GIT_COMMITTER_DATE="$current_date 12:00:00" \
  git commit -m "Dummy commit cho ngày $current_date"
  git push
  # Tăng ngày lên 1 đơn vị
  current_date=$(date -I -d "$current_date + 1 day")
done

echo "Hoàn thành việc tạo commit từ $START_DATE đến $END_DATE."
