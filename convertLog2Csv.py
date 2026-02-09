import re
import csv

def parse_log_to_csv(log_file_path, output_csv_1, output_csv_2):
    # Regex để bắt các thông tin cơ bản ở đầu mỗi dòng log (Epoch, Step, Time, ETA)
    # Mẫu: 2025-12-04 12:18:17,358 - Epoch:   0, Step:     0 /   832, time: 13.967s, ETA: 11634.39s, ...
    base_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})\s-\sEpoch:\s*(\d+),\s*Step:\s*(\d+)\s*/\s*(\d+),\s*time:\s*([\d.]+)s,\s*ETA:\s*([\d.]+)s')

    # Regex cho giai đoạn 1 (Epoch < 5): Chỉ lấy Pre_train_G_loss
    pretrain_pattern = re.compile(r',\s*Pre_train_G_loss:\s*([\d.]+)')

    # Danh sách chứa dữ liệu cho 2 file
    data_phase1 = [] # Cho 5 epoch đầu
    data_phase2 = [] # Cho các epoch còn lại
    
    # Tập hợp các tên cột loss cho file thứ 2 (để tự động phát hiện các loại loss)
    loss_keys_phase2 = set()

    print(f"Đang đọc file: {log_file_path}...")
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Bỏ qua các dòng không phải log training (ví dụ header info)
            base_match = base_pattern.search(line)
            
            if base_match:
                timestamp, epoch, step, total_steps, time_val, eta = base_match.groups()
                epoch = int(epoch)
                
                # Tạo dictionary chứa thông tin cơ bản
                row = {
                    'Timestamp': timestamp,
                    'Epoch': epoch,
                    'Step': int(step),
                    'Total_Steps': int(total_steps),
                    'Time_s': float(time_val),
                    'ETA_s': float(eta)
                }
                
                if epoch < 5:
                    # --- Xử lý 5 Epoch đầu ---
                    pt_match = pretrain_pattern.search(line)
                    if pt_match:
                        row['Pre_train_G_loss'] = float(pt_match.group(1))
                        data_phase1.append(row)
                else:
                    # --- Xử lý các Epoch còn lại ---
                    # Phần dữ liệu loss nằm sau chuỗi "ETA: ...s, "
                    try:
                        # Tách lấy phần đuôi chứa các giá trị loss
                        parts = line.split(f"ETA: {eta}s, ")
                        if len(parts) > 1:
                            loss_part = parts[1]
                            # Chuẩn hóa các dấu phân cách ' ~ ' và ' || ' thành dấu phẩy để dễ tách
                            # Ví dụ gốc: D_loss:0.562 ~ G_loss: 43.589 || G_support_loss: ...
                            standardized_loss = loss_part.replace(' ~ ', ', ').replace(' || ', ', ')
                            
                            # Tách từng cặp key:value
                            loss_items = standardized_loss.split(', ')
                            for item in loss_items:
                                if ':' in item:
                                    key, val = item.split(':', 1)
                                    key = key.strip()
                                    try:
                                        row[key] = float(val.strip())
                                        loss_keys_phase2.add(key)
                                    except ValueError:
                                        continue # Bỏ qua nếu không convert được số
                            
                            data_phase2.append(row)
                    except Exception as e:
                        print(f"Lỗi khi parse dòng epoch {epoch}: {e}")

    # --- Ghi file CSV 1 (5 epoch đầu) ---
    header1 = ['Timestamp', 'Epoch', 'Step', 'Total_Steps', 'Time_s', 'ETA_s', 'Pre_train_G_loss']
    with open(output_csv_1, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header1)
        writer.writeheader()
        writer.writerows(data_phase1)
    print(f"Đã tạo file: {output_csv_1} ({len(data_phase1)} dòng)")

    # --- Ghi file CSV 2 (Các epoch còn lại) ---
    # Sắp xếp header: Thông tin cơ bản trước, sau đó đến các loại loss
    base_cols = ['Timestamp', 'Epoch', 'Step', 'Total_Steps', 'Time_s', 'ETA_s']
    sorted_loss_cols = sorted(list(loss_keys_phase2))
    header2 = base_cols + sorted_loss_cols
    
    with open(output_csv_2, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header2)
        writer.writeheader()
        writer.writerows(data_phase2)
    print(f"Đã tạo file: {output_csv_2} ({len(data_phase2)} dòng)")

# --- Chạy hàm chuyển đổi ---
# File input
input_file = 'training_111.log'
# File output
output_1 = 'pretrain_log_epoch_0_4.csv'
output_2 = 'gan_training_log_epoch_5_plus.csv'

parse_log_to_csv(input_file, output_1, output_2)