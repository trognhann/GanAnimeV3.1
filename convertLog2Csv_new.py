import csv
import re

def convert_log_to_csv(log_file_path, output_csv_path):
    # Định nghĩa các cột dữ liệu bạn muốn lấy
    # Dựa trên file log của bạn, đây là các trường quan trọng
    headers = [
        'Epoch', 'Step', 'Time', 
        'D_loss', 'G_loss', 
        'G_support_loss', 'g_s_loss', 'con_loss', 'rs_loss', 'sty_loss', 
        'color_loss', 'tv_loss', 'D_support_loss',
        'G_main_loss', 'g_m_loss', 'p0_loss', 'p4_loss', 'tv_loss_m', 'D_main_loss'
    ]

    data_rows = []

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Bỏ qua các dòng log hệ thống hoặc giai đoạn pre-train (chỉ lấy giai đoạn train chính)
                # Dòng train chính chứa "G_support_loss"
                if not line or "G_support_loss" not in line:
                    continue

                row_data = {}

                # 1. Trích xuất thông tin cơ bản (Epoch, Step, Time) bằng Regex
                # Mẫu: Epoch:   9, Step:   157 /  832, time: 3.452s
                match_meta = re.search(r'Epoch:\s*(\d+).*Step:\s*(\d+).*time:\s*([\d\.]+)s', line)
                if match_meta:
                    row_data['Epoch'] = match_meta.group(1)
                    row_data['Step'] = match_meta.group(2)
                    row_data['Time'] = match_meta.group(3)

                # 2. Làm sạch dòng log để dễ tách dữ liệu
                # Thay thế các ký tự ngăn cách đặc biệt '||' và '~' thành dấu phẩy ','
                clean_line = line.replace('||', ',').replace('~', ',')
                
                # Tách chuỗi thành các phần nhỏ dựa trên dấu phẩy
                parts = clean_line.split(',')

                # 3. Duyệt qua từng phần để lấy key:value
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Chỉ lấy các trường nằm trong danh sách headers
                        if key in headers:
                            row_data[key] = value

                # Đảm bảo dòng dữ liệu có đủ cột (nếu thiếu điền rỗng)
                if row_data:
                    data_rows.append({h: row_data.get(h, '') for h in headers})

        # Ghi ra file CSV
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data_rows)

        print(f"✅ Đã chuyển đổi thành công! Có {len(data_rows)} dòng dữ liệu.")
        print(f"📂 File kết quả: {output_csv_path}")

    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {log_file_path}")
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Thay đổi tên file nếu cần thiết
    input_log = 'train.log'  
    output_csv = 'training_data.csv'
    
    convert_log_to_csv(input_log, output_csv)