import pyautogui
import time

print("Chương trình đang chạy... Nhấn 'Space' mỗi 30 giây.")
print("Để dừng chương trình: Nhấn Ctrl + C trong cửa sổ này.")

try:
    while True:
        # Nhấn phím Space     
        pyautogui.press('space')
        print("Đã nhấn Space")
        
        # Chờ 30 giây
        time.sleep(30)

except KeyboardInterrupt:
    print("\nĐã dừng chương trình.") 