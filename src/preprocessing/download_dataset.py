import os
import zipfile
import gdown
from pathlib import Path

from configs.paths import DATA_ROOT

def download_and_extract_data():
    """
    Tự động tải file dataset MC-OCR (định dạng .zip) từ Google Drive
    và giải nén vào thư mục cấu hình nội bộ.
    """
    # ---------------------------------------------------------
    # TODO: Thay thế FILE_ID này bằng ID file ZIP public của bạn 
    # Ví dụ link: https://drive.google.com/file/d/1XyZabc.../view
    # Thì ID sẽ là: 1XyZabc...
    # ---------------------------------------------------------
    FILE_ID = 'NHAP_ID_FILE_ZIP_CUA_BAN_VAO_DAY'
    ZIP_PATH = DATA_ROOT / 'mcocr_dataset.zip'
    
    print(f"=== Đang kiểm tra dữ liệu tại {DATA_ROOT} ===")
    
    # Kiểm tra xem folder chứa ảnh cấu hình ở paths.py (train_images) đã có chưa
    # Sơ bộ ta check xem folder cha có tầm vài trăm file ảnh không, để đỡ phải tải lại
    # Tạm gán 1 folder mồi
    expected_image_dir = DATA_ROOT / 'train_images'
    if expected_image_dir.exists() and len(list(expected_image_dir.glob('*.jpg'))) > 100:
        print("✅ Dữ liệu có vẻ đã tồn tại! Bỏ qua lệnh tải về.")
        return

    # Nếu ID chưa được thay thế
    if FILE_ID == 'NHAP_ID_FILE_ZIP_CUA_BAN_VAO_DAY':
        print("❌ LỖI: Bạn chưa cung cấp ID file Google Drive chia sẻ.")
        print("Vui lòng mở file src/preprocessing/download_dataset.py và dán ID public vào.")
        return
        
    # Bắt đầu tải
    print("⏳ Dữ liệu chưa đầy đủ. Tiến hành tải nén từ Google Drive...")
    
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    try:
        gdown.download(url, str(ZIP_PATH), quiet=False)
    except Exception as e:
        print(f"Lỗi tải xuống: {e}")
        return
        
    if not ZIP_PATH.exists():
        print("❌ Không tìm thấy file ZIP sau khi tải.")
        return
        
    print(f"📦 Đang giải nén bộ dữ liệu vào {DATA_ROOT}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(str(DATA_ROOT))
        
    # Xoá file zip tạm cho nhẹ bộ nhớ
    os.remove(ZIP_PATH)
    
    print("✅ Hoàn tất tải và giải nén dữ liệu MC-OCR!")

if __name__ == '__main__':
    download_and_extract_data()
