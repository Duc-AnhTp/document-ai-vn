"""Script tự động tải và giải nén dataset MC-OCR từ Google Drive."""

import os
import zipfile
from pathlib import Path

import gdown

from configs.paths import DATA_ROOT, IMAGE_DIR

# ID file ZIP public trên Google Drive
# Cách lấy ID: mở file trên Drive → Share → Copy link
# Ví dụ link: https://drive.google.com/file/d/<FILE_ID>/view
# Dán <FILE_ID> vào biến dưới đây sau khi upload file ZIP của bạn.
GDRIVE_FILE_ID = 'REPLACE_WITH_YOUR_GDRIVE_FILE_ID'

MIN_IMAGE_COUNT = 100  # Ngưỡng tối thiểu để coi dữ liệu là đã tồn tại


def is_data_ready() -> bool:
    """Kiểm tra xem dữ liệu đã được giải nén đầy đủ chưa.

    Returns:
        True nếu thư mục ảnh tồn tại và có ít nhất MIN_IMAGE_COUNT ảnh.
    """
    if not IMAGE_DIR.exists():
        return False
    image_count = len(list(IMAGE_DIR.glob('*.jpg')))
    return image_count >= MIN_IMAGE_COUNT


def download_and_extract() -> None:
    """Tải file ZIP từ Google Drive và giải nén vào DATA_ROOT.

    Bỏ qua nếu dữ liệu đã tồn tại. Báo lỗi rõ ràng nếu GDRIVE_FILE_ID
    chưa được cấu hình.
    """
    print(f'Kiểm tra dữ liệu tại: {DATA_ROOT}')

    if is_data_ready():
        print('✅ Dữ liệu đã sẵn sàng. Bỏ qua bước tải.')
        return

    if GDRIVE_FILE_ID == 'REPLACE_WITH_YOUR_GDRIVE_FILE_ID':
        raise ValueError(
            'Chưa cấu hình GDRIVE_FILE_ID.\n'
            'Mở file src/preprocessing/download_dataset.py và điền ID file ZIP vào.'
        )

    zip_path = DATA_ROOT / 'mcocr_dataset.zip'
    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'

    print('⏳ Đang tải dataset từ Google Drive...')
    gdown.download(url, str(zip_path), quiet=False)

    if not zip_path.exists():
        raise FileNotFoundError(f'Tải xuống thất bại. Không tìm thấy: {zip_path}')

    print(f'📦 Đang giải nén vào {DATA_ROOT}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(str(DATA_ROOT))

    os.remove(zip_path)
    print('✅ Hoàn tất. Dataset đã sẵn sàng.')


if __name__ == '__main__':
    download_and_extract()
