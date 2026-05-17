r"""
download_mcocr.py — Script tải lại dữ liệu MC-OCR nếu bị thiếu/corrupt.

Hướng dẫn sử dụng:
    python data_preparation/download_mcocr.py

Yêu cầu:
    - Tài khoản Kaggle + kaggle.json đã được cấu hình
      (đặt tại ~/.kaggle/kaggle.json hoặc %USERPROFILE%\.kaggle\kaggle.json)
    - Cài kaggle CLI: pip install kaggle

Bộ dữ liệu: https://www.kaggle.com/competitions/mc-ocr
"""
import os
import subprocess
import sys
import zipfile

OUT_DIR = os.path.join("data", "mc-ocr", "raw")
KAGGLE_DATASET = "mc-ocr"  # competition slug


def check_kaggle_cli():
    """Kiểm tra kaggle CLI đã được cài chưa."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    print("Kaggle CLI chưa được cài. Chạy: pip install kaggle")
    return False


def check_kaggle_creds():
    """Kiểm tra file credentials kaggle.json."""
    home = os.path.expanduser("~")
    cred_path = os.path.join(home, ".kaggle", "kaggle.json")
    if os.path.isfile(cred_path):
        return True
    print(f"Không tìm thấy {cred_path}")
    print("Tải kaggle.json từ: https://www.kaggle.com/settings → API → Create New Token")
    return False


def download():
    """Tải dataset MC-OCR từ Kaggle competition."""
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Đang tải MC-OCR dataset vào {OUT_DIR}...")
    cmd = [
        "kaggle", "competitions", "download",
        "-c", KAGGLE_DATASET,
        "-p", OUT_DIR,
    ]
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print("Tải thất bại. Hãy kiểm tra lại thông tin đăng nhập Kaggle.")
        sys.exit(1)

    # Giải nén tất cả file zip trong thư mục
    for fname in os.listdir(OUT_DIR):
        if fname.endswith(".zip"):
            zip_path = os.path.join(OUT_DIR, fname)
            print(f"Đang giải nén {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(OUT_DIR)
            os.remove(zip_path)

    print("Hoàn tất tải dữ liệu MC-OCR.")


def verify():
    """Kiểm tra các file cần thiết đã có chưa."""
    required = [
        os.path.join(OUT_DIR, "mcocr_train_df.csv"),
        os.path.join(OUT_DIR, "train_images"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("Thiếu các file sau:")
        for p in missing:
            print(f"  {p}")
        return False
    print("Dữ liệu MC-OCR đầy đủ.")
    return True


def main():
    print("=== MC-OCR Download Tool ===\n")

    if verify():
        print("Dữ liệu đã tồn tại, không cần tải lại.")
        return

    if not check_kaggle_cli():
        sys.exit(1)
    if not check_kaggle_creds():
        sys.exit(1)

    download()
    verify()


if __name__ == "__main__":
    main()
