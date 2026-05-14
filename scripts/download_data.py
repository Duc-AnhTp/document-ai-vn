"""
Tải dataset về thư mục data/.

Sử dụng:
    python scripts/download_data.py --dataset mcocr --output data/
    python scripts/download_data.py --dataset cord --output data/
    python scripts/download_data.py --dataset sroie --output data/
    python scripts/download_data.py --dataset all --output data/
"""

import argparse
import os
import subprocess
import sys


def download_mcocr(output_dir: str):
    """Tải MC-OCR 2021 từ Kaggle."""
    dest = os.path.join(output_dir, "mc-ocr", "raw")
    if os.path.exists(dest) and os.listdir(dest):
        print(f"[SKIP] MC-OCR đã tồn tại tại {dest}")
        return

    os.makedirs(dest, exist_ok=True)
    print("[INFO] Đang tải MC-OCR 2021 từ Kaggle...")
    print("[INFO] Đảm bảo đã cài kaggle CLI: pip install kaggle")
    print("[INFO] Đảm bảo đã có file ~/.kaggle/kaggle.json")

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "domixi1989/vietnamese-receipts-mc-ocr-2021",
                "-p", dest,
                "--unzip",
            ],
            check=True,
        )
        print(f"[OK] MC-OCR đã tải về {dest}")
    except FileNotFoundError:
        print("[LỖI] Không tìm thấy lệnh 'kaggle'. Cài đặt: pip install kaggle")
        print(f"[ALT] Tải thủ công: https://kaggle.com/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021")
        print(f"[ALT] Giải nén vào: {dest}")
    except subprocess.CalledProcessError as e:
        print(f"[LỖI] Tải MC-OCR thất bại: {e}")


def download_cord(output_dir: str):
    """Tải CORD v2 từ HuggingFace."""
    dest = os.path.join(output_dir, "cord-v2")
    if os.path.exists(dest) and os.listdir(dest):
        print(f"[SKIP] CORD v2 đã tồn tại tại {dest}")
        return

    os.makedirs(dest, exist_ok=True)
    print("[INFO] Đang tải CORD v2 từ HuggingFace...")

    try:
        from datasets import load_dataset

        ds = load_dataset("naver-clova-ix/cord-v2")
        ds.save_to_disk(dest)
        print(f"[OK] CORD v2 đã lưu về {dest}")
    except ImportError:
        print("[LỖI] Cần cài: pip install datasets")
    except Exception as e:
        print(f"[LỖI] Tải CORD thất bại: {e}")


def download_sroie(output_dir: str):
    """Hướng dẫn tải SROIE 2019."""
    dest = os.path.join(output_dir, "sroie")
    if os.path.exists(dest) and os.listdir(dest):
        print(f"[SKIP] SROIE đã tồn tại tại {dest}")
        return

    os.makedirs(dest, exist_ok=True)
    print("[INFO] SROIE 2019 cần tải thủ công hoặc từ Kaggle:")
    print("  Kaggle: kaggle datasets download -d urbikn/sroie-datasetv2")
    print(f"  Giải nén vào: {dest}")
    print("  Hoặc từ: https://rrc.cvc.uab.es/?ch=13")

    # Thử tải từ Kaggle nếu có CLI
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "urbikn/sroie-datasetv2",
                "-p", dest,
                "--unzip",
            ],
            check=True,
        )
        print(f"[OK] SROIE đã tải về {dest}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[INFO] Tải thủ công theo hướng dẫn ở trên.")


def main():
    parser = argparse.ArgumentParser(description="Tải dataset cho dự án Document AI KIE")
    parser.add_argument("--dataset", choices=["mcocr", "cord", "sroie", "all"], required=True)
    parser.add_argument("--output", default="data/", help="Thư mục đầu ra")
    args = parser.parse_args()

    download_fns = {
        "mcocr": download_mcocr,
        "cord": download_cord,
        "sroie": download_sroie,
    }

    if args.dataset == "all":
        for name, fn in download_fns.items():
            print(f"\n{'='*50}")
            print(f"Tải {name.upper()}")
            print(f"{'='*50}")
            fn(args.output)
    else:
        download_fns[args.dataset](args.output)

    print("\n[DONE] Hoàn tất.")


if __name__ == "__main__":
    main()
