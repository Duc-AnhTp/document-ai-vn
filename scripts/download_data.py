"""
Download datasets used by the Donut-only main branch.

Usage:
    python scripts/download_data.py --dataset mcocr --output data/
    python scripts/download_data.py --dataset cord --output data/
    python scripts/download_data.py --dataset all --output data/
"""

import argparse
import os
import subprocess


def download_mcocr(output_dir: str):
    """Download MC-OCR 2021 from Kaggle."""
    dest = os.path.join(output_dir, "mc-ocr", "raw")
    if os.path.exists(dest) and os.listdir(dest):
        print(f"[SKIP] MC-OCR already exists at {dest}")
        return

    os.makedirs(dest, exist_ok=True)
    print("[INFO] Downloading MC-OCR 2021 from Kaggle...")
    print("[INFO] Make sure the Kaggle CLI is installed: pip install kaggle")
    print("[INFO] Make sure kaggle.json exists in your user .kaggle directory")

    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "domixi1989/vietnamese-receipts-mc-ocr-2021",
                "-p",
                dest,
                "--unzip",
            ],
            check=True,
        )
        print(f"[OK] MC-OCR downloaded to {dest}")
    except FileNotFoundError:
        print("[ERROR] Command 'kaggle' was not found. Install it with: pip install kaggle")
        print("[ALT] Manual download: https://kaggle.com/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021")
        print(f"[ALT] Extract the dataset into: {dest}")
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] MC-OCR download failed: {exc}")


def download_cord(output_dir: str):
    """Download CORD v2 from HuggingFace."""
    dest = os.path.join(output_dir, "cord-v2")
    if os.path.exists(dest) and os.listdir(dest):
        print(f"[SKIP] CORD v2 already exists at {dest}")
        return

    os.makedirs(dest, exist_ok=True)
    print("[INFO] Downloading CORD v2 from HuggingFace...")

    try:
        from datasets import load_dataset

        ds = load_dataset("naver-clova-ix/cord-v2")
        ds.save_to_disk(dest)
        print(f"[OK] CORD v2 saved to {dest}")
    except ImportError:
        print("[ERROR] Missing dependency. Install it with: pip install datasets")
    except Exception as exc:
        print(f"[ERROR] CORD v2 download failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Donut-only training")
    parser.add_argument("--dataset", choices=["mcocr", "cord", "all"], required=True)
    parser.add_argument("--output", default="data/", help="Output directory")
    args = parser.parse_args()

    download_fns = {
        "mcocr": download_mcocr,
        "cord": download_cord,
    }

    if args.dataset == "all":
        for name, fn in download_fns.items():
            print(f"\n{'=' * 50}")
            print(f"Download {name.upper()}")
            print(f"{'=' * 50}")
            fn(args.output)
    else:
        download_fns[args.dataset](args.output)

    print("\n[DONE] Finished.")


if __name__ == "__main__":
    main()
