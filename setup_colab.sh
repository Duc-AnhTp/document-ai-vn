#!/bin/bash
# =============================================================
# Script setup môi trường trên Google Colab
# Chạy cell đầu tiên trong notebook: !bash setup_colab.sh
# =============================================================

set -e

echo "=== [1/4] Mount Google Drive ==="
# Drive được mount qua notebook cell, script chỉ kiểm tra
if [ ! -d "/content/drive/MyDrive" ]; then
    echo "CẢNH BÁO: Google Drive chưa mount. Hãy chạy:"
    echo "  from google.colab import drive"
    echo "  drive.mount('/content/drive')"
    exit 1
fi

echo "=== [2/4] Clone / cập nhật repo ==="
REPO_DIR="/content/document-ai-vn"
REPO_URL="https://github.com/Duc-AnhTp/document-ai-vn.git"

if [ -d "$REPO_DIR" ]; then
    echo "Repo đã tồn tại, pull latest..."
    cd "$REPO_DIR" && git pull
else
    git clone "$REPO_URL" "$REPO_DIR"
fi

echo "=== [3/4] Cài đặt dependencies ==="
pip install -q paddlepaddle-gpu
pip install -q paddleocr
pip install -q transformers datasets seqeval
pip install -q albumentations opencv-python-headless
pip install -q matplotlib pandas scikit-learn Pillow

echo "=== [4/4] Thêm repo vào PYTHONPATH ==="
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
echo "PYTHONPATH=$PYTHONPATH"

echo ""
echo "✅ Setup hoàn tất! Repo tại: $REPO_DIR"
echo "📂 Dataset tại: /content/drive/MyDrive/mcocr/"
