#!/bin/bash
# =============================================================
# Script setup môi trường trên Google Colab
# Sử dụng: !bash setup_colab.sh (trong notebook cell đầu tiên)
# =============================================================

set -e

REPO_URL="https://github.com/Duc-AnhTp/document-ai-vn.git"
REPO_DIR="/content/document-ai-vn"

echo "=== [1/3] Clone / cập nhật repo ==="
if [ -d "$REPO_DIR" ]; then
    echo "Repo đã tồn tại, cập nhật..."
    cd "$REPO_DIR" && git pull --quiet
else
    git clone --quiet "$REPO_URL" "$REPO_DIR"
fi

echo "=== [2/3] Cài đặt dependencies ==="
cd "$REPO_DIR"
# Cài gdown trước vì cần để tải dataset
pip install -q gdown
pip install -q paddlepaddle-gpu paddleocr
pip install -r requirements.txt -q

echo "=== [3/3] Cấu hình PYTHONPATH ==="
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

echo ""
echo "✅ Setup hoàn tất!"
echo "   Repo : $REPO_DIR"
echo "   Data : $REPO_DIR/data/mcocr/"
echo ""
echo "Bước tiếp theo:"
echo "  python -m src.preprocessing.download_dataset  # Tải data"
echo "  python -m src.preprocessing.run_prepare_data  # Tiền xử lý"
echo "  python -m src.kie.train                        # Huấn luyện"
