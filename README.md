# Document AI - Vietnamese Invoice Extraction

Hệ thống Document AI sử dụng **PaddleOCR** và **LayoutLMv3** để nhận dạng và trích xuất thông tin từ hóa đơn/biên lai tiếng Việt.

## 🎯 Mục tiêu

- **OCR**: Phát hiện và nhận dạng văn bản trong ảnh hóa đơn (tiếng Việt)
- **KIE**: Trích xuất thông tin quan trọng: `SELLER`, `SELLER_ADDRESS`, `TIMESTAMP`, `TOTAL_COST`
- **Target**: F1 ≥ 0.80 trên dataset MC-OCR 2021

## 📐 Pipeline

```
Ảnh hóa đơn → PaddleOCR → Text + Bounding Boxes → LayoutLMv3 → Nhãn thực thể → Output JSON
```

## 📂 Cấu trúc dự án

```
document-ai-vn/
├── configs/
│   ├── paths.py              # Đường dẫn dataset, output
│   └── layoutlmv3_config.py  # Hyperparams, label schema
├── src/
│   ├── preprocessing/
│   │   ├── parse_mcocr.py     # Parse CSV annotation → dict
│   │   ├── clean_data.py      # Làm sạch text, chuẩn hóa label
│   │   ├── split_data.py      # Chia train/val (stratified)
│   │   ├── augment.py         # Augmentation ảnh
│   │   └── run_prepare_data.py# Pipeline tiền xử lý đầy đủ
│   ├── ocr/
│   │   └── run_paddleocr.py   # PaddleOCR text detection + recognition
│   ├── kie/
│   │   ├── dataset.py         # HuggingFace Dataset + LayoutLMv3 encoding
│   │   ├── train.py           # Huấn luyện mô hình
│   │   ├── evaluate.py        # Đánh giá Precision/Recall/F1
│   │   └── infer.py           # Inference: ảnh → JSON
│   ├── utils/
│   │   ├── bbox.py            # Polygon → bbox, normalize cho LayoutLMv3
│   │   ├── io.py              # Đọc/ghi JSONL
│   │   └── labels.py          # Label constants, normalization map
│   └── visualization/
│       └── plot_samples.py    # Vẽ mẫu kèm bounding box, biểu đồ
├── notebooks/
│   ├── 01_eda.ipynb           # Khám phá dữ liệu
│   ├── 02_preprocessing.ipynb # Demo tiền xử lý
│   └── 03_train_layoutlmv3.ipynb # Huấn luyện & đánh giá
├── data/
│   └── README.md              # Mô tả dataset (không lưu data trong repo)
├── outputs/
│   ├── processed/             # JSONL sau tiền xử lý
│   ├── checkpoints/           # Model weights
│   ├── figures/               # Biểu đồ EDA
│   ├── logs/                  # Training logs
│   └── predictions/           # Kết quả inference
├── requirements.txt
├── setup_colab.sh
└── README.md
```

## 📊 Dataset: MC-OCR 2021

- **2,000+** ảnh hóa đơn/biên lai tiếng Việt chụp điều kiện thực tế
- Annotation: `img_id`, `anno_texts`, `anno_labels`, `anno_polygons`, `anno_image_quality`
- Lưu trên Google Drive tại `/content/drive/MyDrive/mcocr/`

## 🚀 Hướng dẫn sử dụng

### Trên Google Colab

1. Mount Google Drive và clone repo:
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/Duc-AnhTp/document-ai-vn.git /content/document-ai-vn
```

2. Setup môi trường:
```bash
!bash /content/document-ai-vn/setup_colab.sh
```

3. Chạy notebook theo thứ tự:
   - `01_eda.ipynb` → Khám phá dữ liệu
   - `02_preprocessing.ipynb` → Tiền xử lý
   - `03_train_layoutlmv3.ipynb` → Huấn luyện & đánh giá

### Tiền xử lý dữ liệu (CLI)

```bash
cd /content/document-ai-vn
python -m src.preprocessing.run_prepare_data
```

### Inference

```bash
python -m src.kie.infer --image path/to/invoice.jpg
```

### Table Detection

```bash
python -m src.table_detection.detector --image path/to/invoice.jpg
```

## 📈 Metrics

| Task | Metric |
|------|--------|
| OCR | CER, WER |
| KIE | Precision, Recall, **F1** |
| MC-OCR chính thức | CER end-to-end |

## ⚙️ Cấu hình

Chỉnh sửa trong `configs/`:
- `paths.py`: đường dẫn dataset, thư mục output
- `layoutlmv3_config.py`: model name, batch size, learning rate, epochs