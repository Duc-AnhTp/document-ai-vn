# 🧾 Document AI — Trích xuất Thông tin Biên lai Tiếng Việt

> Đồ án môn **Thị giác Máy tính** — HUCE
>
> Fine-tune mô hình [Donut](https://arxiv.org/abs/2111.15664) (OCR-free Document Understanding Transformer) để trích xuất thông tin từ biên lai chụp điện thoại tiếng Việt.

## Sơ đồ kiến trúc

![Kiến trúc tổng thể](docs/architecture.png)

## Thí nghiệm

| # | Thí nghiệm | Phương pháp | Dataset |
|---|------------|-------------|---------|
| E1 | Baseline | PaddleOCR + regex/rule | MC-OCR 2021 |
| E2 | **Mô hình chính** | Donut fine-tune | CORD (warm-up) → MC-OCR |
| E3 | Cross-dataset | Donut (từ E2) fine-tune thêm | SROIE 2019 |

## Cài đặt

```bash
pip install -r requirements.txt
```

## Tải dữ liệu

```bash
python scripts/download_data.py --dataset all --output data/
```

Hoặc tải riêng:
```bash
python scripts/download_data.py --dataset mcocr --output data/
python scripts/download_data.py --dataset cord --output data/
python scripts/download_data.py --dataset sroie --output data/
```

## Convert annotation

```bash
python scripts/convert_mcocr.py \
  --input data/mc-ocr/raw/ \
  --output data/mc-ocr/donut_format/ \
  --split-ratio 0.8 0.1 0.1
```

## Chạy thí nghiệm

### E1 — PaddleOCR Baseline
```bash
python scripts/baseline_paddleocr.py \
  --test-dir data/mc-ocr/donut_format/test/ \
  --output results/e1_baseline/
```

### E2 — Donut Fine-tune
```bash
# Warm-up trên CORD v2
python scripts/train_donut.py --config configs/donut_cord.yaml

# Fine-tune trên MC-OCR
python scripts/train_donut.py --config configs/donut_mcocr.yaml

# Evaluate
python scripts/evaluate.py \
  --checkpoint results/e2_donut/checkpoints/mcocr \
  --test-dir data/mc-ocr/donut_format/test/ \
  --output results/e2_donut/metrics.json
```

### E3 — Cross-dataset SROIE
```bash
python scripts/train_donut_sroie.py --config configs/donut_sroie.yaml
```

## Kết quả

| Thí nghiệm | F1 | Precision | Recall |
|-------------|-----|-----------|--------|
| E1 (PaddleOCR) | — | — | — |
| E2 (Donut) | — | — | — |
| E3 (SROIE) | — | — | — |

> Kết quả sẽ được cập nhật sau khi chạy thí nghiệm.

## Cấu trúc thư mục

```
document-ai-vn/
├── configs/          # Config YAML cho từng thí nghiệm
├── scripts/          # Code chính (train, evaluate, baseline)
├── notebooks/        # Demo & trình bày kết quả
├── data/             # Dataset (không commit, tải bằng script)
├── results/          # Kết quả thí nghiệm (metrics commit, checkpoints không)
├── docs/             # Tài liệu bổ sung
├── PROJECT.md        # Mô tả chi tiết dự án
└── README.md         # File này
```

## Tham khảo

- [Donut: OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664) — Kim et al., 2022
- [Donut GitHub](https://github.com/clovaai/donut)
- [MC-OCR 2021](https://kaggle.com/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021)
- [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
- [SROIE 2019](https://rrc.cvc.uab.es/?ch=13)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
