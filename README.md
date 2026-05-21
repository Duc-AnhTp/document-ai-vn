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

SROIE cần convert riêng trước khi chạy E3:
```bash
python scripts/convert_sroie.py \
  --input data/sroie/ \
  --output data/sroie/donut_format/
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

`metrics.json` hiện lưu thêm `avg_inference_ms` để đối chiếu bảng so sánh trong `PROJECT.md`.

### E3 — Cross-dataset SROIE
```bash
python scripts/convert_sroie.py --input data/sroie/ --output data/sroie/donut_format/

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

---

## Hướng dẫn chạy nhanh

Nếu bạn chỉ muốn chạy project theo thứ tự ít rối nhất, hãy làm như sau trong `Windows PowerShell`.

### 1. Tạo môi trường và cài thư viện

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Tải dữ liệu

```powershell
python scripts/download_data.py --dataset mcocr --output data/
python scripts/download_data.py --dataset cord --output data/
python scripts/download_data.py --dataset sroie --output data/
```

Lưu ý:
- `MC-OCR` và `SROIE` cần `kaggle` CLI nếu muốn tải tự động.
- Nếu `SROIE` không tải được tự động, hãy giải nén thủ công vào `data/sroie/`.
- SROIE cần đúng cấu trúc `data/sroie/SROIE2019/train/img/` + `train/entities/` và `test/img/` + `test/entities/`.

### 3. Convert dữ liệu

```powershell
python scripts/convert_mcocr.py --input data/mc-ocr/raw/ --output data/mc-ocr/donut_format/ --split-ratio 0.8 0.1 0.1
python scripts/convert_sroie.py --input data/sroie/ --output data/sroie/donut_format/
```

Lưu ý:
- `convert_mcocr.py` sẽ in sample record và hỏi xác nhận mapping cột.
- Chỉ dùng `--force` khi bạn chắc mapping đã đúng.

### 4. Kiểm tra dữ liệu đã ổn chưa

```powershell
python scripts/eda.py --data-dir data/mc-ocr/donut_format/ --output docs/eda_figures/
```

Kết quả mong đợi:
- `docs/eda_figures/` có các file như `split_counts.png`, `field_distribution.png`, `samples.png`.

### 5. Chạy baseline E1

```powershell
python scripts/baseline_paddleocr.py --test-dir data/mc-ocr/donut_format/test/ --output results/e1_baseline/
```

Kết quả mong đợi:
- `results/e1_baseline/metrics.json`
- `results/e1_baseline/predictions.json`

### 6. Chạy Donut E2

```powershell
python scripts/train_donut.py --config configs/donut_cord.yaml
python scripts/train_donut.py --config configs/donut_mcocr.yaml
python scripts/evaluate.py --checkpoint results/e2_donut/checkpoints/mcocr --test-dir data/mc-ocr/donut_format/test/ --output results/e2_donut/metrics.json
```

Lưu ý:
- Warm-up CORD hiện chọn checkpoint theo `val_loss`.
- `metrics.json` của E2 có thêm `avg_inference_ms`.

Kết quả mong đợi:
- `results/e2_donut/checkpoints/cord_warmup`
- `results/e2_donut/checkpoints/mcocr`
- `results/e2_donut/training_log.csv`
- `results/e2_donut/metrics.json`

### 7. Chạy E3

```powershell
python scripts/train_donut_sroie.py --config configs/donut_sroie.yaml
```

Kết quả mong đợi:
- `results/e3_cross/checkpoints`
- `results/e3_cross/training_log.csv`
- `results/e3_cross/error_analysis.json`

### Cách nhớ nhanh

- Bước 1: cài thư viện
- Bước 2: tải data
- Bước 3: convert data
- Bước 4: kiểm tra data
- Bước 5: chạy E1
- Bước 6: chạy E2
- Bước 7: chạy E3

### Lỗi thường gặp

- Không tìm thấy `kaggle`:
  - Chạy `pip install kaggle`
- Thiếu `kaggle.json`:
  - Đặt file vào `C:\Users\<ten_user>\.kaggle\kaggle.json`
- SROIE sai cấu trúc:
  - Kiểm tra lại `data/sroie/SROIE2019/train/img/` + `train/entities/` và `test/img/` + `test/entities/`
- MC-OCR map sai CSV:
  - Chạy convert không dùng `--force` để kiểm tra sample record
- Không có GPU hoặc thiếu VRAM:
  - Có thể test bằng CPU, nhưng train Donut sẽ chậm
