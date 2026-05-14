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

## HƯỚNG DẪN CHẠY DỰ ÁN TỪ ĐẦU ĐẾN CUỐI

Phần này hướng dẫn chạy toàn bộ project theo thứ tự từ chuẩn bị môi trường, tải dữ liệu, tiền xử lý, EDA, chạy E1, E2, E3 cho đến cách đọc kết quả. Tất cả ví dụ lệnh bên dưới ưu tiên `Windows PowerShell`.

### 1. Chuẩn bị môi trường

- Yêu cầu tối thiểu:
  - Python 3.10+.
  - `pip`.
  - GPU là tùy chọn, nhưng nên có nếu muốn train Donut nhanh và ổn định hơn.
- Mở PowerShell tại thư mục project rồi chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Kiểm tra nhanh phiên bản Python, pip và khả năng nhận GPU của PyTorch:

```powershell
python --version
pip --version
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Kết quả mong đợi:
- `python --version` và `pip --version` chạy được bình thường.
- Lệnh PyTorch in ra `CUDA available: True` nếu máy có GPU cấu hình đúng; nếu ra `False` thì vẫn có thể chạy, nhưng train Donut sẽ chậm hơn đáng kể.

### 2. Chuẩn bị công cụ tải dữ liệu

- `MC-OCR` và `SROIE` phụ thuộc `kaggle` CLI nếu muốn tải tự động.
- Trên Windows, file credential Kaggle thường nằm ở:
  - `C:\Users\<ten_user>\.kaggle\kaggle.json`
- Nếu chưa có `kaggle` CLI, cài bằng:

```powershell
pip install kaggle
```

- `CORD v2` được tải qua thư viện `datasets`, đã có trong `requirements.txt`.

Kết quả mong đợi:
- Bạn có thể chạy `kaggle --help` trong PowerShell nếu Kaggle CLI đã sẵn sàng.
- File `kaggle.json` đã tồn tại đúng chỗ nếu muốn tải MC-OCR/SROIE tự động.

### 3. Tải dữ liệu

- Tải toàn bộ dataset:

```powershell
python scripts/download_data.py --dataset all --output data/
```

- Hoặc tải riêng từng dataset:

```powershell
python scripts/download_data.py --dataset mcocr --output data/
python scripts/download_data.py --dataset cord --output data/
python scripts/download_data.py --dataset sroie --output data/
```

Lưu ý:
- Nếu `SROIE` không tải tự động được, hãy tải thủ công rồi giải nén vào `data/sroie/`.
- Với `MC-OCR`, dữ liệu raw sau khi tải cần nằm trong `data/mc-ocr/raw/`.

Sau bước này, bạn nên có cấu trúc tối thiểu như sau:

```text
data/
├── mc-ocr/
│   └── raw/
├── cord-v2/
└── sroie/
    ├── img/
    └── key/
```

Kết quả mong đợi:
- `data/mc-ocr/raw/` có ảnh và CSV annotation.
- `data/cord-v2/` đã được lưu về từ HuggingFace.
- `data/sroie/` có đúng hai thư mục `img/` và `key/`.

### 4. Tiền xử lý dữ liệu

#### 4.1. Convert MC-OCR sang `donut_format`

```powershell
python scripts/convert_mcocr.py `
  --input data/mc-ocr/raw/ `
  --output data/mc-ocr/donut_format/ `
  --split-ratio 0.8 0.1 0.1
```

Lưu ý:
- Script sẽ tự tìm CSV, in `sample record`, in mapping cột và hỏi xác nhận.
- Chỉ dùng `--force` khi bạn đã chắc mapping cột đúng:

```powershell
python scripts/convert_mcocr.py `
  --input data/mc-ocr/raw/ `
  --output data/mc-ocr/donut_format/ `
  --split-ratio 0.8 0.1 0.1 `
  --force
```

Kết quả mong đợi:
- Tạo ra:

```text
data/mc-ocr/donut_format/
├── train/
├── val/
└── test/
```

- Mỗi split chứa ảnh và file `metadata.jsonl`.

#### 4.2. Convert SROIE sang `donut_format`

```powershell
python scripts/convert_sroie.py `
  --input data/sroie/ `
  --output data/sroie/donut_format/
```

Lưu ý:
- Script kỳ vọng dữ liệu SROIE có cấu trúc:
  - `data/sroie/img/`
  - `data/sroie/key/`
- Nếu có file key parse lỗi, script sẽ cảnh báo số file bị bỏ qua.

Kết quả mong đợi:
- Tạo ra:

```text
data/sroie/donut_format/
├── train/
├── val/
└── test/
```

- Mỗi split chứa ảnh và file `metadata.jsonl`.

### 5. EDA / sanity check dữ liệu

Chạy EDA cho dữ liệu MC-OCR đã convert:

```powershell
python scripts/eda.py `
  --data-dir data/mc-ocr/donut_format/ `
  --output docs/eda_figures/
```

Kết quả mong đợi:
- Thư mục `docs/eda_figures/` có các file như:
  - `split_counts.png`
  - `field_distribution.png`
  - `text_lengths.png`
  - `image_sizes.png`
  - `samples.png`

Bạn nên kiểm tra:
- Số lượng split có hợp lý không.
- Các field có phân phối quá lệch hay không.
- `samples.png` có đúng ảnh và annotation mong muốn hay không.

### 6. Chạy E1 — PaddleOCR baseline

```powershell
python scripts/baseline_paddleocr.py `
  --test-dir data/mc-ocr/donut_format/test/ `
  --output results/e1_baseline/
```

Kết quả mong đợi:
- Tạo thư mục `results/e1_baseline/`.
- Có ít nhất hai file:
  - `results/e1_baseline/metrics.json`
  - `results/e1_baseline/predictions.json`

Ý nghĩa:
- `metrics.json` chứa Precision, Recall, F1 của baseline.
- `predictions.json` chứa text OCR, dự đoán rule-based và ground truth để bạn debug.

### 7. Chạy E2 — Donut fine-tune

#### 7.1. Warm-up trên CORD v2

```powershell
python scripts/train_donut.py --config configs/donut_cord.yaml
```

Lưu ý:
- Warm-up CORD hiện chọn checkpoint tốt nhất theo `val_loss`, không phải theo F1.
- Checkpoint dự kiến được lưu tại:
  - `results/e2_donut/checkpoints/cord_warmup`

Kết quả mong đợi:
- Sinh log:
  - `results/e2_donut/cord_warmup_log.csv`
- Sinh checkpoint warm-up:
  - `results/e2_donut/checkpoints/cord_warmup`

#### 7.2. Fine-tune trên MC-OCR

```powershell
python scripts/train_donut.py --config configs/donut_mcocr.yaml
```

Kết quả mong đợi:
- Sinh log:
  - `results/e2_donut/training_log.csv`
- Sinh checkpoint:
  - `results/e2_donut/checkpoints/mcocr`

#### 7.3. Evaluate checkpoint MC-OCR

```powershell
python scripts/evaluate.py `
  --checkpoint results/e2_donut/checkpoints/mcocr `
  --test-dir data/mc-ocr/donut_format/test/ `
  --output results/e2_donut/metrics.json
```

Kết quả mong đợi:
- Có file:
  - `results/e2_donut/metrics.json`
- File này ngoài Precision/Recall/F1 còn có thêm:
  - `avg_inference_ms`

### 8. Chạy E3 — Cross-dataset SROIE

Nếu bạn chưa convert SROIE, hãy chạy lại bước convert trước:

```powershell
python scripts/convert_sroie.py `
  --input data/sroie/ `
  --output data/sroie/donut_format/
```

Sau đó chạy E3:

```powershell
python scripts/train_donut_sroie.py --config configs/donut_sroie.yaml
```

Kết quả mong đợi:
- Sinh checkpoint:
  - `results/e3_cross/checkpoints`
- Sinh log:
  - `results/e3_cross/training_log.csv`
- Sinh phân tích lỗi:
  - `results/e3_cross/error_analysis.json`

### 9. Cách đọc kết quả sau khi chạy

Các file quan trọng bạn nên kiểm tra:

- E1:
  - `results/e1_baseline/metrics.json`
  - `results/e1_baseline/predictions.json`
- E2:
  - `results/e2_donut/cord_warmup_log.csv`
  - `results/e2_donut/training_log.csv`
  - `results/e2_donut/checkpoints/mcocr`
  - `results/e2_donut/metrics.json`
- E3:
  - `results/e3_cross/training_log.csv`
  - `results/e3_cross/checkpoints`
  - `results/e3_cross/error_analysis.json`

Khi đọc kết quả, hãy đối chiếu:
- `F1`, `Precision`, `Recall` với mục tiêu trong `PROJECT.md`.
- `avg_inference_ms` với bảng so sánh kết quả nếu bạn muốn bổ sung tốc độ inference cho báo cáo.

### 10. Các lỗi thường gặp

#### Không tìm thấy `kaggle`

Nguyên nhân:
- Chưa cài Kaggle CLI hoặc chưa nằm trong PATH.

Cách xử lý:

```powershell
pip install kaggle
```

#### Thiếu `kaggle.json` trên Windows

Nguyên nhân:
- Chưa cấu hình credential Kaggle.

Cách xử lý:
- Tạo thư mục:
  - `C:\Users\<ten_user>\.kaggle\`
- Đặt file `kaggle.json` vào đó.

#### SROIE không có đúng thư mục `img/` và `key/`

Nguyên nhân:
- Giải nén sai cấp thư mục.

Cách xử lý:
- Đảm bảo sau khi giải nén có:
  - `data/sroie/img/`
  - `data/sroie/key/`

#### MC-OCR có nhiều CSV và mapping sai

Nguyên nhân:
- Raw data có nhiều CSV hoặc header không đúng kỳ vọng.

Cách xử lý:
- Chạy convert không dùng `--force`.
- Đọc kỹ `sample record` và `mapping`.
- Xóa hoặc di chuyển CSV không đúng trước khi convert lại.

#### Không có GPU hoặc thiếu VRAM khi train Donut

Nguyên nhân:
- Máy không có GPU hoặc bộ nhớ GPU không đủ.

Cách xử lý:
- Vẫn có thể chạy CPU để test flow.
- Nếu train thật, cân nhắc giảm `batch_size` trong config hoặc dùng máy có GPU mạnh hơn.

#### Không tìm thấy checkpoint warm-up trước khi fine-tune MC-OCR

Nguyên nhân:
- Warm-up CORD chưa chạy xong hoặc chưa sinh checkpoint.

Cách xử lý:
- Kiểm tra:
  - `results/e2_donut/checkpoints/cord_warmup`
- Nếu chưa có, chạy lại:

```powershell
python scripts/train_donut.py --config configs/donut_cord.yaml
```
