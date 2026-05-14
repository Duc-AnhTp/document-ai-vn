# Document AI — Key Information Extraction trên Biên lai Tiếng Việt

> Dự án fine-tune mô hình **Donut** (end-to-end document understanding) để trích xuất thông tin từ biên lai chụp điện thoại tiếng Việt, kèm so sánh với baseline PaddleOCR và thí nghiệm cross-dataset trên SROIE.

---

## 1. Mục tiêu Metric

| Metric    | Ngưỡng yêu cầu |
|-----------|-----------------|
| F1 (KIE)  | ≥ 0.80          |
| Precision | ≥ 0.75          |
| Recall    | ≥ 0.78          |

---

## 2. Dataset

### 2.1 MC-OCR 2021 — Dataset chính

| Thuộc tính   | Chi tiết                                                              |
|--------------|-----------------------------------------------------------------------|
| Ngôn ngữ     | Tiếng Việt                                                            |
| Số lượng     | ~2.000 ảnh biên lai chụp điện thoại (unconstrained)                   |
| Annotation   | KIE annotation JSON                                                   |
| Trường trích xuất | Tên cửa hàng · Ngày · Tổng tiền · Địa chỉ                      |
| Nguồn        | [kaggle.com/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021](https://kaggle.com/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021) |

### 2.2 CORD v2 — Warm-up / Baseline

| Thuộc tính   | Chi tiết                                                              |
|--------------|-----------------------------------------------------------------------|
| Ngôn ngữ     | Tiếng Anh                                                            |
| Số lượng     | 1.000 ảnh receipt                                                     |
| Annotation   | JSON phân cấp                                                         |
| Nguồn        | [HuggingFace: naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) |
| Vai trò      | Dataset gốc Donut dùng để benchmark; dùng warm-up trước khi fine-tune tiếng Việt |

### 2.3 SROIE 2019 — So sánh cross-dataset

| Thuộc tính   | Chi tiết                                                              |
|--------------|-----------------------------------------------------------------------|
| Ngôn ngữ     | Tiếng Anh                                                            |
| Số lượng     | 626 ảnh receipt · 4 trường KIE đơn giản                              |
| Vai trò      | Nhẹ, huấn luyện nhanh, kết quả so sánh rõ — dùng cho thí nghiệm E3 cross-dataset |

### 2.4 SynthDoG-VI — Augmentation

| Thuộc tính   | Chi tiết                                                              |
|--------------|-----------------------------------------------------------------------|
| Loại         | Sinh ảnh tổng hợp tự động (script có sẵn trong repo Donut)           |
| Cách dùng    | Font tiếng Việt + template hóa đơn nội địa                           |
| Vai trò      | Tăng data khi MC-OCR chỉ ~2.000 ảnh                                  |

---

## 3. Thí nghiệm so sánh (đáp ứng TC4)

### E1 — PaddleOCR + Rule-based KIE (Baseline)

- **Phương pháp:** Chạy PaddleOCR lấy text → regex/rule trích xuất entity
- **Train trên:** MC-OCR 2021
- **Ưu điểm:** Nhanh, dễ cài, dùng làm điểm tham chiếu F1
- **Kỳ vọng F1:** ≈ 0.60–0.70
- **Đáp ứng:** TC1 (baseline so sánh) · TC3 (kiến trúc đơn giản, dễ giải thích)

### E2 — Donut Fine-tune (Mô hình chính)

- **Phương pháp:** Fine-tune `donut-base` trên CORD (warm-up) → MC-OCR 2021
- **Kỹ thuật:** Thêm Vietnamese tokens vào decoder
- **Augmentation:** SynthDoG-VI
- **Kỳ vọng F1:** ≥ 0.80 (mục tiêu chính)
- **Đáp ứng:** TC1 (mô hình chính) · TC2 (augmentation SynthDoG-VI) · TC3 (kiến trúc end-to-end)

### E3 — Donut Fine-tune trên SROIE (Cross-dataset)

- **Phương pháp:** Dùng lại Donut đã train ở E2 → fine-tune thêm trên SROIE 2019
- **Phân tích:** So sánh khả năng generalize sang receipt tiếng Anh · Error analysis cross-lingual · Grad-CAM
- **Đáp ứng:** TC4 (>2 thí nghiệm) · TC4 (error analysis cross-lingual)

---

## 4. Pipeline huấn luyện

```
┌─────────────────────────────────────────────────────────────┐
│                     LUỒNG HUẤN LUYỆN                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  E2 (Mô hình chính):                                       │
│    CORD v2 (warm-up)                                        │
│      → donut-base fine-tune                                 │
│        → MC-OCR 2021 + SynthDoG-VI                          │
│          → Evaluate F1, Precision, Recall                   │
│                                                             │
│  E1 (Baseline):                                             │
│    PaddleOCR                                                │
│      → MC-OCR 2021 rule-based KIE                           │
│        → So sánh với E2 (bảng F1)                           │
│                                                             │
│  E3 (Cross-dataset):                                        │
│    Donut (từ E2)                                            │
│      → SROIE 2019 fine-tune thêm                            │
│        → Error analysis + Grad-CAM                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Kiến trúc tổng thể

```
         ┌──────────────────────────┐
         │  MC-OCR 2021 + CORD v2   │
         └────┬─────────┬─────────┬─┘
              │         │         │
              ▼         ▼         ▼
    ┌─────────────┐ ┌─────────────────┐ ┌──────────────────┐
    │ E1           │ │ E2               │ │ E3                │
    │ PaddleOCR    │ │ Donut fine-tune  │ │ Donut + SROIE     │
    │ OCR + rule-  │ │ End-to-end,      │ │ Cross-dataset,    │
    │ based KIE    │ │ mô hình chính    │ │ generalize        │
    └──────┬──────┘ └────────┬────────┘ └────────┬───────────┘
           │                 │                    │
           ▼                 ▼                    ▼
    ┌─────────────┐ ┌─────────────────┐ ┌──────────────────┐
    │ F1 ≈ 0.60   │ │ F1 ≥ 0.80       │ │ F1 tiếng Anh     │
    │ –0.70       │ │ (mục tiêu)      │ │ Error analysis   │
    │ Baseline    │ │ Vietnamese +    │ │ cross-lingual    │
    │ tham chiếu  │ │ Grad-CAM        │ │                  │
    └──────┬──────┘ └────────┬────────┘ └────────┬───────────┘
           │                 │                    │
           └────────────┬────┘────────────────────┘
                        ▼
              ┌──────────────────────┐
              │ Bảng so sánh         │
              │ E1 vs E2 vs E3       │
              │ P / R / F1 /         │
              │ Inference time       │
              │ → TC1 + TC4          │
              └──────────────────────┘
```

---

## 6. Cấu trúc thư mục dự kiến

```
document-ai-vn/
├── data/
│   ├── mc-ocr/              # MC-OCR 2021 raw + processed
│   │   ├── raw/
│   │   └── donut_format/    # Converted gt_parse JSON
│   ├── cord-v2/             # CORD v2 từ HuggingFace
│   ├── sroie/               # SROIE 2019
│   └── synthdog-vi/         # Ảnh tổng hợp SynthDoG-VI
├── notebooks/
│   ├── 01_eda.ipynb         # EDA: phân phối, visualize mẫu
│   └── 02_convert.ipynb     # Convert annotation → gt_parse
├── scripts/
│   ├── convert_mcocr.py     # Script convert MC-OCR → Donut format
│   ├── train_donut.py       # Training script E2
│   ├── eval.py              # Evaluation F1/P/R
│   └── baseline_paddle.py   # E1 PaddleOCR baseline
├── configs/
│   └── donut_mcocr.yaml     # Config huấn luyện
├── results/
│   ├── e1_baseline/
│   ├── e2_donut/
│   └── e3_cross/
├── PROJECT.md               # File này
├── README.md
└── requirements.txt
```

---

## 7. Việc cần làm ngay (ưu tiên cao)

- [ ] **Tải dataset:**
  - MC-OCR 2021 từ Kaggle
  - CORD v2 từ HuggingFace (`naver-clova-ix/cord-v2`)
- [ ] **EDA:**
  - Đếm số ảnh per class
  - Kiểm tra phân phối field (tên cửa hàng, ngày, tổng tiền, địa chỉ)
  - Visualize 10–20 mẫu ảnh + annotation
- [ ] **Convert annotation MC-OCR → format `gt_parse` của Donut**
  - Đây là bước mất nhiều thời gian nhất
  - Format đầu ra: `{"gt_parse": {"store_name": "...", "date": "...", "total": "...", "address": "..."}}`
- [ ] **Setup môi trường:**
  - Cài `donut-python`, `transformers`, `datasets`
  - Kiểm tra GPU availability

---

## 8. Bảng so sánh kết quả (template)

| Thí nghiệm | Dataset train       | Dataset test | F1    | Precision | Recall | Inference (ms) |
|-------------|---------------------|-------------|-------|-----------|--------|----------------|
| E1          | MC-OCR 2021         | MC-OCR test | —     | —         | —      | —              |
| E2          | CORD + MC-OCR + Aug | MC-OCR test | —     | —         | —      | —              |
| E3          | E2 + SROIE          | SROIE test  | —     | —         | —      | —              |

---

## 9. Tham khảo

- [Donut — OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664)
- [Donut GitHub](https://github.com/clovaai/donut)
- [MC-OCR 2021 trên Kaggle](https://kaggle.com/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021)
- [CORD v2 trên HuggingFace](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
- [SROIE 2019](https://rrc.cvc.uab.es/?ch=13)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
