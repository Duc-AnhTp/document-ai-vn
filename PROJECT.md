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

> **Lưu ý về format Ground Truth:** Annotation gốc MC-OCR 2021 bao gồm cả cụm nhãn hiển thị trong giá trị trường (ví dụ: `date: "Ngày bán: 15/08/2020"`, `total: "TỔNG TIỀN PHẢI T.TOÁN 6.000"`). Đây là đặc thù của bộ dữ liệu gốc, không phải lỗi converter. Model cần học sinh lại toàn bộ chuỗi bao gồm prefix, và metric exact-match vẫn công bằng vì cả pred lẫn gold đều so sánh cùng format.

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

### 2.4 SynthDoG-VI — Augmentation [ĐÃ LOẠI BỎ KHỎI SCOPE]

*Lưu ý: Đã loại bỏ phần này để giảm thiểu độ phức tạp của dự án và tập trung tối ưu hóa trên dữ liệu thực tế MC-OCR 2021.*

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
- **Augmentation:** Không dùng (sử dụng cấu hình mặc định của Donut)
- **Kỳ vọng F1:** ≥ 0.80 (mục tiêu chính)
- **Đáp ứng:** TC1 (mô hình chính) · TC3 (kiến trúc end-to-end)

### E3 — Donut Fine-tune trên SROIE (Cross-dataset)

- **Phương pháp:** Dùng lại Donut đã train ở E2 → fine-tune thêm trên SROIE 2019
- **Phân tích:** So sánh khả năng generalize sang receipt tiếng Anh · Error analysis cross-lingual · Cross-Attention Visualization
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
│        → MC-OCR 2021                                        │
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
│        │ Error analysis + Cross-Attention                   │
│        │ Visualization                                      │
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
    │ tham chiếu  │ │ Attention Viz   │ │                  │
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
│   └── sroie/               # SROIE 2019
├── notebooks/
│   ├── 01_eda.ipynb         # EDA: phân phối, visualize mẫu
│   └── 02_convert.ipynb     # Convert annotation → gt_parse
├── scripts/
│   ├── convert_mcocr.py     # Convert MC-OCR → Donut format
│   ├── convert_sroie.py     # Convert SROIE → Donut format
│   ├── download_data.py     # Tải dataset tự động
│   ├── eda.py               # EDA dataset
│   ├── train_donut.py       # Training script E2
│   ├── train_donut_sroie.py # Training script E3 + error analysis
│   ├── evaluate.py          # Evaluation F1/P/R
│   ├── baseline_paddleocr.py # E1 PaddleOCR baseline
│   ├── visualize_attention.py # Cross-attention visualization
│   └── utils.py             # Hàm tiện ích dùng chung
├── configs/
│   ├── donut_cord.yaml      # Config warm-up CORD
│   ├── donut_mcocr.yaml     # Config fine-tune MC-OCR
│   └── donut_sroie.yaml     # Config cross-dataset SROIE
├── tests/
│   └── test_utils.py        # Unit tests cho utils.py
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
| E2          | CORD + MC-OCR       | MC-OCR test | —     | —         | —      | —              |
| E3          | E2 + SROIE          | SROIE test  | —     | —         | —      | —              |

---

## 9. Tham khảo

- [Donut — OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664)
- [Donut GitHub](https://github.com/clovaai/donut)
- [MC-OCR 2021 trên Kaggle](https://kaggle.com/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021)
- [CORD v2 trên HuggingFace](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
- [SROIE 2019](https://rrc.cvc.uab.es/?ch=13)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
