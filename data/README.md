# 📂 Dữ liệu MC-OCR Nội Bộ

Thư mục này được sử dụng để chứa tập dữ liệu MC-OCR gốc trực tiếp mà không cần mount qua Google Drive ngoài. Việc cấu hình cục bộ giúp dự án dễ dàng đóng gói (dockerize) hoặc người khác tái sử dụng hơn.

## 📥 Tự động lấy dữ liệu (Auto-Download)

Nếu bạn vừa Clone repo trống này về ở một Server mới, bạn chỉ cần chạy lệnh:
```bash
python -m src.preprocessing.download_dataset
```

*(Lưu ý: Để lệnh trên chạy thành công, chủ nhân repo cần phải đặt ID của file ZIP Google Drive public vào trong code file `download_dataset.py` trước)*

## 💡 Hướng dẫn tạo cục nén tải tự động (Dành cho Chủ Repo)

Đễ hỗ trợ lệnh auto-dowload phía trên:
1. Bạn hãy nộp chung thư mục `train_images/` và file CSV `train_df.csv` vào một tệp nén duy nhất (VD: `mcocr_dataset.zip`).
   - *Đảm bảo khi giải nén ra sẽ thấy ngay `train_df.csv` chứ không bị bọc thêm 1 cấp folder nữa.*
2. Cầm file ZIP đó up lên Google Drive của bạn.
3. Chỉnh quyền truy cập Drive thành **"Bất kỳ ai có liên kết"** (Anyone with the link).
4. Sao chép cái Link dài dòng đó. Tìm chuỗi ký tự dài ngoằng gọi là ID rác. (VD: `1pB-g88yZ4_...`)
5. Mở tệp tin `src/preprocessing/download_dataset.py`, dán ID này vào biến `FILE_ID`.

Và từ rày về sau, ai lấy code của bạn cũng chỉ cần gõ 1 lệnh là xong!

## Gợi ý cấu trúc đích sau khi giải nén tự động:
```text
data/
└── mcocr/
    ├── train_images/
    │   ├── img_001.jpg
    │   └── ...
    └── train_df.csv
```