# Hướng dẫn Huấn luyện Donut trên Kaggle (GPU T4 x2)

Tài liệu này hướng dẫn chi tiết cách chạy huấn luyện mô hình Donut trên Kaggle trong khoảng 7 tiếng một cách an toàn, không lo bị ngắt kết nối mạng giữa chừng.

---

## 1. Các Cải Tiến Đã Thực Hiện Cho Code

Để chuẩn bị tốt nhất cho môi trường huấn luyện Kaggle (đặc biệt là khi chạy ngầm - Background Run), chúng tôi đã thực hiện các cải tiến sau trực tiếp vào mã nguồn:

1. **Tự động hóa bước xác nhận mapping trong [convert_mcocr.py](file:///d:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/scripts/convert_mcocr.py):**
   * **Vấn đề cũ:** Script yêu cầu nhập `y/n` từ bàn phím. Khi chạy ngầm trên Kaggle, việc này sẽ gây ra lỗi `EOFError` và dừng tiến trình.
   * **Giải pháp đã sửa:** Code hiện tại đã import module `sys` và kiểm tra `sys.stdin.isatty()`. Nếu phát hiện môi trường không tương tác (như Kaggle background run), script sẽ **tự động xác nhận và tiếp tục chạy** mà không bị crash. Bạn không cần lo lắng về việc quên truyền cờ `--force`.

2. **Tối ưu hóa thư viện phụ thuộc trong [requirements.txt](file:///d:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/requirements.txt):**
   * **Vấn đề cũ:** Gói `donut-python` khi cài đặt có thể cố cài đè một phiên bản PyTorch cũ hơn, gây xung đột với GPU Drivers và phiên bản PyTorch tối ưu sẵn trên Kaggle.
   * **Giải pháp đã sửa:** Qua kiểm tra, dự án tự xây dựng Dataloader và chỉ gọi trực tiếp từ Hugging Face `transformers` (không import gói `donut-python`). Chúng tôi đã comment gói này trong `requirements.txt` giúp quá trình cài đặt trên Kaggle diễn ra trơn tru và nhanh chóng.

3. **Cơ Chế Tiết Kiệm Dung Lượng Đĩa (Disk Space):**
   * Kaggle giới hạn `/kaggle/working` ở mức **20GB**. Nhờ cơ chế chỉ lưu checkpoint tốt nhất (`best`) và checkpoint cuối cùng (`last` để resume) trong [train_donut.py](file:///d:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/scripts/train_donut.py), dung lượng lưu trữ chỉ tốn tối đa khoảng **2.4GB**, hoàn toàn an toàn.

---

## 2. Hướng Dẫn Từng Bước Đưa Lên Kaggle Train Trong 7 Tiếng

Để chạy train ổn định trong thời gian dài trên Kaggle mà không sợ bị ngắt kết nối khi tắt trình duyệt, bạn **bắt buộc** phải sử dụng tính năng **Save Version -> Save & Run All (Commit)** để Kaggle chạy ngầm dưới server.

### Bước 1: Khởi tạo Notebook trên Kaggle
1. Truy cập [Kaggle](https://www.kaggle.com/) và tạo một Notebook mới.
2. Ở bảng điều khiển bên phải (Settings):
   * **Accelerator:** Chọn **GPU T4 x2** (để sử dụng 2 GPU song song).
   * **Internet on:** **Bật (ON)** (Bắt buộc để tải thư viện và tải model `donut-base` từ Hugging Face).

### Bước 2: Thêm Dataset MC-OCR 2021 gốc vào Notebook
Thay vì tự tải xuống bằng script rất mất thời gian, hãy tận dụng hạ tầng của Kaggle:
1. Nhấn nút **Add Input** (hoặc **Add Data**) ở góc trên bên phải Notebook.
2. Tìm kiếm dataset: `vietnamese-receipts-mc-ocr-2021` (của tác giả `domixi1989`).
3. Nhấn **Add** để thêm vào Notebook.
4. Đường dẫn chứa ảnh raw và file CSV lúc này thường sẽ là:
   `/kaggle/input/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021`

> [!WARNING]
> Nếu bạn gặp lỗi `FileNotFoundError: No such file or directory` ở bước này, hãy chạy lệnh sau trong một cell mới để kiểm tra tên thư mục thực tế của dataset trong Kaggle:
> ```bash
> !ls /kaggle/input
> ```
> Kết quả lệnh trên sẽ hiển thị tên thư mục chính xác (ví dụ: `vietnamese-receipts-mc-ocr-2021` hoặc `mcocr2021`). Hãy thay thế tên thư mục này vào tham số `--input` ở Bước 4.

### Bước 3: Đồng bộ mã nguồn từ nhánh `kaggle-donut-traning` lên Kaggle
Có 2 cách để đưa code lên Kaggle:
* **Cách 1 (Khuyên dùng - Clone trực tiếp từ GitHub):** Chạy lệnh clone chỉ định riêng nhánh `kaggle-donut-traning` từ repository của bạn trong ô code của Kaggle Notebook:
  ```bash
  !git clone -b kaggle-donut-traning https://github.com/Duc-AnhTp/document-ai-vn.git
  %cd document-ai-vn
  ```
* **Cách 2 (Upload thủ công):** Đảm bảo trên máy local bạn đang ở nhánh `kaggle-donut-traning` (`git checkout kaggle-donut-traning`), nén file zip toàn bộ source code (trừ các thư mục `.git`, `data`, `results`, `.venv`), sau đó upload trực tiếp lên Kaggle Notebook.

### Bước 4: Thiết lập môi trường và tiền xử lý dữ liệu trực tiếp trên Kaggle
Tạo một ô code mới (Code cell) trong Notebook để cài đặt thư viện và chuẩn bị dữ liệu:

```python
# 1. Cài đặt các thư viện cần thiết
!pip install -r requirements.txt

# 2. Kiểm tra các thư mục trong input để lấy đường dẫn chính xác
import os
print("Các thư mục hiện có trong /kaggle/input/:", os.listdir("/kaggle/input"))

# 3. Tạo thư mục chứa dữ liệu đầu ra sau khi convert
os.makedirs("/kaggle/working/data/mc-ocr/donut_format", exist_ok=True)

# 4. Chạy script convert dữ liệu (sửa lại tên thư mục input cho khớp với bước 2 nếu khác)
!python scripts/convert_mcocr.py \
    --input /kaggle/input/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021 \
    --output /kaggle/working/data/mc-ocr/donut_format \
    --split-ratio 0.8 0.1 0.1 \
    --force
```

### Bước 5: Chỉnh sửa cấu hình YAML phù hợp với đường dẫn Kaggle
Do dữ liệu sau convert sẽ nằm trong `/kaggle/working/data/mc-ocr/donut_format`, bạn chạy đoạn script python này trực tiếp trong Notebook trước khi train để cập nhật cấu hình:

```python
import yaml

# Đọc cấu hình Kaggle hiện tại
with open("configs/donut_mcocr_kaggle.yaml", "r") as f:
    config = yaml.safe_load(f)

# Cập nhật đường dẫn thực tế trên Kaggle
config["data"]["train_dir"] = "/kaggle/working/data/mc-ocr/donut_format/train"
config["data"]["val_dir"] = "/kaggle/working/data/mc-ocr/donut_format/val"
config["data"]["test_dir"] = "/kaggle/working/data/mc-ocr/donut_format/test"

# Lưu lại file cấu hình
with open("configs/donut_mcocr_kaggle.yaml", "w") as f:
    yaml.safe_dump(config, f)
```

### Bước 6: Kích hoạt quá trình huấn luyện
Viết các lệnh chạy train vào cell tiếp theo:

```python
# 1. Chạy bước khởi động (Warm-up) với tập CORD v2 (sử dụng 3 epochs mặc định)
!python scripts/train_donut.py --config configs/donut_cord_kaggle.yaml

# 2. Chạy Fine-tune chính thức trên tập MC-OCR 2021 (30 epochs)
!python scripts/train_donut.py --config configs/donut_mcocr_kaggle.yaml
```

> [!TIP]
> **Mẹo bỏ qua bước Warm-up để train nhanh hơn:**
> Nếu bạn muốn thử nghiệm nhanh mà không cần chạy bước warm-up CORD v2 (tiết kiệm thêm thời gian), bạn có thể cấu hình file `configs/donut_mcocr_kaggle.yaml` để fine-tune trực tiếp từ mô hình gốc:
> ```python
> config["model"]["pretrained"] = "naver-clova-ix/donut-base"
> ```
> Khi đó, bạn chỉ cần chạy bước 2 (Fine-tune MC-OCR) mà không cần chạy bước 1.

### Bước 7: Đánh giá mô hình sau khi huấn luyện xong
```python
# 3. Đánh giá chất lượng mô hình trên tập test và lưu metric
!python scripts/evaluate.py \
    --checkpoint /kaggle/working/results/e2_donut/checkpoints/mcocr \
    --test-dir /kaggle/working/data/mc-ocr/donut_format/test \
    --output /kaggle/working/results/e2_donut/metrics.json \
    --task-prompt "<s_mcocr>"
```

---

## 3. Cách Treo Máy Chạy Nền (Chạy Off-line 7 Tiếng)

Để tránh việc máy tính bị sleep hoặc mất mạng làm gián đoạn:
1. Nhấn nút **Save Version** ở góc trên bên phải giao diện Kaggle Notebook.
2. Chọn **Save & Run All (Commit)** trong bảng hiện ra.
3. Nhấn nút **Save**.
4. Lúc này Kaggle sẽ mở một session chạy ngầm hoàn toàn độc lập. Bạn có thể tắt máy tính, tắt trình duyệt và đi ngủ.
5. Để kiểm tra tiến trình: Nhấn vào biểu tượng thông báo/Viewer ở góc dưới bên trái màn hình Kaggle hoặc truy cập vào phần **Active Events** để xem log in ra theo thời gian thực (Real-time logs).

Sau khi chạy xong (thường mất từ 1.5 đến 3 tiếng cho toàn bộ pipeline bao gồm cả warm-up và fine-tune), toàn bộ file checkpoints và file kết quả `metrics.json` sẽ nằm trong phần **Output** của Notebook. Bạn có thể vào đó tải về máy cục bộ.
