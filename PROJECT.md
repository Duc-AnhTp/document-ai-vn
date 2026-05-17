# PROJECT.md

# OCR-based Rule Scoring Pipeline for Vietnamese Receipt Key Information Extraction

## 1. Giới thiệu

Đồ án này xây dựng một hệ thống trích xuất thông tin quan trọng từ ảnh hóa đơn tiếng Việt, tập trung trên bộ dữ liệu **MC-OCR2021**. Thay vì sử dụng mô hình end-to-end phức tạp như Donut làm hướng chính, hệ thống được thiết kế theo hướng **pipeline đơn giản, dễ triển khai, dễ debug và dễ demo**.

Bài toán chính là nhận ảnh hóa đơn đầu vào và trích xuất 4 trường thông tin quan trọng:

```text
SELLER
SELLER_ADDRESS
TIMESTAMP
TOTAL_COST
```

Output cuối cùng được chuẩn hóa theo định dạng:

```text
SELLER|||SELLER_ADDRESS|||TIMESTAMP|||TOTAL_COST
```

Ví dụ:

```text
MINIMART ANAN|||Chợ Sủi Phú Thị Gia Lâm|||09/08/2020 09:26|||115000
```

---

## 2. Mục tiêu đồ án

### 2.1. Mục tiêu chính

Xây dựng một pipeline OCR-based có khả năng:

- Nhận ảnh hóa đơn tiếng Việt chụp bằng điện thoại.
- Phát hiện và nhận dạng các dòng chữ trên hóa đơn.
- Sắp xếp các dòng OCR theo đúng thứ tự đọc.
- Trích xuất 4 trường thông tin chính:
  - Tên cửa hàng/người bán.
  - Địa chỉ người bán.
  - Thời gian giao dịch.
  - Tổng tiền thanh toán.
- Chuẩn hóa kết quả đầu ra.
- Đánh giá bằng các metric phù hợp như CER, Exact Match, Precision, Recall, F1.
- Cung cấp giao diện demo đơn giản bằng Streamlit hoặc Gradio.

### 2.2. Mục tiêu phụ

- So sánh nhiều biến thể pipeline để thỏa mãn yêu cầu thực nghiệm.
- Phân tích lỗi theo từng field.
- Trực quan hóa bounding box OCR và kết quả trích xuất.
- Tổ chức code rõ ràng, dễ tái sử dụng và dễ mở rộng.

---

## 3. Bài toán

### 3.1. Input

Input của hệ thống là một ảnh hóa đơn ở định dạng:

```text
.jpg
.jpeg
.png
```

Ảnh có thể có các đặc điểm:

- Chụp bằng điện thoại.
- Bị nghiêng nhẹ.
- Bị mờ hoặc nhiễu.
- Nền phức tạp.
- Ánh sáng không đều.
- Hóa đơn bị nhăn, cong hoặc bị che một phần.

### 3.2. Output

Output của hệ thống gồm hai dạng.

#### Dạng JSON dùng cho demo

```json
{
  "SELLER": "MINIMART ANAN",
  "SELLER_ADDRESS": "Chợ Sủi Phú Thị Gia Lâm",
  "TIMESTAMP": "09/08/2020 09:26",
  "TOTAL_COST": "115000"
}
```

#### Dạng text theo format MC-OCR

```text
MINIMART ANAN|||Chợ Sủi Phú Thị Gia Lâm|||09/08/2020 09:26|||115000
```

Nếu một field không tìm được, để trống field đó:

```text
MINIMART ANAN||||||115000
```

Tương ứng với:

```text
SELLER = MINIMART ANAN
SELLER_ADDRESS = ""
TIMESTAMP = ""
TOTAL_COST = 115000
```

---

## 4. Tính mới của đồ án

Đồ án không đặt mục tiêu đề xuất một kiến trúc deep learning hoàn toàn mới. Tính mới nằm ở việc thiết kế một pipeline thực dụng, phù hợp với dữ liệu hóa đơn tiếng Việt chụp bằng điện thoại.

Các điểm đóng góp chính:

1. **Pipeline nhẹ và dễ triển khai cho tiếng Việt**

   Hệ thống kết hợp OCR, thông tin vị trí dòng, rule scoring và hậu xử lý thay vì phụ thuộc hoàn toàn vào một mô hình end-to-end khó kiểm soát.

2. **Khai thác đồng thời text và layout**

   Mỗi dòng OCR không chỉ được xét theo nội dung text mà còn theo vị trí trên hóa đơn, thứ tự dòng và mối liên hệ với các keyword đặc trưng.

3. **Rule scoring chuyên biệt cho hóa đơn tiếng Việt**

   Các trường như tổng tiền, ngày giờ, địa chỉ và tên cửa hàng được trích xuất bằng hệ thống chấm điểm dựa trên keyword, regex và vị trí.

4. **Post-processing phù hợp với dữ liệu Việt Nam**

   Hệ thống chuẩn hóa Unicode, khoảng trắng, tiền tệ, ngày tháng và các lỗi OCR phổ biến.

5. **Dễ phân tích lỗi và cải tiến**

   Vì hệ thống được chia thành nhiều module rõ ràng, có thể xác định lỗi đến từ OCR, sắp xếp dòng, rule extraction hay post-processing.

---

## 5. Tổng quan pipeline

Pipeline chính của hệ thống:

```text
Ảnh hóa đơn
  ↓
Preprocessing nhẹ
  ↓
PaddleOCR
  ↓
Chuẩn hóa OCR result
  ↓
Sort OCR lines từ trên xuống dưới, trái sang phải
  ↓
Thêm layout features cho từng dòng
  ↓
Rule-based / keyword scoring extractor
  ↓
Post-processing
  ↓
Output JSON + output MC-OCR format
  ↓
Evaluation + visualization + demo
```

---

## 6. Kiến trúc hệ thống

### 6.1. Các module chính

```text
src/
├── preprocessing.py
├── ocr.py
├── line_processing.py
├── extractor.py
├── postprocess.py
├── evaluate.py
├── visualize.py
└── demo/
    └── app.py
```

### 6.2. Vai trò từng module

| Module | Vai trò |
|---|---|
| `preprocessing.py` | Resize ảnh, convert RGB, tăng contrast nhẹ, xử lý ảnh đầu vào |
| `ocr.py` | Chạy PaddleOCR và trả về text, confidence, bounding box |
| `line_processing.py` | Sắp xếp OCR boxes, thêm layout features |
| `extractor.py` | Chấm điểm từng dòng và trích xuất 4 field |
| `postprocess.py` | Chuẩn hóa text, tiền tệ, ngày tháng |
| `evaluate.py` | Tính CER, Exact Match, Precision, Recall, F1 |
| `visualize.py` | Vẽ OCR boxes và highlight field dự đoán |
| `demo/app.py` | Giao diện upload ảnh và hiển thị kết quả |

---

## 7. Preprocessing

### 7.1. Mục tiêu

Preprocessing được giữ ở mức đơn giản để tránh làm mất thông tin chữ. Mục tiêu chính là đưa ảnh về định dạng ổn định cho OCR.

### 7.2. Các bước xử lý

```text
1. Đọc ảnh bằng OpenCV hoặc PIL.
2. Convert ảnh sang RGB.
3. Resize ảnh nếu cạnh lớn nhất vượt quá ngưỡng.
4. Tăng contrast nhẹ nếu cần.
5. Trả về ảnh đã xử lý.
```

### 7.3. Pseudocode

```python
import cv2

def preprocess_image(image_path: str, max_side: int = 1600):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]

    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale)

    return image
```

### 7.4. Lý do chọn preprocessing nhẹ

- Dữ liệu hóa đơn có nhiều font và nền khác nhau.
- Xử lý quá mạnh có thể làm mất dấu tiếng Việt.
- OCR engine hiện đại thường đã có khả năng xử lý ảnh tương đối tốt.
- Dễ so sánh thực nghiệm giữa có và không có preprocessing.

---

## 8. OCR bằng PaddleOCR

### 8.1. Lý do chọn PaddleOCR

PaddleOCR được chọn vì:

- Dễ cài đặt.
- Có API Python trực tiếp.
- Có khả năng phát hiện text và nhận dạng text trong cùng một pipeline.
- Phù hợp để dựng baseline nhanh.
- Dễ tích hợp với demo.

### 8.2. Output của OCR

Mỗi dòng OCR được chuẩn hóa thành dictionary:

```json
{
  "text": "Tổng thanh toán 115.000đ",
  "conf": 0.94,
  "bbox": [x1, y1, x2, y2]
}
```

Trong đó:

| Trường | Ý nghĩa |
|---|---|
| `text` | Nội dung text nhận dạng được |
| `conf` | Confidence score từ OCR |
| `bbox` | Bounding box của dòng text theo dạng `[x1, y1, x2, y2]` |

### 8.3. Pseudocode OCR

```python
from paddleocr import PaddleOCR

ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)

def run_ocr(image_path: str):
    result = ocr_engine.ocr(image_path, cls=True)

    lines = []

    for page in result:
        for item in page:
            box = item[0]
            text = item[1][0]
            score = item[1][1]

            xs = [p[0] for p in box]
            ys = [p[1] for p in box]

            lines.append({
                "text": text,
                "conf": score,
                "bbox": [min(xs), min(ys), max(xs), max(ys)]
            })

    return lines
```

### 8.4. Ghi chú về tiếng Việt

Nếu PaddleOCR nhận dạng tiếng Việt chưa tốt, có thể thử:

```text
- Thay `lang="en"` bằng cấu hình hỗ trợ tiếng Việt nếu phiên bản cài đặt có hỗ trợ.
- Dùng PaddleOCR để detect text, sau đó crop từng dòng và nhận dạng bằng VietOCR.
- Giữ pipeline rule scoring phía sau, chỉ thay OCR engine.
```

---

## 9. Xử lý OCR lines

### 9.1. Vấn đề

OCR engine có thể trả về các dòng text không đúng thứ tự đọc tự nhiên. Nếu không sắp xếp lại, các rule như “seller nằm ở đầu hóa đơn” hoặc “total nằm ở cuối hóa đơn” sẽ hoạt động kém.

### 9.2. Sort OCR lines

Các dòng được sort theo:

```text
1. Tọa độ y từ trên xuống dưới.
2. Nếu y gần nhau, sort theo x từ trái sang phải.
```

Pseudocode:

```python
def sort_lines(lines):
    return sorted(lines, key=lambda x: (x["bbox"][1], x["bbox"][0]))
```

### 9.3. Thêm layout features

Sau khi sort, mỗi dòng được thêm các feature:

```text
line_id
relative_y
x_center
y_center
width
height
text_lower
```

Pseudocode:

```python
def add_line_features(lines, image_height):
    lines = sort_lines(lines)
    n = len(lines)

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line["bbox"]

        line["line_id"] = i
        line["x_center"] = (x1 + x2) / 2
        line["y_center"] = (y1 + y2) / 2
        line["width"] = x2 - x1
        line["height"] = y2 - y1
        line["relative_y"] = line["y_center"] / image_height
        line["text_lower"] = line["text"].lower()

    return lines
```

---

## 10. Rule-based Scoring Extractor

### 10.1. Ý tưởng chính

Thay vì dùng rule cứng, hệ thống dùng **scoring function** cho từng field. Mỗi dòng OCR được chấm điểm xem nó có khả năng thuộc field nào.

Các field:

```text
SELLER
SELLER_ADDRESS
TIMESTAMP
TOTAL_COST
```

Dòng có điểm cao nhất cho từng field sẽ được chọn làm kết quả dự đoán.

---

## 11. Trích xuất TOTAL_COST

### 11.1. Đặc điểm của TOTAL_COST

Dòng tổng tiền thường có các đặc điểm:

- Nằm ở nửa dưới hóa đơn.
- Có keyword như:
  - `tổng`
  - `thanh toán`
  - `phải trả`
  - `thành tiền`
  - `total`
  - `amount`
- Có số tiền.
- Có ký hiệu tiền tệ như `đ`, `vnd`, `vnđ`.

### 11.2. Keyword

```python
TOTAL_KEYWORDS = [
    "tổng", "tong",
    "thanh toán", "thanh toan",
    "phải trả", "phai tra",
    "thành tiền", "thanh tien",
    "total", "amount",
    "cộng", "cong"
]
```

### 11.3. Scoring function

```python
import re

def money_score(text: str) -> int:
    text_lower = text.lower()
    score = 0

    if any(k in text_lower for k in TOTAL_KEYWORDS):
        score += 3

    if re.search(r"\d{1,3}([.,]\d{3})+", text):
        score += 2

    if "đ" in text_lower or "vnd" in text_lower or "vnđ" in text_lower:
        score += 1

    return score
```

### 11.4. Extractor

```python
def extract_total_cost(lines):
    best_line = None
    best_score = -1

    for line in lines:
        score = money_score(line["text"])

        if line["relative_y"] > 0.5:
            score += 1

        if score > best_score:
            best_score = score
            best_line = line

    if best_line is None or best_score <= 0:
        return ""

    return best_line["text"]
```

---

## 12. Trích xuất TIMESTAMP

### 12.1. Đặc điểm của TIMESTAMP

Dòng thời gian thường có:

- Ngày/tháng/năm.
- Có thể có giờ/phút/giây.
- Keyword như:
  - `ngày`
  - `giờ`
  - `date`
  - `time`

### 12.2. Regex ngày giờ

```python
DATE_PATTERNS = [
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
    r"\d{1,2}\.\d{1,2}\.\d{2,4}",
    r"\d{1,2}\s*tháng\s*\d{1,2}\s*năm\s*\d{2,4}"
]

TIME_PATTERNS = [
    r"\d{1,2}:\d{2}(:\d{2})?"
]
```

### 12.3. Scoring function

```python
def timestamp_score(text: str) -> int:
    score = 0
    text_lower = text.lower()

    if any(re.search(p, text_lower) for p in DATE_PATTERNS):
        score += 3

    if any(re.search(p, text_lower) for p in TIME_PATTERNS):
        score += 2

    if any(k in text_lower for k in ["ngày", "ngay", "date", "time", "giờ", "gio"]):
        score += 1

    return score
```

### 12.4. Extractor

```python
def extract_timestamp(lines):
    best_line = None
    best_score = -1

    for line in lines:
        score = timestamp_score(line["text"])

        if score > best_score:
            best_score = score
            best_line = line

    if best_line is None or best_score <= 0:
        return ""

    return best_line["text"]
```

---

## 13. Trích xuất SELLER

### 13.1. Đặc điểm của SELLER

Tên cửa hàng/người bán thường:

- Nằm ở phần đầu hóa đơn.
- Có font lớn hơn các dòng khác.
- Không chứa số tiền.
- Không chứa ngày giờ.
- Không phải địa chỉ.
- Không quá dài.

### 13.2. Scoring function

```python
ADDRESS_KEYWORDS = [
    "đường", "duong",
    "phường", "phuong",
    "quận", "quan",
    "huyện", "huyen",
    "tỉnh", "tinh",
    "tp", "thành phố", "thanh pho",
    "số", "so ",
    "ngõ", "ngo"
]

def seller_score(line):
    text = line["text"]
    text_lower = text.lower()

    score = 0

    if line["relative_y"] < 0.35:
        score += 3

    if 3 <= len(text.split()) <= 8:
        score += 1

    if any(k in text_lower for k in ADDRESS_KEYWORDS):
        score -= 2

    if timestamp_score(text) > 0:
        score -= 3

    if money_score(text) > 0:
        score -= 3

    return score
```

### 13.3. Extractor

```python
def extract_seller(lines):
    candidates = lines[:min(10, len(lines))]

    best_line = None
    best_score = -999

    for line in candidates:
        score = seller_score(line)

        if score > best_score:
            best_score = score
            best_line = line

    if best_line is None:
        return ""

    return best_line["text"]
```

---

## 14. Trích xuất SELLER_ADDRESS

### 14.1. Đặc điểm của SELLER_ADDRESS

Địa chỉ người bán thường:

- Nằm gần phần đầu hóa đơn.
- Nằm gần dòng seller.
- Có keyword địa chỉ.
- Có thể dài hơn tên cửa hàng.
- Có thể bị OCR tách thành nhiều dòng.

### 14.2. Scoring function

```python
def address_score(line):
    text = line["text"]
    text_lower = text.lower()
    score = 0

    if any(k in text_lower for k in ADDRESS_KEYWORDS):
        score += 3

    if line["relative_y"] < 0.45:
        score += 1

    if timestamp_score(text) > 0:
        score -= 3

    if money_score(text) > 0:
        score -= 3

    return score
```

### 14.3. Extractor

```python
def extract_address(lines):
    candidates = lines[:min(15, len(lines))]

    best_line = None
    best_score = -999

    for line in candidates:
        score = address_score(line)

        if score > best_score:
            best_score = score
            best_line = line

    if best_line is None or best_score <= 0:
        return ""

    return best_line["text"]
```

### 14.4. Mở rộng: merge nhiều dòng địa chỉ

Một số địa chỉ có thể bị tách thành nhiều dòng. Có thể mở rộng bằng cách nối dòng hiện tại với dòng kế tiếp nếu:

```text
- Dòng kế tiếp vẫn nằm ở vùng đầu hóa đơn.
- Dòng kế tiếp không chứa ngày giờ.
- Dòng kế tiếp không chứa tiền.
- Dòng kế tiếp có keyword địa chỉ hoặc là phần tiếp theo của địa chỉ.
```

---

## 15. Post-processing

### 15.1. Mục tiêu

Post-processing giúp chuẩn hóa output và giảm lỗi do OCR.

Các bước chính:

```text
- Chuẩn hóa Unicode tiếng Việt.
- Xóa khoảng trắng thừa.
- Chuẩn hóa tiền tệ.
- Chuẩn hóa ngày giờ.
- Loại bỏ keyword không cần thiết trong field.
```

---

### 15.2. Chuẩn hóa text

```python
import re
import unicodedata

def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text
```

---

### 15.3. Chuẩn hóa tiền

Input có thể là:

```text
115.000đ
115,000 VNĐ
Tổng thanh toán: 115.000
Thanh toán 115000
```

Output mong muốn:

```text
115000
```

Pseudocode:

```python
def normalize_money(text: str) -> str:
    if text is None:
        return ""

    text = text.lower()
    text = text.replace("vnđ", "")
    text = text.replace("vnd", "")
    text = text.replace("đ", "")

    numbers = re.findall(r"\d[\d.,]*", text)

    if not numbers:
        return ""

    value = max(numbers, key=len)
    value = re.sub(r"[^\d]", "", value)

    return value
```

---

### 15.4. Chuẩn hóa ngày giờ

Có thể giữ nguyên dòng timestamp hoặc tách riêng ngày giờ nếu cần.

Ví dụ:

```python
def normalize_timestamp(text: str) -> str:
    text = normalize_text(text)

    date_match = None
    time_match = None

    for pattern in DATE_PATTERNS:
        m = re.search(pattern, text.lower())
        if m:
            date_match = m.group(0)
            break

    for pattern in TIME_PATTERNS:
        m = re.search(pattern, text.lower())
        if m:
            time_match = m.group(0)
            break

    if date_match and time_match:
        return f"{date_match} {time_match}"

    if date_match:
        return date_match

    return text
```

---

## 16. Hàm tổng hợp extraction

```python
def extract_fields(lines, image_height):
    lines = add_line_features(lines, image_height)

    seller = extract_seller(lines)
    address = extract_address(lines)
    timestamp = extract_timestamp(lines)
    total_cost = extract_total_cost(lines)

    fields = {
        "SELLER": normalize_text(seller),
        "SELLER_ADDRESS": normalize_text(address),
        "TIMESTAMP": normalize_timestamp(timestamp),
        "TOTAL_COST": normalize_money(total_cost)
    }

    return fields
```

Format output:

```python
def format_mcocr_output(fields):
    return "|||".join([
        fields.get("SELLER", ""),
        fields.get("SELLER_ADDRESS", ""),
        fields.get("TIMESTAMP", ""),
        fields.get("TOTAL_COST", "")
    ])
```

---

## 17. Đánh giá mô hình

### 17.1. Metric sử dụng

Hệ thống được đánh giá bằng các metric sau:

| Metric | Ý nghĩa |
|---|---|
| CER | Character Error Rate giữa prediction và ground truth |
| Exact Match | Field được xem là đúng nếu prediction khớp hoàn toàn ground truth sau normalize |
| Precision | Tỉ lệ field dự đoán đúng trên tổng field được dự đoán |
| Recall | Tỉ lệ field đúng được tìm thấy trên tổng field ground truth |
| F1 | Trung bình điều hòa giữa Precision và Recall |
| Macro-F1 | Trung bình F1 của 4 field |

### 17.2. Chuẩn hóa trước khi đánh giá

Trước khi tính metric, cả prediction và ground truth đều được normalize:

```text
- lowercase
- Unicode NFC
- bỏ khoảng trắng thừa
- bỏ ký tự đặc biệt không cần thiết
- chuẩn hóa tiền
- chuẩn hóa ngày giờ
```

### 17.3. Field-level Exact Match

Một field được tính là đúng nếu:

```text
normalize(prediction) == normalize(ground_truth)
```

Ví dụ:

```text
Prediction: 115.000đ
Ground truth: 115000
Sau normalize: 115000 == 115000
=> đúng
```

### 17.4. Precision / Recall / F1

Với mỗi field:

```text
TP: field có prediction và prediction đúng
FP: field có prediction nhưng prediction sai
FN: field ground truth có nhưng prediction thiếu hoặc sai
```

Công thức:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

Macro-F1:

```text
Macro-F1 = mean(F1_SELLER, F1_ADDRESS, F1_TIMESTAMP, F1_TOTAL_COST)
```

---

## 18. Thiết kế thực nghiệm

Để đáp ứng yêu cầu thực nghiệm và phân tích khoa học, đồ án thực hiện ít nhất 3 thí nghiệm.

---

### 18.1. Experiment 1: OCR + simple rule baseline

```text
PaddleOCR
→ sort lines
→ rule đơn giản
→ post-processing
```

Mục tiêu:

- Tạo baseline ban đầu.
- Đo hiệu quả của OCR + rule extraction.
- Làm mốc so sánh cho các thí nghiệm sau.

---

### 18.2. Experiment 2: OCR + preprocessing + rule scoring

```text
Preprocessing nhẹ
→ PaddleOCR
→ sort lines
→ rule scoring
→ post-processing
```

Mục tiêu:

- Đánh giá tác động của preprocessing.
- Đánh giá rule scoring so với rule đơn giản.
- Cải thiện độ chính xác của các field như `TIMESTAMP` và `TOTAL_COST`.

---

### 18.3. Experiment 3: OCR + rule scoring + lightweight classifier

Nếu còn thời gian, bổ sung một model học máy nhẹ.

```text
OCR lines
→ text features + layout features
→ TF-IDF + Logistic Regression / Random Forest
→ classify each line into:
   SELLER / SELLER_ADDRESS / TIMESTAMP / TOTAL_COST / OTHER
→ post-processing
```

Mục tiêu:

- Có thêm một model học máy để so sánh với rule-based method.
- Tăng tính thuyết phục về mặt machine learning.
- Đánh giá xem đặc trưng text + layout có cải thiện so với rule không.

---

### 18.4. Experiment 4: Donut fine-tuning

Nếu đã có thử nghiệm Donut, giữ lại như một experiment so sánh.

```text
Image
→ Donut
→ generate output string
→ post-processing
→ evaluation
```

Mục tiêu:

- So sánh hướng OCR-free end-to-end với hướng OCR-based modular pipeline.
- Phân tích vì sao Donut chưa phù hợp trong điều kiện dữ liệu nhỏ và tiếng Việt có dấu.
- Tăng chiều sâu báo cáo.

---

## 19. Bảng kết quả dự kiến

Bảng này là format báo cáo. Số liệu thực tế sẽ được điền sau khi chạy thí nghiệm.

| Pipeline | SELLER F1 | ADDRESS F1 | TIMESTAMP F1 | TOTAL_COST F1 | Macro-F1 | CER |
|---|---:|---:|---:|---:|---:|---:|
| OCR + simple rule | TBD | TBD | TBD | TBD | TBD | TBD |
| OCR + preprocessing + rule scoring | TBD | TBD | TBD | TBD | TBD | TBD |
| OCR + rule scoring + classifier | TBD | TBD | TBD | TBD | TBD | TBD |
| Donut fine-tuning | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 20. Error analysis

### 20.1. Mục tiêu

Phân tích lỗi giúp xác định hệ thống sai ở đâu và nên cải thiện module nào.

### 20.2. Các nhóm lỗi chính

| Field | Lỗi thường gặp | Nguyên nhân | Cách cải thiện |
|---|---|---|---|
| SELLER | Nhầm slogan/logo với tên cửa hàng | Dòng đầu hóa đơn không phải lúc nào cũng là seller | Dùng thêm keyword negative, OCR confidence, kích thước chữ |
| SELLER_ADDRESS | Thiếu một phần địa chỉ | Địa chỉ bị tách thành nhiều dòng | Merge các dòng gần nhau |
| TIMESTAMP | Nhầm số hóa đơn/số giao dịch với ngày giờ | Cùng dòng có nhiều số | Regex ngày giờ chặt hơn |
| TOTAL_COST | Nhầm subtotal/VAT với tổng thanh toán | Nhiều dòng chứa tiền | Ưu tiên keyword “tổng”, “thanh toán”, “phải trả” |
| Tất cả | OCR sai ký tự hoặc mất dấu | Ảnh mờ, nghiêng, thiếu sáng | Preprocessing, thử OCR engine khác |
| Tất cả | Sort sai thứ tự dòng | Bounding box lệch hoặc ảnh nghiêng | Cải thiện line grouping và deskew |

---

## 21. Visualization

Đồ án cần có visualization để phục vụ báo cáo và demo.

### 21.1. Các visualization cần có

```text
1. Ảnh gốc.
2. Ảnh với OCR bounding boxes.
3. Ảnh highlight field dự đoán:
   - SELLER
   - SELLER_ADDRESS
   - TIMESTAMP
   - TOTAL_COST
4. Confusion matrix cho line classifier nếu có.
5. Bar chart F1 theo từng field.
6. Một số ví dụ prediction đúng/sai.
```

### 21.2. Vẽ bounding box

```python
import cv2

def draw_boxes(image, lines):
    image = image.copy()

    for line in lines:
        x1, y1, x2, y2 = map(int, line["bbox"])
        text = line["text"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            text[:30],
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return image
```

---

## 22. Demo

### 22.1. Công nghệ

Giao diện demo sử dụng:

```text
Streamlit
```

hoặc:

```text
Gradio
```

Khuyến nghị dùng Streamlit vì dễ tạo UI nhanh.

---

### 22.2. Chức năng demo

Demo cần có:

```text
- Upload ảnh hóa đơn.
- Hiển thị ảnh gốc.
- Chạy OCR.
- Hiển thị OCR bounding boxes.
- Hiển thị 4 field trích xuất.
- Hiển thị output format MC-OCR.
- Cho phép download JSON hoặc TXT.
- Xử lý lỗi cơ bản khi ảnh không hợp lệ.
```

---

### 22.3. Pseudocode Streamlit

```python
import streamlit as st
from PIL import Image
import tempfile

st.title("Vietnamese Receipt KIE Demo - MC-OCR2021")

uploaded_file = st.file_uploader(
    "Upload receipt image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input receipt", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    lines = run_ocr(image_path)

    width, height = image.size
    fields = extract_fields(lines, height)

    st.subheader("Extracted fields")
    st.json(fields)

    st.subheader("MC-OCR output format")
    st.code(format_mcocr_output(fields))
```

---

## 23. Edge cases cần xử lý

| Edge case | Cách xử lý |
|---|---|
| File không phải ảnh | Báo lỗi định dạng |
| Ảnh quá lớn | Resize |
| Không phát hiện text | Trả về field rỗng và cảnh báo |
| OCR confidence thấp | Hiển thị cảnh báo |
| Thiếu field | Để field rỗng |
| Có nhiều dòng tiền | Dùng keyword + vị trí để chọn dòng tổng tiền |
| Có nhiều ngày/giờ | Ưu tiên dòng chứa keyword ngày/giờ hoặc nằm gần thông tin giao dịch |
| Địa chỉ bị tách dòng | Merge dòng gần nhau nếu có thời gian |

---

## 24. Cấu trúc repository

```text
mcocr2021-kie/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── annotations/
│   └── splits/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_rules.ipynb
│   ├── 03_error_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── ocr.py
│   ├── line_processing.py
│   ├── extractor.py
│   ├── postprocess.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── demo/
│       └── app.py
│
├── experiments/
│   ├── exp01_simple_rule.yaml
│   ├── exp02_preprocess_scoring.yaml
│   ├── exp03_lightweight_classifier.yaml
│   └── exp04_donut.yaml
│
├── outputs/
│   ├── predictions/
│   ├── visualizations/
│   └── metrics/
│
├── reports/
│   ├── figures/
│   └── final_report.pdf
│
├── README.md
├── PROJECT.md
├── requirements.txt
└── run_demo.sh
```

---

## 25. Cài đặt

### 25.1. Tạo môi trường

```bash
python -m venv .venv
source .venv/bin/activate
```

Trên Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 25.2. Cài thư viện

```bash
pip install -r requirements.txt
```

Gợi ý `requirements.txt`:

```text
opencv-python
pillow
numpy
pandas
scikit-learn
matplotlib
streamlit
paddleocr
paddlepaddle
python-Levenshtein
tqdm
```

---

## 26. Cách chạy

### 26.1. Chạy OCR cho một ảnh

```bash
python src/ocr.py --image data/raw/sample.jpg --output outputs/ocr_sample.json
```

### 26.2. Chạy extraction cho một ảnh

```bash
python src/extractor.py --image data/raw/sample.jpg --output outputs/prediction_sample.json
```

### 26.3. Chạy evaluation

```bash
python src/evaluate.py \
  --pred outputs/predictions.json \
  --gt data/annotations/ground_truth.json
```

### 26.4. Chạy demo

```bash
streamlit run src/demo/app.py
```

---

## 27. Kế hoạch triển khai

### Ngày 1: Dataset + OCR baseline

```text
- Đọc format annotation MC-OCR2021.
- Chạy thử PaddleOCR trên 20 ảnh.
- Chuẩn hóa OCR output thành text + bbox + confidence.
- Viết hàm sort_lines.
```

### Ngày 2: Rule extractor

```text
- Viết extractor cho TOTAL_COST.
- Viết extractor cho TIMESTAMP.
- Viết extractor cho SELLER.
- Viết extractor cho SELLER_ADDRESS.
- Format output theo chuẩn MC-OCR.
```

### Ngày 3: Evaluation

```text
- Viết normalize function.
- Viết CER.
- Viết Exact Match.
- Viết Precision / Recall / F1.
- Chạy baseline trên validation set.
```

### Ngày 4: Cải tiến pipeline

```text
- Thêm preprocessing.
- Thêm rule scoring.
- So sánh trước/sau preprocessing.
- Ghi lại kết quả thực nghiệm.
```

### Ngày 5: Visualization + error analysis

```text
- Vẽ OCR boxes.
- Highlight các field dự đoán.
- Phân tích lỗi theo từng field.
- Lưu các ví dụ prediction đúng/sai.
```

### Ngày 6: Demo

```text
- Làm Streamlit demo.
- Test demo với nhiều ảnh.
- Thêm xử lý edge case.
```

### Ngày 7: Báo cáo + slide

```text
- Hoàn thiện README.
- Hoàn thiện PROJECT.md.
- Viết báo cáo.
- Làm slide.
- Chuẩn bị demo live.
```

---

## 28. Hạn chế

Pipeline hiện tại có một số hạn chế:

```text
- Phụ thuộc nhiều vào chất lượng OCR.
- Rule-based extractor có thể sai nếu layout hóa đơn khác thường.
- SELLER và SELLER_ADDRESS khó phân biệt nếu OCR thiếu dòng đầu.
- TOTAL_COST có thể nhầm với subtotal hoặc VAT.
- TIMESTAMP có thể nhầm với số giao dịch nếu OCR không rõ.
- Chưa xử lý tốt địa chỉ bị tách nhiều dòng.
- Chưa sử dụng mô hình học sâu cho KIE trong pipeline chính.
```

---

## 29. Hướng phát triển

Các hướng mở rộng:

```text
1. Thay rule-based extractor bằng line classifier.
2. Dùng TF-IDF + layout features + Logistic Regression.
3. Dùng PhoBERT kết hợp đặc trưng layout.
4. Dùng LayoutXLM cho token/line classification.
5. Dùng VietOCR để cải thiện nhận dạng tiếng Việt.
6. Tự annotate thêm table/signature nếu muốn mở rộng sang Document AI tổng quát.
7. Fine-tune OCR engine trên dữ liệu hóa đơn tiếng Việt.
8. Thêm module deskew/dewarp cho ảnh nghiêng hoặc cong.
```

---

## 30. Kết luận

Đồ án xây dựng một pipeline OCR-based đơn giản, thực dụng và dễ triển khai cho bài toán trích xuất thông tin hóa đơn tiếng Việt trên MC-OCR2021. Hệ thống sử dụng PaddleOCR để nhận dạng text, sắp xếp các dòng theo layout, áp dụng rule scoring để chọn 4 field quan trọng và hậu xử lý để chuẩn hóa kết quả.

So với hướng end-to-end như Donut, pipeline này có ưu điểm là:

```text
- Dễ triển khai hơn.
- Ít phụ thuộc vào dữ liệu huấn luyện lớn.
- Dễ debug và phân tích lỗi.
- Dễ demo live.
- Phù hợp với annotation dạng text line của MC-OCR2021.
- Có thể mở rộng dần sang các mô hình mạnh hơn như PhoBERT hoặc LayoutXLM.
```

Đây là hướng phù hợp để hoàn thành đồ án đúng hạn, đáp ứng các tiêu chí về baseline, thực nghiệm, phân tích lỗi, demo và báo cáo khoa học.
