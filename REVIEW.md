# Code Review hiện trạng

Review này dựa trên trạng thái codebase hiện tại và nội dung trong `PROJECT.md` để đối chiếu mức độ khớp giữa mục tiêu dự án và implementation đang có. Review không dùng kết quả train/inference thực tế để kết luận chất lượng mô hình, và có cập nhật so với vòng review trước vì repo đã bổ sung `docs/architecture.png` và `scripts/convert_sroie.py`.

## Findings

1. Warm-up CORD hiện chưa có thước đo hợp lệ để chọn checkpoint trong [scripts/train_donut.py](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/scripts/train_donut.py:72), [scripts/train_donut.py](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/scripts/train_donut.py:137), [scripts/train_donut.py](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/scripts/train_donut.py:256), [scripts/utils.py](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/scripts/utils.py:23), [scripts/utils.py](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/scripts/utils.py:98).  
   `validate()` luôn tính metric thông qua `compute_metrics()` và `parse_donut_output()`, nhưng hai hàm này chỉ hiểu đúng 4 field cố định `store_name`, `date`, `total`, `address`. Trong khi đó, warm-up trên CORD dùng schema khác và `DonutHFDataset` không hề ánh xạ schema đó về 4 field này. Tác động là `best_f1` cho warm-up có thể luôn bằng 0 hoặc không phản ánh gì hữu ích, khiến checkpoint `cord_warmup` có nguy cơ không bao giờ được lưu dù [configs/donut_mcocr.yaml](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/configs/donut_mcocr.yaml:1) đang phụ thuộc trực tiếp vào nó làm pretrained model cho E2.

2. `DonutHFDataset` cho CORD đang linearize `gt_parse` theo cách dễ tạo target sai format ở [scripts/train_donut.py](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/scripts/train_donut.py:89).  
   Code hiện dựng target bằng `"<s_{k}>{v}</s_{k}>"` cho mọi cặp key-value trong `gt_parse`. Nếu value của CORD là dict, list hoặc cấu trúc lồng nhau, `v` sẽ bị ép sang string Python mặc định thay vì một serialization Donut ổn định và có chủ đích. Tác động là dữ liệu huấn luyện warm-up có thể bị méo format ngay từ đầu, làm giảm giá trị của bước CORD và kéo theo rủi ro chất lượng cho checkpoint dùng làm đầu vào của fine-tune MC-OCR.

3. `scripts/convert_sroie.py` nuốt lỗi parse bằng `except: pass`, tạo rủi ro mất dữ liệu im lặng ở [scripts/convert_sroie.py](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/scripts/convert_sroie.py:70).  
   Mọi file `.txt` của SROIE parse lỗi đều bị bỏ qua hoàn toàn mà không log file nào lỗi, lỗi gì, hay có bao nhiêu record bị loại. Tác động là số lượng mẫu đầu vào cho E3 có thể giảm đi một cách âm thầm trước khi split train/val/test, làm sai phân phối dữ liệu và khiến việc debug kết quả cross-dataset trở nên khó hơn nhiều.

4. `scripts/convert_mcocr.py` vẫn chọn CSV đầu tiên trong thư mục input và map cột bằng heuristic ở [scripts/convert_mcocr.py](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/scripts/convert_mcocr.py:37), [scripts/convert_mcocr.py](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/scripts/convert_mcocr.py:113).  
   Script hiện đã thêm bước in sample record và confirm thủ công, nhưng logic nguồn vẫn là “lấy file `.csv` đầu tiên rồi đoán mapping cột”. Nếu raw dump có nhiều CSV hoặc header hơi khác kỳ vọng, script vẫn có thể tạo supervision sai mà người dùng chỉ phát hiện nếu đọc kỹ sample và biết trước format đúng. Tác động là rủi ro silent data corruption đã giảm nhưng chưa biến mất, nhất là trong môi trường chạy batch hoặc khi dùng `--force`.

5. Repo vẫn còn lệch `PROJECT.md` ở các deliverable quan trọng: SynthDoG-VI augmentation, Grad-CAM cho E3, và đo inference time trong bảng so sánh.  
   `PROJECT.md` mô tả đây là các phần của pipeline hoặc kết quả thí nghiệm, nhưng codebase hiện chưa có script, notebook workflow rõ ràng, hay logic evaluation tương ứng để hiện thực chúng end-to-end. Tác động là repo đã đủ hơn trước cho E1/E2/E3 cơ bản, nhưng vẫn chưa đạt mức bám sát đầy đủ spec nghiên cứu/đồ án đã viết ra.

6. README chưa tài liệu hóa luồng chuẩn bị SROIE trước E3 dù script convert đã tồn tại ở [README.md](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/README.md:38), [README.md](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/README.md:71).  
   Tài liệu hiện chỉ nêu bước convert cho MC-OCR rồi nhảy thẳng sang lệnh chạy E3. Vì [scripts/convert_sroie.py](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/scripts/convert_sroie.py:53) là bước tiền xử lý bắt buộc để tạo `data/sroie/donut_format/...`, E3 chưa được mô tả end-to-end trong tài liệu chính. Tác động là người đọc repo có thể không tái lập được thí nghiệm E3 chỉ từ README.

7. `compute_metrics()` vẫn dùng exact match tuyệt đối ở [scripts/utils.py](/D:/Users/Documents/HUCE/Thi_Giac_May_Tinh/document-ai-vn/scripts/utils.py:33).  
   Metric hiện chỉ normalize lowercase và khoảng trắng, sau đó yêu cầu chuỗi phải khớp hoàn toàn mới tính đúng. Với bài toán KIE, các field như `date`, `total`, `address` thường có nhiều cách biểu diễn tương đương nhưng khác dấu phân cách, định dạng số hoặc viết tắt. Tác động là F1/P/R hiện tại mang tính bảo thủ và có thể thấp hơn năng lực hữu ích thực tế của hệ thống; đây không còn là blocker pipeline chính, nhưng vẫn ảnh hưởng trực tiếp đến cách đối chiếu với ngưỡng trong `PROJECT.md`.

## Nhận định tổng quan

Codebase hiện đã tiến thêm một bước rõ rệt so với trạng thái trước: các lỗi lớn về target format cho Donut local datasets, task prompt khi `generate()`, asset tài liệu bị thiếu, và luồng convert SROIE cơ bản đều đã được xử lý. Repo cũng đã gần hơn với một pipeline có thể chạy từ E1 đến E3 ở mức khung triển khai.

Tuy vậy, blocker chính hiện đã chuyển sang phần warm-up CORD, chất lượng data preparation, và khoảng cách giữa spec trong `PROJECT.md` với những gì repo thực sự hiện thực. Cụ thể, bước warm-up trên CORD vẫn chưa có cơ chế metric/checkpoint đáng tin, còn E3 và bảng so sánh kết quả vẫn thiếu vài deliverable nghiên cứu quan trọng. Nói ngắn gọn, Donut pipeline không còn “hỏng vì lý do cũ”, nhưng cũng chưa đủ chặt để xem là đã đáp ứng đầy đủ mục tiêu thí nghiệm của dự án.
