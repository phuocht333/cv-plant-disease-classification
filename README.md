# KẾ HOẠCH NGHIÊN CỨU: CHẨN ĐOÁN BỆNH CÂY TRỒNG TRONG NHÀ KÍNH BẰNG CNN RÚT GỌN (SOTA 2024-2026)

**Nhóm thực hiện:** 03 Thành viên
**Môi trường thực thi:** Kaggle / Google Colab (Free Tier - T4 GPU)
**Thời gian:** 04 Tuần (Mở rộng để phân tích chuyên sâu)

## I. PHÂN TÍCH KỸ THUẬT VÀ CƠ SỞ LÝ THUYẾT (NÂNG CAO)

Phần này cung cấp các định nghĩa toán học và lý do kỹ thuật để đưa vào chương "Cơ sở lý thuyết" của báo cáo Master.

### 1. Chiến lược Tiền xử lý & Augmentation

* **CLAHE (Contrast Limited Adaptive Histogram Equalization):**
    * **Định nghĩa:** Khác với cân bằng Histogram thông thường (toàn cục), CLAHE chia ảnh thành các ô nhỏ (tiles) và cân bằng cục bộ, đồng thời giới hạn độ tương phản để tránh khuếch đại nhiễu.
    * **Lý do sử dụng:** Môi trường nhà kính có độ ẩm cao tạo ra lớp sương mờ (fog) và ánh sáng đèn LED tạo ra các vùng chói cục bộ. CLAHE giúp "nhìn xuyên" lớp sương này để làm rõ các vết bệnh.

### 2. Chiến lược Tối ưu hóa (Optimization Strategy)

* **Hàm mất mát: Label Smoothing Cross-Entropy**
    * **Vấn đề:** Khi sử dụng Cross-Entropy truyền thống ($y_{target}=1$), mô hình thường trở nên quá tự tin (over-confident), dẫn đến Overfitting trên tập dữ liệu nhiễu như ảnh thực tế.
    * **Giải pháp:** Thay vì ép mô hình dự đoán xác suất 1.0 cho nhãn đúng, ta làm "mềm" nhãn mục tiêu:
    
    $$y_{new} = (1 - \alpha) \times y_{target} + \frac{\alpha}{K}$$
    
    *(Trong đó: $\alpha$ là hệ số làm mềm, thường là 0.1; $K$ là số lượng lớp).*

    * **Công thức Loss:**
    
    $$\mathcal{L}_{LS} = - \sum_{i=1}^{K} y_{new}^{(i)} \log(p_i)$$

    * **Tác dụng:** Giúp các cụm đặc trưng (feature clusters) trong không gian vector chặt chẽ hơn, tăng khả năng tổng quát hóa (Generalization) khi gặp dữ liệu lạ.

* **Lịch trình huấn luyện: Cosine Annealing Warm Restarts**
    * **Định nghĩa:** Learning Rate (LR) giảm dần theo hàm Cosine từ giá trị cực đại xuống cực tiểu, sau đó đột ngột "khởi động lại" (warm restart) về giá trị lớn.
    * **Lý do sử dụng:** Các mạng CNN hiện đại (như ConvNeXt) có hàm mất mát rất phức tạp với nhiều cực tiểu địa phương (local minima). Việc giảm LR giúp hội tụ, nhưng việc "Restart" giúp mô hình "nhảy" ra khỏi cực tiểu tồi để tìm kiếm cực tiểu toàn cục tốt hơn (Global Minima).

## II. BẢNG CẤU HÌNH HYPERPARAMETER (KHUYẾN NGHỊ SOTA)

Dưới đây là cấu hình tham khảo để đạt kết quả tốt nhất trên Kaggle T4 GPU (16GB VRAM):

| Tham số (Hyperparameter) | Giá trị cấu hình | Lý giải kỹ thuật |
| :--- | :--- | :--- |
| **Image Size** | $224 \times 224$ | Chuẩn công nghiệp cho MobileNet/ConvNeXt, cân bằng giữa thông tin chi tiết và tốc độ. |
| **Batch Size** | 32 (hoặc 64 nếu dùng FP16) | Batch size lớn hơn giúp ước lượng Gradient chính xác hơn, ổn định quá trình Batch Normalization. |
| **Optimizer** | **AdamW** | AdamW tách biệt Weight Decay khỏi Gradient update, giúp hội tụ tốt hơn Adam thường đối với các kiến trúc lai (ConvNeXt). |
| **Initial Learning Rate** | $1e-3$ (0.001) | Mức khởi đầu an toàn cho Transfer Learning. |
| **Weight Decay** | $0.05$ | Giúp kiểm soát độ phức tạp của trọng số, tránh overfitting mạnh hơn L2 Regularization thường. |
| **Label Smoothing** | $\alpha = 0.1$ | Ngăn chặn mô hình quá tự tin vào dữ liệu Training nhiễu. |
| **Epochs** | 50 | Đủ để mô hình hội tụ trên tập dữ liệu ~10k-50k ảnh. |
| **Precision** | **Mixed Precision (FP16)** | Giảm 50% bộ nhớ VRAM, tăng tốc độ train 2x trên T4 GPU. |

## III. CÁC CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS)

Để báo cáo đạt chuẩn Master, không chỉ cần Accuracy mà cần bộ chỉ số đa chiều.

| Metric | Định nghĩa | Vai trò trong Đồ án này | Tiêu chí đạt (Target) |
| :--- | :--- | :--- | :--- |
| **Accuracy (Top-1)** | Tỷ lệ dự đoán đúng trên tổng số ảnh. | Đánh giá tổng quan hiệu năng phân loại. | $> 92\%$ |
| **F1-Score (Macro)** | Trung bình điều hòa của Precision và Recall, tính trung bình trên tất cả các lớp. | **Quan trọng nhất:** Do dữ liệu bệnh cây thường mất cân bằng, F1-Score phản ánh đúng thực tế hơn Accuracy. | $> 88\%$ |
| **Inference Latency** | Thời gian xử lý 1 ảnh (đơn vị: ms) với Batch size = 1. | Đánh giá khả năng đáp ứng thời gian thực (Real-time) trên thiết bị IoT/Edge. | $< 30ms$ (trên T4 GPU) |
| **Throughput** | Số lượng ảnh xử lý được trong 1 giây (img/s). | Đánh giá khả năng xử lý luồng dữ liệu lớn (Batch processing) của hệ thống. | $> 100$ img/s |
| **Model Size** | Kích thước file trọng số (.pth/.onnx) tính bằng MB. | Đánh giá khả năng triển khai lên các thiết bị nhớ thấp (như Raspberry Pi). | $< 10$ MB |
| **FLOPs** | Floating Point Operations (Tỷ phép tính dấu phẩy động). | Đánh giá độ phức tạp thuật toán và mức tiêu thụ năng lượng lý thuyết. | $< 1.0$ GFLOPs |

## IV. KẾT QUẢ DỰ KIẾN (EXPECTED OUTPUTS)

Báo cáo cuối cùng cần trình bày được các kết quả đầu ra sau đây để chứng minh tính hiệu quả:

### 1. Bảng so sánh định lượng (Quantitative Comparison Table)

| Model Architecture | Params (M) | FLOPs (G) | Accuracy (%) | Macro F1 (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV4** | ... | ... | ... | ... | ... |
| **ConvNeXt-V2** | ... | ... | ... | ... | ... |
| **GhostNetV3** | ... | ... | ... | ... | ... |

### 2. Các biểu đồ trực quan (Visualizations)
* **Biểu đồ Learning Curve:** Đường Loss (Train/Val) và Accuracy (Train/Val) qua 50 epochs.
* **Confusion Matrix (Ma trận nhầm lẫn):** Biểu đồ nhiệt thể hiện sự nhầm lẫn giữa các lớp bệnh.
* **Biểu đồ Trade-off (Accuracy vs. Efficiency):** Biểu đồ phân tán (Scatter plot) với trục tung là Accuracy và trục hoành là Latency.

## V. PHÂN CÔNG CÔNG VIỆC CHI TIẾT (LỘ TRÌNH 04 TUẦN)

### THÀNH VIÊN 01: TRƯỞNG NHÓM & KIẾN TRÚC SƯ (MobileNetV4)
**Trọng tâm:** Quản lý quy trình Pipeline chuẩn và Mô hình hiệu năng cao.

* **Tuần 1: Khởi tạo & Pipeline**
    * Thiết lập GitHub Repo và cấu trúc thư mục chuẩn.
    * Xây dựng hàm `Train_Epoch` và `Validate_Epoch` dùng chung cho cả nhóm.
* **Tuần 2: Baseline Training**
    * Huấn luyện MobileNetV4 với cấu hình mặc định (Baseline).
    * Ghi lại Accuracy/Loss ban đầu.
* **Tuần 3: Tối ưu hóa (Ablation Study)**
    * Thử nghiệm bật/tắt module **Universal Inverted Bottleneck (UIB)**.
    * Tinh chỉnh Weight Decay.
* **Tuần 4: Tổng hợp & Viết báo cáo**
    * Viết chương "Kiến trúc hệ thống" và "Cơ sở lý thuyết".
    * Tổng hợp biểu đồ so sánh cả 3 mô hình.

### THÀNH VIÊN 02: KỸ SƯ DỮ LIỆU & CONVNEXT (ConvNeXt-V2)
**Trọng tâm:** Chất lượng dữ liệu đầu vào và Mô hình hiện đại.

* **Tuần 1: Xử lý dữ liệu (Data Engineering)**
    * Tải dataset, lọc bỏ ảnh lỗi.
    * **Quan trọng:** Cài đặt thuật toán CLAHE và viết script visualize sự khác biệt.
* **Tuần 2: Baseline Training**
    * Huấn luyện ConvNeXt-V2 Nano.
    * Theo dõi hiện tượng Overfitting.
* **Tuần 3: Data Augmentation nâng cao**
    * Áp dụng **RandAugment** hoặc **MixUp**.
    * So sánh kết quả khi có và không có MixUp.
* **Tuần 4: Phân tích lỗi (Error Analysis)**
    * Xuất ra Confusion Matrix.
    * Phân tích các cặp lớp dễ nhầm lẫn.

### THÀNH VIÊN 03: TỐI ƯU HÓA & GHOSTNET (GhostNetV3)
**Trọng tâm:** Hiệu năng thực tế và Triển khai.

* **Tuần 1: Công cụ đo lường (Benchmarking Tools)**
    * Viết script đo số lượng tham số (Params) và khối lượng tính toán (FLOPs).
    * Viết script đo FPS (Frames Per Second) trên CPU và GPU.
* **Tuần 2: Baseline Training**
    * Huấn luyện GhostNetV3.
    * Lưu ý: Theo dõi kỹ biểu đồ Loss do GhostNet thường hội tụ chậm.
* **Tuần 3: Thử nghiệm suy luận (Inference Test)**
    * Chạy 3 mô hình đã train trên tập Test độc lập.
    * Đo thời gian suy luận trung bình (Latency) với Batch size = 1.
* **Tuần 4: Viết báo cáo phần Kết quả**
    * Vẽ biểu đồ Trade-off: Trục tung là Accuracy, trục hoành là Latency.
    * Viết nhận xét: Mô hình nào tối ưu nhất?

## VI. TÀI LIỆU THAM KHẢO & SOURCE CODE
1.  **MobileNetV4:** `timm.create_model('mobilenetv4_conv_small', ...)`
2.  **ConvNeXt-V2:** `timm.create_model('convnextv2_nano', ...)`
3.  **GhostNetV3:** Clone từ repo chính thức hoặc dùng bản implement trong `timm` (nếu có).
4.  **Paper tham chiếu:**
    * *Müller et al. (2019).* "When does label smoothing help?" (NeurIPS).
    * *Loshchilov & Hutter (2019).* "Decoupled Weight Decay Regularization" (AdamW Paper).