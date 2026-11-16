# Báo cáo: Phát hiện ảnh bất thường bằng Autoencoder

## 1. Mục tiêu

Xây dựng bộ phát hiện bất thường one-class cho Fashion-MNIST. Autoencoder
được huấn luyện chỉ trên lớp Sneaker; ảnh có reconstruction error lớn hơn
ngưỡng được gắn nhãn bất thường. Vì vậy, “bất thường” ở đây có nghĩa là
không thuộc phân bố ảnh Sneaker mà mô hình đã học, không phải kết luận về
độ an toàn hay tính bất thường của vật thể trong đời thực.

## 2. Dữ liệu và phương pháp

- Dữ liệu: Fashion-MNIST, ảnh grayscale 28×28, pixel chuẩn hóa về `[0, 1]`.
- Lớp bình thường: Sneaker (nhãn 7); chín lớp còn lại là bất thường.
- Seed: 42.
- Tập huấn luyện: 4.800 ảnh Sneaker từ tập train gốc.
- Validation-normal: 1.200 ảnh Sneaker còn lại.
- Validation-mixed: 1.200 ảnh bình thường và 1.200 ảnh bất thường được
  lấy từ tập train gốc; tập này dùng để so sánh cấu hình.
- Test gốc: 10.000 ảnh, gồm 1.000 bình thường và 9.000 bất thường; chỉ dùng
  cho đánh giá cuối sau khi đã chọn mô hình và ngưỡng.

Hai Autoencoder fully-connected có cùng kiến trúc, khác latent dimension
(16 và 64). Huấn luyện tối đa 30 epoch bằng MSE, Adam (`lr=1e-3`), batch
size 128 và early stopping với patience 5. Ngưỡng P90/P95/P99 được tính từ
reconstruction error của validation-normal. Quy tắc dự đoán là bất thường
khi error lớn hơn ngưỡng.

## 3. So sánh trên validation

| Latent dimension | Epoch tốt nhất | ROC-AUC | Precision (P95) | Recall (P95) | F1 (P95) |
|---:|---:|---:|---:|---:|---:|
| 16 | 30 | 0.9831 | 0.9488 | 0.9275 | **0.9381** |
| 64 | 30 | **0.9836** | 0.9487 | 0.9242 | 0.9363 |

Chọn latent 16 và P95 vì F1 trên validation-mixed cao hơn trong hai cấu
hình P95; ROC-AUC của hai cấu hình gần tương đương. Ngưỡng đã cố định là
`0.0212676078`. Không dùng test để chọn mô hình hoặc ngưỡng.

## 4. Đánh giá cuối trên test

Đánh giá một lần với latent 16 và ngưỡng P95 đã cố định:

- ROC-AUC: **0.9828**
- Precision: **0.9949**
- Recall: **0.9126**
- F1-score: **0.9520**
- Confusion matrix (positive = bất thường):

|  | Dự đoán bình thường | Dự đoán bất thường |
|---|---:|---:|
| Thực tế bình thường | TN = 958 | FP = 42 |
| Thực tế bất thường | FN = 787 | TP = 8.213 |

Trên test, mô hình phát hiện đúng 8.213/9.000 ảnh bất thường và bỏ sót
787 ảnh. Có 42/1.000 ảnh bình thường bị báo nhầm. Precision cao cần được
diễn giải cùng với tỷ lệ lớp test (90% ảnh bất thường), chứ không nên xem
xét riêng lẻ.

## 5. Hạn chế

- Reconstruction error không đảm bảo tương đương với hiểu biết ngữ nghĩa
  về vật thể; lớp khác có hình dạng tương tự Sneaker có thể được tái tạo
  tốt và bị bỏ sót.
- Validation-mixed được lấy cân bằng để so sánh; phân bố lớp này khác test
  gốc nên precision/F1 không thể so sánh trực tiếp nếu không xét tỷ lệ
  lớp.
- Ảnh tải lên từ bên ngoài có thể khác nền, tỷ lệ, màu sắc và bố cục của
  Fashion-MNIST. Demo chỉ là minh họa cho preprocessing và reconstruction
  error, không phải công cụ phân loại đồ vật tổng quát.
- Chỉ khảo sát một lớp bình thường, một seed và hai latent dimension.

## 6. Hướng phát triển

- Thử các lớp bình thường khác và nhiều seed để kiểm tra độ ổn định.
- Khảo sát latent dimension, regularization và các kiến trúc convolutional.
- Báo cáo thêm PR-AUC và đánh giá ở nhiều tỷ lệ lớp/threshold theo yêu cầu
  vận hành.
- Nếu dùng ngoài Fashion-MNIST, thu thập dữ liệu đại diện và hiệu chỉnh lại
  preprocessing, mô hình và threshold trên validation phù hợp.
- Cải thiện demo với hướng dẫn căn chỉnh ảnh đầu vào và thông tin cảnh báo
  về giới hạn miền dữ liệu.

## 7. Hình kết quả

- [Training/validation loss](outputs/figures/training_loss.png)
- [ROC trên validation](outputs/figures/validation_roc.png)
- [Phân bố lỗi trên validation](outputs/figures/validation_error_distribution.png)
- [So sánh mô hình trên validation](outputs/figures/validation_model_comparison.png)
- [Phân bố lỗi trên test](outputs/figures/test_error_distribution.png)
- [Confusion matrix trên test](outputs/figures/test_confusion_matrix.png)
- [Ví dụ TN/FP/FN/TP và ảnh tái tạo](outputs/figures/test_reconstruction_examples.png)

## 8. Tái lập kết quả

Từ thư mục dự án, cài dependencies bằng `uv sync`, sau đó:

```powershell
uv run python src\data_loader.py
uv run python src\train.py --latent-dim 16
uv run python src\train.py --latent-dim 64
uv run python src\evaluate.py --latent-dim 16
uv run python src\evaluate.py --latent-dim 64
uv run python src\visualize.py
uv run streamlit run app\streamlit_app.py
```

Kết quả test cuối đã được lưu ở `outputs/metrics/final_test.json`. Script
`src/evaluate_test.py` không ghi đè kết quả đó và chủ động từ chối chạy lại
để tránh vô tình sử dụng test set nhiều lần.
