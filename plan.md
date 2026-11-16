# PLAN.md --- Phát hiện ảnh bất thường bằng Autoencoder

**Môn học:** AI406DE01 -- Trí tuệ nhân tạo nâng cao\
**Đề tài:** Phát hiện bất thường trên ảnh bằng Autoencoder dựa trên
Reconstruction Error

------------------------------------------------------------------------

## 1. Tổng quan đề tài

### 1.1. Mục tiêu

Xây dựng hệ thống phát hiện ảnh bất thường theo hướng **one-class
anomaly detection**. Autoencoder được huấn luyện chủ yếu trên ảnh thuộc
một lớp bình thường để học cách mã hóa và tái tạo đặc trưng của lớp đó.
Khi nhận ảnh thuộc lớp khác, mô hình thường tái tạo kém hơn và tạo ra
reconstruction error cao hơn.

Đầu ra của hệ thống gồm:

-   Ảnh đầu vào và ảnh tái tạo.
-   Reconstruction error (điểm bất thường).
-   Ngưỡng phát hiện được xác định từ validation set.
-   Nhãn dự đoán: **Bình thường** hoặc **Bất thường**.

### 1.2. Phạm vi và giả định

-   Sử dụng **Fashion-MNIST** vì có 10 lớp quần áo/phụ kiện và ảnh
    grayscale kích thước 28 × 28.
-   Chọn một lớp làm dữ liệu bình thường; chín lớp còn lại là bất
    thường.
-   Đề xuất chọn lớp **Sneaker** làm lớp bình thường. Đây là lựa chọn
    ban đầu để thực hiện; có thể đổi sang lớp khác nếu kết quả thực
    nghiệm cho thấy phù hợp hơn.
-   Chỉ sử dụng ảnh thuộc lớp bình thường để cập nhật trọng số trong quá
    trình huấn luyện.
-   Dữ liệu bất thường được giữ lại để kiểm tra khả năng phân biệt và
    đánh giá.
-   Huấn luyện trên CPU, ưu tiên kiến trúc fully-connected nhỏ để thời
    gian chạy phù hợp.

------------------------------------------------------------------------

## 2. Câu hỏi thực nghiệm

1.  Autoencoder có học được đặc trưng của lớp bình thường và tái tạo ảnh
    cùng lớp tốt hơn ảnh bất thường hay không?
2.  Kích thước latent space ảnh hưởng thế nào đến reconstruction error
    và chất lượng phát hiện?
3.  Ngưỡng anomaly threshold được lựa chọn theo percentile của lỗi trên
    validation-normal có tạo ra sự cân bằng phù hợp giữa false positive
    và false negative không?
4.  Hệ thống hoạt động như thế nào khi người dùng tải lên một ảnh mới
    thông qua giao diện demo?

------------------------------------------------------------------------

## 3. Dữ liệu và cách chia tập

### 3.1. Dataset

**Fashion-MNIST**

#### Nguồn tải dataset

Sử dụng Fashion-MNIST do Zalando Research công bố:

-   Trang GitHub chính thức: https://github.com/zalandoresearch/fashion-mnist
-   Thư mục chứa các file dữ liệu IDX: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
-   Dataset cũng được tích hợp sẵn trong `torchvision.datasets.FashionMNIST`, vì vậy nhóm **không cần tải thủ công từng file `.gz`**.

Dataset gồm 4 phần dữ liệu gốc:

-   `train-images-idx3-ubyte.gz`: ảnh huấn luyện.
-   `train-labels-idx1-ubyte.gz`: nhãn huấn luyện.
-   `t10k-images-idx3-ubyte.gz`: ảnh kiểm thử.
-   `t10k-labels-idx1-ubyte.gz`: nhãn kiểm thử.

Khuyến nghị dùng `torchvision` để tự tải, giải nén và đọc dữ liệu. Khi chạy lần đầu cần có kết nối Internet; các lần chạy sau có thể dùng dữ liệu đã lưu trong thư mục `data/fashion_mnist/`.

#### Code tải Fashion-MNIST bằng PyTorch

Tạo file `src/data_loader.py` hoặc chạy đoạn mã sau để kiểm tra việc tải dữ liệu:

```python
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="./data/fashion_mnist",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.FashionMNIST(
    root="./data/fashion_mnist",
    train=False,
    download=True,
    transform=transform,
)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

print("Train:", len(train_dataset))
print("Test:", len(test_dataset))
print("Image shape:", train_dataset[0][0].shape)
print("Label:", train_dataset[0][1], class_names[train_dataset[0][1]])
```

Kết quả mong đợi:

```text
Train: 60000
Test: 10000
Image shape: torch.Size([1, 28, 28])
```

`transforms.ToTensor()` chuyển ảnh sang tensor kiểu số thực và đưa giá trị pixel từ `[0, 255]` về `[0, 1]`, phù hợp với đầu ra `Sigmoid` của Autoencoder. Tham số `download=True` cho phép tự tải dữ liệu nếu máy chưa có dataset.



-   60.000 ảnh train và 10.000 ảnh test theo cấu trúc dữ liệu gốc.
-   10 nhãn: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt,
    Sneaker, Bag, Ankle boot.
-   Mỗi ảnh có kích thước 28 × 28, một kênh màu.
-   Pixel được chuẩn hóa về khoảng `[0, 1]`.

### 3.2. Quy tắc tạo bài toán one-class

Với lớp bình thường là `Sneaker`:

  --------------------------------------------------------------------------
  Tập                 Ảnh bình thường   Ảnh bất thường    Mục đích
  ------------------- ----------------- ----------------- ------------------
  Train               Chỉ Sneaker       Không dùng        Huấn luyện
                                                          Autoencoder

  Validation-normal   Chỉ Sneaker       Không dùng        Ước lượng phân bố
                                                          reconstruction
                                                          error bình thường
                                                          và chọn ngưỡng

  Validation-mixed    Sneaker           Chín lớp còn lại  Kiểm tra ngưỡng,
                                                          phân tích
                                                          Precision/Recall
                                                          và chọn cấu hình

  Test                Sneaker           Chín lớp còn lại  Đánh giá cuối
                                                          cùng, không dùng
                                                          để chọn mô
                                                          hình/ngưỡng
  --------------------------------------------------------------------------

**Lưu ý chống rò rỉ dữ liệu:**

-   Không đưa ảnh validation/test vào quá trình cập nhật trọng số.
-   Không chọn threshold dựa trên test set.
-   Tách validation từ tập train gốc trước khi huấn luyện.
-   Dùng test set đúng một lần cho đánh giá cuối cùng sau khi đã cố định
    mô hình và threshold.
-   Khi báo cáo, ghi rõ số lượng ảnh bình thường/bất thường trong từng
    tập.

### 3.3. Tỷ lệ chia đề xuất

Từ tập train Fashion-MNIST:

-   80% ảnh của lớp bình thường: training.
-   20% ảnh của lớp bình thường: validation-normal.
-   Lấy một phần ảnh thuộc các lớp còn lại làm validation-mixed để kiểm
    tra và chọn cấu hình.
-   Tập test gốc được dùng làm test cuối cùng, gồm lớp bình thường và
    các lớp bất thường.

Nên cố định `random_seed = 42` để kết quả có thể tái lập.

------------------------------------------------------------------------

## 4. Thiết kế mô hình Autoencoder

### 4.1. Kiến trúc cơ bản

Sử dụng Autoencoder fully-connected:

1.  Flatten ảnh từ `1 × 28 × 28` thành vector 784 chiều.
2.  Encoder nén vector đầu vào thành latent representation.
3.  Decoder tái tạo vector 784 chiều.
4.  Reshape đầu ra thành ảnh `1 × 28 × 28`.

Kiến trúc tham khảo:

``` text
Input (784)
   |
Linear(784, 256) + ReLU
   |
Linear(256, 128) + ReLU
   |
Linear(128, latent_dim)
   |
Latent representation
   |
Linear(latent_dim, 128) + ReLU
   |
Linear(128, 256) + ReLU
   |
Linear(256, 784) + Sigmoid
   |
Reconstructed image (1 × 28 × 28)
```

### 4.2. So sánh latent space

Thực hiện ít nhất hai cấu hình, giữ nguyên các thành phần khác để so
sánh công bằng:

  ------------------------------------------------------------------------
  Cấu hình                          Latent dimension Mục đích
  --------------------- ---------------------------- ---------------------
  AE-small                                        16 Không gian tiềm ẩn
                                                     nhỏ, mức nén mạnh

  AE-large                                        64 Không gian tiềm ẩn
                                                     lớn hơn, khả năng
                                                     biểu diễn cao hơn
  ------------------------------------------------------------------------

Có thể bổ sung `latent_dim = 32` nếu thời gian cho phép.

### 4.3. Hàm loss và optimizer

-   Reconstruction loss: Mean Squared Error (MSE).
-   Optimizer: Adam.
-   Learning rate ban đầu: `1e-3`.
-   Batch size: `128`.
-   Epoch tối đa: `30`.
-   Early stopping: dừng nếu validation loss không cải thiện sau 5
    epoch.
-   Chỉ tính validation loss trên ảnh bình thường để đánh giá khả năng
    tái tạo dữ liệu mà mô hình được học.

Không dùng ảnh bất thường để tối ưu loss huấn luyện.

### 4.4. Công thức reconstruction error

Với ảnh đầu vào (x), ảnh tái tạo (`\hat{x}`{=tex}), có thể dùng MSE
trung bình trên toàn bộ pixel:

\[ E(x) =
`\frac{1}{784}`{=tex}`\sum`{=tex}\_{i=1}\^{784}(x_i-`\hat{x}`{=tex}\_i)\^2
\]

Giá trị (E(x)) càng lớn thì ảnh càng có dấu hiệu bất thường so với phân
bố mà Autoencoder đã học.

------------------------------------------------------------------------

## 5. Xác định ngưỡng bất thường

### 5.1. Phương pháp chính: percentile trên validation-normal

1.  Chạy mô hình đã huấn luyện trên toàn bộ `validation-normal`.
2.  Tính reconstruction error cho từng ảnh.
3.  Lấy percentile thứ 95 của các lỗi này làm threshold ban đầu:

\[ T = P\_{95}(E\_{`\text{validation-normal}`{=tex}}) \]

4.  Quy tắc phân loại:

\[ `\hat{y}`{=tex}(x)=
```{=tex}
\begin{cases}
\text{Bình thường}, & E(x) \leq T\\
\text{Bất thường}, & E(x) > T
\end{cases}
```
\]

Lý do chọn percentile 95: ngưỡng được xác định từ dữ liệu bình thường và
cho phép khoảng 5% ảnh validation-normal có lỗi cao hơn ngưỡng. Đây là
một lựa chọn khởi đầu có thể giải thích được, không phải ngưỡng tối ưu
cho mọi bài toán.

### 5.2. Phân tích độ nhạy của threshold

Thử thêm các percentile:

-   P90: nhạy hơn với bất thường, thường tăng Recall nhưng có thể tăng
    False Positive.
-   P95: cấu hình chính.
-   P99: nghiêm ngặt hơn khi gắn nhãn bất thường, có thể giảm False
    Positive nhưng bỏ sót nhiều bất thường hơn.

So sánh Precision, Recall, F1-score và False Positive Rate trên
validation-mixed. Có thể báo cáo thêm threshold tối ưu F1 trên
validation-mixed như một thí nghiệm phụ, nhưng phải ghi rõ đây là cách
chọn ngưỡng có sử dụng nhãn validation. Không được dùng test set để tối
ưu threshold.

------------------------------------------------------------------------

## 6. Quy trình huấn luyện và đánh giá

### 6.1. Pipeline tổng thể

``` text
Fashion-MNIST
     |
Tiền xử lý và chia tập
     |
Chọn một lớp bình thường
     |
Train Autoencoder bằng ảnh bình thường
     |
Tính reconstruction error trên validation-normal
     |
Chọn threshold (P95)
     |
Đánh giá trên validation-mixed
     |
Cố định mô hình và threshold
     |
Đánh giá cuối cùng trên test
     |
Trực quan hóa + Streamlit demo
```

### 6.2. Các bước triển khai

-   [x] Khởi tạo môi trường bằng uv, cài dependencies và tải Fashion-MNIST bằng `torchvision.datasets.FashionMNIST`.
-   [x] Kiểm tra số lượng mẫu, nhãn và kích thước ảnh.
-   [x] Lọc lớp bình thường và tạo các tập train/validation/test.
-   [x] Chuẩn hóa ảnh về `[0, 1]`.
-   [x] Cài đặt Autoencoder trong PyTorch.
-   [x] Huấn luyện cấu hình latent dimension 16.
-   [x] Huấn luyện cấu hình latent dimension 64.
-   [x] Lưu trọng số và lịch sử loss.
-   [x] Tính reconstruction error trên validation-normal.
-   [x] Tính threshold P90, P95 và P99.
-   [x] Đánh giá trên validation-mixed để chọn cấu hình.
-   [x] Cố định mô hình/ngưỡng và đánh giá trên test.
-   [x] Vẽ biểu đồ kết quả huấn luyện, validation và test; trực quan hóa ví dụ FP/FN.
-   [x] Xây dựng giao diện demo Streamlit.
-   [x] Viết báo cáo, mô tả hạn chế.
-   [x] Mô tả hướng phát triển.

------------------------------------------------------------------------

## 7. Chỉ số đánh giá

Quy ước:

-   Positive: ảnh bất thường.
-   Negative: ảnh bình thường.
-   TP: ảnh bất thường được phát hiện đúng.
-   FP: ảnh bình thường bị dự đoán nhầm là bất thường.
-   TN: ảnh bình thường được nhận diện đúng.
-   FN: ảnh bất thường bị bỏ sót.

Các chỉ số cần báo cáo:

  -----------------------------------------------------------------------
  Chỉ số                              Ý nghĩa
  ----------------------------------- -----------------------------------
  ROC-AUC                             Khả năng xếp hạng ảnh bất thường có
                                      điểm lỗi cao hơn ảnh bình thường

  Precision                           Trong các ảnh bị dự đoán bất
                                      thường, tỷ lệ thực sự bất thường

  Recall                              Trong các ảnh bất thường thật, tỷ
                                      lệ được phát hiện

  F1-score                            Trung bình điều hòa giữa Precision
                                      và Recall

  Confusion Matrix                    Số lượng TP, FP, TN, FN
  -----------------------------------------------------------------------

**Lưu ý:** ROC-AUC được tính từ reconstruction error liên tục, không
phải nhãn dự đoán sau khi áp dụng threshold. Với quy ước bất thường là
positive, điểm anomaly càng cao càng nghiêng về positive.

### 7.1. Phân tích False Positive và False Negative

-   **False Positive:** ảnh Sneaker bình thường có reconstruction error
    vượt threshold. Kiểm tra xem ảnh có hình dạng/góc chụp khác biệt
    hoặc chất lượng ảnh thấp không.
-   **False Negative:** ảnh thuộc lớp khác nhưng reconstruction error
    thấp hơn threshold. Có thể do hình dạng ảnh tương tự Sneaker hoặc
    Autoencoder tái tạo được cả đặc trưng của lớp bất thường.

Chọn một số ảnh FP/FN tiêu biểu, hiển thị ảnh gốc, ảnh tái tạo và
reconstruction error để giải thích.

------------------------------------------------------------------------

## 8. Trực quan hóa kết quả

Các biểu đồ/hình ảnh tối thiểu:

1.  **Training/Validation Loss:** đường loss theo epoch cho từng latent
    dimension.
2.  **Ảnh gốc và ảnh tái tạo:** hiển thị theo từng cặp, gồm ảnh bình
    thường và ảnh bất thường.
3.  **Phân bố reconstruction error:** histogram hoặc KDE của normal và
    anomaly; vẽ đường threshold.
4.  **ROC Curve:** so sánh khả năng phân biệt của hai cấu hình.
5.  **Confusion Matrix:** thể hiện kết quả phân loại tại threshold được
    chọn.
6.  **Bảng so sánh mô hình:** latent dimension, số epoch, ROC-AUC,
    Precision, Recall, F1-score.

Không chỉ trình bày mô hình tốt nhất; cần cho thấy kết quả của cả hai
latent dimension để đáp ứng yêu cầu so sánh.

------------------------------------------------------------------------

## 9. Chương trình demo bằng Streamlit

### 9.1. Chức năng

Người dùng có thể:

1.  Tải ảnh lên (`PNG`, `JPG`, `JPEG`).
2.  Xem ảnh gốc sau khi tiền xử lý.
3.  Hệ thống chuyển ảnh về grayscale, resize thành 28 × 28 và chuẩn hóa
    về `[0, 1]`.
4.  Chọn mô hình đã huấn luyện (AE-small hoặc AE-large), nếu muốn so
    sánh.
5.  Nhấn nút dự đoán.
6.  Xem ảnh tái tạo.
7.  Xem reconstruction error và threshold.
8.  Xem kết luận:
    -   `Bình thường` nếu error ≤ threshold.
    -   `Bất thường` nếu error \> threshold.

### 9.2. Lưu ý về ảnh đầu vào

Fashion-MNIST là ảnh grayscale 28 × 28, do đó ảnh tải lên cần được tiền
xử lý nhất quán với dữ liệu huấn luyện. Demo nên ghi rõ mô hình được
huấn luyện trên Fashion-MNIST; ảnh tự tải lên có nền, tỷ lệ, màu sắc
hoặc bố cục khác biệt có thể tạo reconstruction error cao dù không phải
vật thể bất thường theo nghĩa thông thường.

### 9.3. Bố cục giao diện đề xuất

``` text
-------------------------------------------------
       FASHION-MNIST ANOMALY DETECTION
-------------------------------------------------
[ Upload image ]       [ Select model ]

[ Original image ]     [ Reconstructed image ]

Reconstruction Error: 0.0xxx
Threshold:             0.0xxx

Prediction: NORMAL / ANOMALY
-------------------------------------------------
```

------------------------------------------------------------------------

## 10. Công nghệ và cấu trúc thư mục

### 10.1. Công nghệ

-   Python 3.10+.
-   PyTorch: xây dựng và huấn luyện Autoencoder.
-   torchvision: tải Fashion-MNIST và tiền xử lý.
-   NumPy: xử lý dữ liệu và tính toán.
-   scikit-learn: ROC-AUC, Precision, Recall, F1-score, confusion
    matrix.
-   Matplotlib/Seaborn: trực quan hóa.
-   Streamlit: xây dựng demo.
-   Pillow: đọc và xử lý ảnh tải lên.

### 10.2. Cấu trúc thư mục đề xuất

``` text
autoencoder-anomaly-detection/
│
├── data/
│   └── fashion_mnist/              # Dataset tải tự động
│
├── models/
│   ├── autoencoder.py              # Định nghĩa kiến trúc
│   └── checkpoints/
│       ├── ae_latent16.pth
│       └── ae_latent64.pth
│
├── src/
│   ├── data_loader.py              # Tải, lọc và chia dữ liệu
│   ├── train.py                    # Huấn luyện mô hình
│   ├── evaluate.py                 # Tính metrics và đánh giá
│   ├── threshold.py                 # Tính và lưu threshold
│   └── inference.py                # Dự đoán một ảnh
│
├── app/
│   └── streamlit_app.py            # Giao diện demo
│
├── outputs/
│   ├── figures/                    # Biểu đồ và ảnh tái tạo
│   ├── metrics/                    # Kết quả thực nghiệm
│   └── thresholds.json             # Ngưỡng của từng mô hình
│
├── notebooks/
│   └── exploration.ipynb           # Khảo sát dữ liệu (nếu cần)
│
├── pyproject.toml                  # Khai báo project và dependencies
├── uv.lock                         # Khóa phiên bản thư viện đã resolve
├── README.md
└── plan.md
```

### 10.3. Tạo môi trường và cài thư viện bằng uv

Sử dụng **uv** để quản lý Python, môi trường ảo và dependencies thay cho `pip`/`requirements.txt`. Hướng dẫn dưới đây dành cho Windows PowerShell và ưu tiên chạy mô hình trên CPU.

#### Bước 1 — Cài uv

Nếu máy chưa có uv, chạy trong PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Đóng và mở lại terminal, sau đó kiểm tra:

```powershell
uv --version
```

Tài liệu chính thức: https://docs.astral.sh/uv/

#### Bước 2 — Khởi tạo project

Di chuyển đến thư mục dự án, sau đó chạy:

```powershell
mkdir autoencoder-anomaly-detection
cd autoencoder-anomaly-detection

uv init --python 3.11
uv python pin 3.11
```

`uv init` tạo file `pyproject.toml`; `uv python pin` cố định phiên bản Python 3.11 cho dự án. Nếu đã có thư mục dự án và `pyproject.toml`, không cần chạy lại `uv init`.

#### Bước 3 — Cài các thư viện cần thiết

```powershell
uv add torch torchvision
uv add numpy scikit-learn matplotlib seaborn streamlit pillow
```

Các thư viện sử dụng:

| Thư viện | Mục đích |
|---|---|
| `torch` | Xây dựng và huấn luyện Autoencoder |
| `torchvision` | Tải Fashion-MNIST, biến đổi ảnh |
| `numpy` | Xử lý dữ liệu số |
| `scikit-learn` | ROC-AUC, Precision, Recall, F1, confusion matrix |
| `matplotlib`, `seaborn` | Vẽ biểu đồ và phân bố reconstruction error |
| `streamlit` | Xây dựng giao diện demo |
| `pillow` | Đọc và tiền xử lý ảnh người dùng tải lên |

Với Windows và phạm vi chạy CPU của đề tài, cài `torch`/`torchvision` thông qua PyPI như lệnh trên là lựa chọn đơn giản. Không cần tự chọn CUDA nếu không huấn luyện bằng GPU. Nếu nhóm triển khai trên Linux hoặc muốn ép dùng CPU-only trên nền tảng khác, cần chọn đúng PyTorch index theo hướng dẫn chính thức của uv: https://docs.astral.sh/uv/guides/integration/pytorch/

#### Bước 4 — Kiểm tra cài đặt

```powershell
uv run python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Đề tài chạy CPU nên `CUDA available: False` vẫn là bình thường.

#### Bước 5 — Chạy chương trình bằng uv

Không cần tự kích hoạt `.venv`; dùng `uv run` để chạy lệnh trong môi trường của project:

```powershell
uv run python src/data_loader.py
uv run python src/train.py
uv run streamlit run app/streamlit_app.py
```

Sau khi thêm hoặc thay đổi thư viện, chạy `uv sync` để đồng bộ môi trường theo `pyproject.toml` và `uv.lock`.

Khi thành viên khác clone project về máy:

```powershell
uv sync
```

Lệnh này cài đúng các dependencies đã khóa trong `uv.lock`, giúp môi trường giữa các thành viên nhất quán. Nhóm cần commit cả `pyproject.toml` và `uv.lock` lên Git.

**Lưu ý:** Không cần duy trì `requirements.txt` trong quy trình chính của project này. `pyproject.toml` là nơi khai báo dependencies; `uv.lock` lưu phiên bản đã resolve để tái lập môi trường.

------------------------------------------------------------------------

## 11. Phân công công việc nhóm

  -----------------------------------------------------------------------
  Thành viên              Nhiệm vụ                Sản phẩm bàn giao
  ----------------------- ----------------------- -----------------------
  Sinh viên 1             Tải/khảo sát            `data_loader.py`, mô tả
                          Fashion-MNIST; chọn lớp dataset và cách chia
                          bình thường; chia dữ    tập
                          liệu; thiết kế thực     
                          nghiệm                  

  Sinh viên 2             Xây dựng Autoencoder;   `autoencoder.py`,
                          huấn luyện hai latent   `train.py`, checkpoint,
                          dimension; xác định     `threshold.py`
                          threshold               

  Sinh viên 3             Xây dựng demo; trực     `streamlit_app.py`,
                          quan hóa; tính          biểu đồ, bảng metrics,
                          toán/biểu diễn kết quả  phần phân tích
                          và phân tích FP/FN      
  -----------------------------------------------------------------------

**Phối hợp chung:** cả nhóm thống nhất quy tắc chia dữ liệu, random
seed, cách tính reconstruction error và định nghĩa positive/negative
trước khi chạy thực nghiệm. Sinh viên 3 cần phối hợp với sinh viên 2 để
bảo đảm demo sử dụng đúng checkpoint và threshold đã lưu.

------------------------------------------------------------------------

## 12. Kế hoạch thực hiện

  -----------------------------------------------------------------------
  Giai đoạn               Công việc               Kết quả
  ----------------------- ----------------------- -----------------------
  1\. Khảo sát            Đọc yêu cầu, tìm hiểu   Chốt lớp bình thường,
                          Autoencoder và          thiết kế dữ liệu
                          Fashion-MNIST           

  2\. Chuẩn bị dữ liệu    Tải, lọc nhãn, chia     Data pipeline chạy được
                          tập, chuẩn hóa          

  3\. Xây dựng mô hình    Cài đặt Encoder/Decoder Autoencoder hoạt động
                          và kiểm tra forward     
                          pass                    

  4\. Huấn luyện          Chạy latent 16 và 64    Checkpoint và loss
                          trên CPU                

  5\. Chọn ngưỡng         Tính error              Threshold cho từng mô
                          validation-normal, thử  hình
                          P90/P95/P99             

  6\. Đánh giá            Tính ROC-AUC,           Bảng kết quả, phân tích
                          Precision, Recall, F1   FP/FN
                          và confusion matrix     

  7\. Demo                Xây dựng giao diện      Demo tải ảnh và dự đoán
                          Streamlit               

  8\. Hoàn thiện          Viết báo cáo, chuẩn bị  Sản phẩm nộp cuối cùng
                          slide và kiểm thử       
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 13. Tiêu chí hoàn thành

-   [ ] Có pipeline tải và chia Fashion-MNIST theo one-class.
-   [ ] Autoencoder có encoder và decoder đầy đủ.
-   [ ] Huấn luyện được ít nhất hai cấu hình latent dimension trên CPU.
-   [ ] Có threshold được tính từ validation-normal và giải thích rõ
    cách chọn.
-   [ ] Có so sánh hai cấu hình trên cùng tập test.
-   [ ] Báo cáo ROC-AUC, Precision, Recall, F1-score và confusion
    matrix.
-   [ ] Có hình ảnh gốc/tái tạo và phân bố reconstruction error.
-   [ ] Có ví dụ phân tích False Positive và False Negative.
-   [ ] Demo cho phép tải ảnh và hiển thị ảnh tái tạo, anomaly score,
    threshold và kết luận.
-   [ ] Có `pyproject.toml`, `uv.lock` và README hướng dẫn cài đặt bằng uv, tải dữ liệu, huấn luyện và chạy demo.

------------------------------------------------------------------------

## 14. Rủi ro, giới hạn và hướng cải tiến

### Rủi ro/giới hạn

-   Autoencoder có thể tái tạo tốt một số ảnh bất thường, khiến
    reconstruction error thấp và gây False Negative.
-   Ảnh bình thường nhưng khác biệt về tư thế, góc nhìn hoặc chất lượng
    có thể có error cao, gây False Positive.
-   Threshold phụ thuộc vào lớp bình thường, cách tiền xử lý và phân bố
    dữ liệu.
-   Kết quả trên Fashion-MNIST không đảm bảo tổng quát hóa sang ảnh đời
    thực.
-   Latent dimension lớn hơn không mặc định đem lại kết quả phát hiện
    tốt hơn.

### Hướng cải tiến nếu còn thời gian

-   Thử Convolutional Autoencoder để tận dụng cấu trúc không gian của
    ảnh.
-   Thử thêm lớp bình thường khác để xem độ nhạy của bài toán.
-   So sánh MSE với MAE làm reconstruction error.
-   Thử nhiều percentile hoặc phương pháp chọn threshold khác.
-   Thêm hiển thị vùng sai khác giữa ảnh gốc và ảnh tái tạo.

------------------------------------------------------------------------

## 15. Tài liệu tham khảo cần đọc

-   Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
-   PyTorch Tutorials: https://pytorch.org/tutorials/
-   scikit-learn Metrics:
    https://scikit-learn.org/stable/modules/model_evaluation.html
-   Streamlit Documentation: https://docs.streamlit.io/

**Kết quả cuối cùng cần nộp:** mã nguồn, checkpoint mô hình, file
threshold, biểu đồ/thực nghiệm, ứng dụng demo và báo cáo giải thích
thiết kế, kết quả, hạn chế.
