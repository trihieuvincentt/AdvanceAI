import json
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize

from autoencoder_anomaly_detection.autoencoder import Autoencoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"
THRESHOLDS_PATH = PROJECT_ROOT / "outputs" / "thresholds.json"
MODEL_OPTIONS = {
    "AE-small (latent 16)": ("ae_latent16", 16),
    "AE-large (latent 64)": ("ae_latent64", 64),
}


@st.cache_resource
def load_model(model_key: str, latent_dim: int) -> tuple[Autoencoder, float]:
    if not THRESHOLDS_PATH.is_file():
        raise FileNotFoundError(f"Threshold configuration not found: {THRESHOLDS_PATH}")
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    if model_key not in thresholds:
        raise KeyError(f"No saved threshold configuration for {model_key}")

    checkpoint_path = CHECKPOINT_ROOT / f"{model_key}.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["latent_dim"]) != latent_dim:
        raise ValueError(f"Checkpoint latent dimension does not match {model_key}")

    model = Autoencoder(latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    threshold = float(thresholds[model_key]["selected_threshold"])
    return model, threshold


def preprocess_image(image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
    grayscale = ImageOps.exif_transpose(image).convert("L")
    resized = resize(
        grayscale,
        [28, 28],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    tensor = pil_to_tensor(resized).to(dtype=torch.float32).div_(255.0)
    return resized, tensor.unsqueeze(0)


def main() -> None:
    st.set_page_config(
        page_title="Fashion-MNIST Anomaly Detection",
        page_icon="👟",
        layout="centered",
    )
    st.title("Fashion-MNIST Anomaly Detection")
    st.write(
        "Tải ảnh để so sánh reconstruction error với ngưỡng P95 của mô hình "
        "Autoencoder huấn luyện trên lớp Sneaker."
    )
    st.warning(
        "Mô hình được huấn luyện trên ảnh Fashion-MNIST 28×28. Ảnh tải lên "
        "có nền, bố cục hoặc tỷ lệ khác biệt có thể bị đánh dấu bất thường "
        "dù vật thể không bất thường trong thực tế."
    )

    selected_label = st.selectbox("Chọn mô hình", tuple(MODEL_OPTIONS))
    uploaded_file = st.file_uploader(
        "Tải ảnh lên",
        type=["png", "jpg", "jpeg"],
        help="Ảnh sẽ được chuyển sang grayscale và resize về 28×28.",
    )

    if uploaded_file is None:
        st.info("Chọn một ảnh PNG hoặc JPG để bắt đầu.")
        return

    try:
        uploaded_image = Image.open(BytesIO(uploaded_file.getvalue()))
        original_image, input_tensor = preprocess_image(uploaded_image)
    except UnidentifiedImageError:
        st.error("Không thể đọc file này dưới dạng ảnh hợp lệ.")
        return

    if not st.button("Phân tích ảnh", type="primary"):
        st.image(original_image, caption="Ảnh sau tiền xử lý (grayscale, 28×28)")
        return

    model_key, latent_dim = MODEL_OPTIONS[selected_label]
    model, threshold = load_model(model_key, latent_dim)
    with torch.inference_mode():
        reconstructed = model(input_tensor)
        error = float((input_tensor - reconstructed).square().mean().item())
    reconstructed_image = reconstructed[0, 0].cpu().numpy()

    original_column, reconstructed_column = st.columns(2)
    original_column.image(
        original_image,
        caption="Ảnh đầu vào đã tiền xử lý",
        clamp=True,
    )
    reconstructed_column.image(
        reconstructed_image,
        caption="Ảnh tái tạo",
        clamp=True,
    )
    st.metric("Reconstruction error (MSE)", f"{error:.6f}")
    st.metric("Ngưỡng P95", f"{threshold:.6f}")
    if error > threshold:
        st.error("Dự đoán: BẤT THƯỜNG")
    else:
        st.success("Dự đoán: BÌNH THƯỜNG")


if __name__ == "__main__":
    main()
