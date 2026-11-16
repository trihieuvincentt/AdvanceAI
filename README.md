# Autoencoder Anomaly Detection

## Set up the environment

Install `uv`, then run these commands from the project directory:

```powershell
uv sync
uv run python src\data_loader.py
```

The data loader downloads Fashion-MNIST to `data/fashion_mnist/` on its first
run and prints dataset sizes, image shape, pixel range, and a sample label.
The downloaded dataset is excluded from Git.

## Autoencoder model

The fully connected model supports the planned latent dimensions 16 and 64:

```powershell
uv run python -c "import torch; from autoencoder_anomaly_detection.autoencoder import Autoencoder; print(Autoencoder(16)(torch.rand(8, 1, 28, 28)).shape)"
```

The model expects batches shaped `(batch_size, 1, 28, 28)` and returns
reconstructed images with the same shape and pixel values in `[0, 1]`.

## Train the model

Train the default latent-dimension-16 model for up to 30 epochs on normal
Sneaker images only:

```powershell
uv run python src\train.py
```

Train the larger latent-dimension-64 model with the same settings:

```powershell
uv run python src\train.py --latent-dim 64
```

Training uses MSE loss, Adam (`lr=1e-3`), batch size 128, seed 42, and early
stopping after five epochs without validation-loss improvement. The best
checkpoint is written to `outputs/checkpoints/`; per-epoch training and
validation losses are written to `outputs/metrics/`.

## Select a threshold and evaluate

After training a model, evaluate it on validation data only:

```powershell
uv run python src\evaluate.py --latent-dim 16
uv run python src\evaluate.py --latent-dim 64
```

The script selects P90/P95/P99 thresholds using validation-normal
reconstruction errors, then reports ROC-AUC and threshold-based metrics on
validation-mixed. It saves the metrics under `outputs/metrics/` and thresholds
under `outputs/thresholds.json`. The test set is intentionally not used for
threshold or model selection.

The current validation comparison freezes `ae_latent16` with the P95
threshold. Run the final test evaluation only after the model and threshold
are fixed, and only once:

```powershell
uv run python src\evaluate_test.py
```

The script writes `outputs/metrics/final_test.json` and refuses to overwrite an
existing result. Generate the report figures from the saved validation and
final-test results with:

```powershell
uv run python src\visualize.py
```

Figures are written to `outputs/figures/`.

## Run the interactive demo

Start the Streamlit app from the project root:

```powershell
uv run streamlit run app\streamlit_app.py
```

Upload a PNG/JPG, select a trained model, and click **Phân tích ảnh**. The app
converts the image to grayscale, resizes it to 28×28, then displays the
reconstruction, MSE anomaly score, saved P95 threshold, and prediction.
Predictions on arbitrary uploaded images should be interpreted cautiously:
the model was trained only on Fashion-MNIST Sneaker images.