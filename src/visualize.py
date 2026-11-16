import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from torchvision import datasets, transforms

from autoencoder_anomaly_detection.autoencoder import Autoencoder
from data_loader import (
    CLASS_NAMES,
    DATA_ROOT,
    NORMAL_CLASS,
    RANDOM_SEED,
    create_one_class_splits,
)
from evaluate import dataset_labels, reconstruction_errors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
METRICS_ROOT = OUTPUT_ROOT / "metrics"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
TEST_RESULT_PATH = METRICS_ROOT / "final_test.json"
MODEL_NAMES = (16, 64)


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required evaluation artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(latent_dim: int) -> Autoencoder:
    checkpoint_path = (
        OUTPUT_ROOT / "checkpoints" / f"ae_latent{latent_dim}.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = Autoencoder(latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_loss_plot() -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for latent_dim in MODEL_NAMES:
        history = read_json(METRICS_ROOT / f"ae_latent{latent_dim}_history.json")
        epochs = [point["epoch"] for point in history["history"]]
        train_loss = [point["train_loss"] for point in history["history"]]
        validation_loss = [point["validation_loss"] for point in history["history"]]
        axis.plot(epochs, train_loss, label=f"Train latent={latent_dim}")
        axis.plot(
            epochs,
            validation_loss,
            linestyle="--",
            label=f"Validation latent={latent_dim}",
        )
    axis.set(title="Training and validation MSE", xlabel="Epoch", ylabel="MSE")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "training_loss.png", dpi=160)
    plt.close(figure)


def save_validation_plots(validation_mixed: object) -> tuple[np.ndarray, np.ndarray]:
    labels = (dataset_labels(validation_mixed) != NORMAL_CLASS).numpy().astype(
        np.int64
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    scores_by_model: list[np.ndarray] = []
    for latent_dim in MODEL_NAMES:
        model = load_model(latent_dim)
        scores = reconstruction_errors(model, validation_mixed)
        scores_by_model.append(scores)
        RocCurveDisplay.from_predictions(
            labels,
            scores,
            name=f"Latent {latent_dim}",
            ax=axis,
        )
    axis.set_title("ROC curves on validation-mixed")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "validation_roc.png", dpi=160)
    plt.close(figure)

    normal_scores = scores_by_model[0][labels == 0]
    anomaly_scores = scores_by_model[0][labels == 1]
    thresholds = read_json(OUTPUT_ROOT / "thresholds.json")["ae_latent16"]["thresholds"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(normal_scores, bins=40, alpha=0.65, density=True, label="Normal")
    axis.hist(anomaly_scores, bins=40, alpha=0.65, density=True, label="Anomaly")
    for percentile in (90, 95, 99):
        axis.axvline(
            thresholds[str(percentile)],
            linestyle="--",
            label=f"P{percentile}",
        )
    axis.set(
        title="Validation reconstruction errors (latent 16)",
        xlabel="Mean squared reconstruction error",
        ylabel="Density",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "validation_error_distribution.png", dpi=160)
    plt.close(figure)
    return labels, scores_by_model[0]


def save_test_plots(test_result: dict[str, object]) -> None:
    labels = np.asarray(test_result["labels"], dtype=np.int64)
    scores = np.asarray(test_result["reconstruction_errors"], dtype=np.float64)
    predictions = np.asarray(test_result["predictions"], dtype=np.int64)
    threshold = float(test_result["threshold"])
    if not (len(labels) == len(scores) == len(predictions)):
        raise ValueError("Final test result has inconsistent per-image arrays")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(scores[labels == 0], bins=50, alpha=0.65, density=True, label="Normal")
    axis.hist(scores[labels == 1], bins=50, alpha=0.65, density=True, label="Anomaly")
    axis.axvline(threshold, color="black", linestyle="--", label=f"P95={threshold:.4f}")
    axis.set(
        title="Final test reconstruction errors (latent 16)",
        xlabel="Mean squared reconstruction error",
        ylabel="Density",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "test_error_distribution.png", dpi=160)
    plt.close(figure)

    display = ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        labels=[0, 1],
        display_labels=["Normal", "Anomaly"],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title("Final test confusion matrix (P95)")
    display.figure_.tight_layout()
    display.figure_.savefig(FIGURES_ROOT / "test_confusion_matrix.png", dpi=160)
    plt.close(display.figure_)

    dataset = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    model = load_model(16)
    categories = (
        ("TN", (labels == 0) & (predictions == 0)),
        ("FP", (labels == 0) & (predictions == 1)),
        ("FN", (labels == 1) & (predictions == 0)),
        ("TP", (labels == 1) & (predictions == 1)),
    )
    selected_indices = [
        int(np.flatnonzero(mask)[0]) if np.any(mask) else None
        for _, mask in categories
    ]
    figure, axes = plt.subplots(4, 2, figsize=(7, 11))
    with torch.inference_mode():
        for row, ((category, _), index) in enumerate(zip(categories, selected_indices)):
            if index is None:
                for axis in axes[row]:
                    axis.axis("off")
                axes[row, 0].set_title(f"{category}: no examples")
                continue
            image, label = dataset[index]
            reconstructed = model(image.unsqueeze(0))[0, 0]
            axes[row, 0].imshow(image[0], cmap="gray", vmin=0, vmax=1)
            axes[row, 1].imshow(reconstructed, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(
                f"{category} input: {CLASS_NAMES[int(label)]}\n"
                f"error={scores[index]:.4f}"
            )
            axes[row, 1].set_title(f"{category} reconstruction")
            for axis in axes[row]:
                axis.axis("off")
    figure.suptitle("Final test examples: input and reconstruction")
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "test_reconstruction_examples.png", dpi=160)
    plt.close(figure)


def save_validation_comparison() -> None:
    rows = []
    for latent_dim in MODEL_NAMES:
        result = read_json(
            METRICS_ROOT / f"ae_latent{latent_dim}_validation.json"
        )
        metrics = result["metrics_by_percentile"]["95"]
        rows.append(
            [
                f"{latent_dim}",
                f"{result['best_epoch']}",
                f"{result['roc_auc']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
            ]
        )
    figure, axis = plt.subplots(figsize=(8, 2.4))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Latent dim", "Best epoch", "ROC-AUC", "Precision", "Recall", "F1 (P95)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    axis.set_title("Model comparison on validation data", pad=15)
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "validation_model_comparison.png", dpi=160)
    plt.close(figure)


def main() -> None:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    test_result = read_json(TEST_RESULT_PATH)
    if test_result["model"] != "ae_latent16" or test_result["threshold_percentile"] != 95:
        raise ValueError("Visualizations must use the frozen latent-16 P95 test result")

    train_data = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    splits = create_one_class_splits(train_data, test_data, random_seed=RANDOM_SEED)

    save_loss_plot()
    save_validation_plots(splits["validation_mixed"])
    save_test_plots(test_result)
    save_validation_comparison()
    print(f"Saved result figures to {FIGURES_ROOT}")


if __name__ == "__main__":
    main()
