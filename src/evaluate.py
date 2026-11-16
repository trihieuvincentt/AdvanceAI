import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST

from autoencoder_anomaly_detection.autoencoder import Autoencoder
from data_loader import (
    DATA_ROOT,
    NORMAL_CLASS,
    RANDOM_SEED,
    create_one_class_splits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERCENTILES = (90, 95, 99)


def dataset_labels(dataset: Dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        return dataset_labels(dataset.dataset)[list(dataset.indices)]
    if isinstance(dataset, ConcatDataset):
        return torch.cat([dataset_labels(part) for part in dataset.datasets])
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets, dtype=torch.long)
    raise TypeError("Dataset must be a Fashion-MNIST dataset or a subset of one")


def reconstruction_errors(
    model: Autoencoder,
    dataset: Dataset,
    batch_size: int = 128,
) -> np.ndarray:
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    errors: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for images, _ in loader:
            reconstructed = model(images)
            per_image_error = (images - reconstructed).square().flatten(1).mean(1)
            errors.append(per_image_error.cpu())
    return torch.cat(errors).numpy()


def evaluate_checkpoint(
    checkpoint_path: Path,
    validation_normal: Dataset,
    validation_mixed: Dataset,
    batch_size: int = 128,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    latent_dim = int(checkpoint["latent_dim"])
    model = Autoencoder(latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])

    normal_errors = reconstruction_errors(model, validation_normal, batch_size)
    mixed_errors = reconstruction_errors(model, validation_mixed, batch_size)
    labels = (dataset_labels(validation_mixed) != NORMAL_CLASS).numpy().astype(np.int64)
    if len(labels) != len(mixed_errors):
        raise ValueError("Validation labels and reconstruction errors have different sizes")
    if np.unique(labels).size != 2:
        raise ValueError("Validation-mixed must contain both normal and anomaly samples")

    thresholds = {
        str(percentile): float(np.percentile(normal_errors, percentile))
        for percentile in PERCENTILES
    }
    results: dict[str, dict[str, object]] = {}
    for percentile, threshold in thresholds.items():
        predictions = (mixed_errors > threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()
        results[percentile] = {
            "threshold": threshold,
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "false_positive_rate": float(fp / (fp + tn)),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

    return {
        "latent_dim": latent_dim,
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint["best_epoch"]),
        "validation_normal_count": len(normal_errors),
        "validation_mixed_count": len(mixed_errors),
        "validation_mixed_normal_count": int((labels == 0).sum()),
        "validation_mixed_anomaly_count": int((labels == 1).sum()),
        "roc_auc": float(roc_auc_score(labels, mixed_errors)),
        "thresholds": thresholds,
        "metrics_by_percentile": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select validation thresholds and evaluate a trained model."
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        choices=(16, 64),
        default=16,
        help="Latent-space size of the trained checkpoint (default: 16).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = (
        PROJECT_ROOT / "outputs" / "checkpoints" / f"ae_latent{args.latent_dim}.pth"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train this latent-dimension model first."
        )

    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=transform,
    )
    splits = create_one_class_splits(
        train_data,
        test_data,
        random_seed=RANDOM_SEED,
    )
    result = evaluate_checkpoint(
        checkpoint_path,
        splits["validation_normal"],
        splits["validation_mixed"],
        batch_size=args.batch_size,
    )

    metrics_path = (
        PROJECT_ROOT
        / "outputs"
        / "metrics"
        / f"ae_latent{args.latent_dim}_validation.json"
    )
    thresholds_path = PROJECT_ROOT / "outputs" / "thresholds.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    thresholds: dict[str, object] = {}
    if thresholds_path.exists():
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    thresholds[f"ae_latent{args.latent_dim}"] = {
        "normal_class": NORMAL_CLASS,
        "selected_percentile": 95,
        "thresholds": result["thresholds"],
        "selected_threshold": result["thresholds"]["95"],
    }
    thresholds_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    print(f"ROC-AUC: {result['roc_auc']:.4f}")
    for percentile in PERCENTILES:
        metrics = result["metrics_by_percentile"][str(percentile)]
        print(
            f"P{percentile}: threshold={metrics['threshold']:.6f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"FPR={metrics['false_positive_rate']:.4f}"
        )
    print(f"Validation metrics: {metrics_path}")
    print(f"Thresholds: {thresholds_path}")


if __name__ == "__main__":
    main()
