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
from torchvision import datasets, transforms

from autoencoder_anomaly_detection.autoencoder import Autoencoder
from data_loader import DATA_ROOT, NORMAL_CLASS
from evaluate import reconstruction_errors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_PATH = PROJECT_ROOT / "outputs" / "thresholds.json"
RESULT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "final_test.json"
EXPECTED_MODEL = "ae_latent16"


def main() -> None:
    if RESULT_PATH.exists():
        raise FileExistsError(
            f"Final test results already exist at {RESULT_PATH}. "
            "The test set is reserved for a single final evaluation."
        )
    if not THRESHOLDS_PATH.is_file():
        raise FileNotFoundError(f"Threshold configuration not found: {THRESHOLDS_PATH}")

    configuration = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    if configuration.get("selected_model") != EXPECTED_MODEL:
        raise ValueError(
            f"Expected the preselected model {EXPECTED_MODEL!r}; "
            f"found {configuration.get('selected_model')!r}."
        )
    selected = configuration[EXPECTED_MODEL]
    if selected["normal_class"] != NORMAL_CLASS or selected["selected_percentile"] != 95:
        raise ValueError("The frozen test configuration must use Sneaker and P95.")
    threshold = float(selected["selected_threshold"])

    checkpoint_path = (
        PROJECT_ROOT / "outputs" / "checkpoints" / f"{EXPECTED_MODEL}.pth"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["latent_dim"]) != 16:
        raise ValueError("Selected checkpoint does not match the frozen model.")

    test_data = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    model = Autoencoder(latent_dim=16)
    model.load_state_dict(checkpoint["model_state_dict"])
    errors = reconstruction_errors(model, test_data)
    labels = (test_data.targets != NORMAL_CLASS).numpy().astype(np.int64)
    predictions = (errors > threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result = {
        "evaluation": "single final evaluation; not used for model or threshold selection",
        "model": EXPECTED_MODEL,
        "latent_dim": 16,
        "checkpoint": str(checkpoint_path),
        "threshold_percentile": 95,
        "threshold": threshold,
        "normal_class": NORMAL_CLASS,
        "sample_count": len(test_data),
        "normal_count": int((labels == 0).sum()),
        "anomaly_count": int((labels == 1).sum()),
        "roc_auc": float(roc_auc_score(labels, errors)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "labels": labels.tolist(),
        "reconstruction_errors": errors.tolist(),
        "predictions": predictions.tolist(),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("x", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2)

    print(f"Final test samples: {result['sample_count']:,}")
    print(
        f"ROC-AUC={result['roc_auc']:.4f} | "
        f"Precision={result['precision']:.4f} | "
        f"Recall={result['recall']:.4f} | F1={result['f1']:.4f}"
    )
    print(f"Confusion matrix (TN, FP, FN, TP): {tn}, {fp}, {fn}, {tp}")
    print(f"Saved final test result: {RESULT_PATH}")


if __name__ == "__main__":
    main()
