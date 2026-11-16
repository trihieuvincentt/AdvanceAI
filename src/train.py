import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from autoencoder_anomaly_detection.autoencoder import Autoencoder
from data_loader import DATA_ROOT, RANDOM_SEED, create_one_class_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: Autoencoder,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_images = 0

    for images, _ in data_loader:
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            reconstructed = model(images)
            loss = loss_function(reconstructed, images)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_images += batch_size

    if total_images == 0:
        raise ValueError("Cannot calculate loss for an empty dataset")
    return total_loss / total_images


def train_model(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    latent_dim: int = 16,
    epochs: int = 30,
    batch_size: int = 128,
    patience: int = 5,
    seed: int = RANDOM_SEED,
    checkpoint_path: Path | None = None,
    history_path: Path | None = None,
) -> list[dict[str, int | float]]:
    if latent_dim not in (16, 64):
        raise ValueError("latent_dim must be 16 or 64")
    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size, and patience must be positive")

    set_seed(seed)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = Autoencoder(latent_dim=latent_dim)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, int | float]] = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_function, optimizer)
        validation_loss = run_epoch(model, validation_loader, loss_function)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train MSE: {train_loss:.6f} | "
            f"validation MSE: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "latent_dim": latent_dim,
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_validation_loss,
                        "seed": seed,
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    "Early stopping: validation loss did not improve "
                    f"for {patience} epochs."
                )
                break

    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(
                {
                    "latent_dim": latent_dim,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "history": history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        f"Best validation MSE: {best_validation_loss:.6f} "
        f"(epoch {best_epoch})"
    )
    if checkpoint_path is not None:
        print(f"Best checkpoint: {checkpoint_path}")
    if history_path is not None:
        print(f"Loss history: {history_path}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the one-class Fashion-MNIST autoencoder."
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        choices=(16, 64),
        default=16,
        help="Latent-space size (default: 16).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
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
        random_seed=args.seed,
    )

    checkpoint_path = (
        PROJECT_ROOT / "outputs" / "checkpoints" / f"ae_latent{args.latent_dim}.pth"
    )
    history_path = (
        PROJECT_ROOT / "outputs" / "metrics" / f"ae_latent{args.latent_dim}_history.json"
    )
    print(
        f"Training on {len(splits['train_normal']):,} normal images; "
        f"validating on {len(splits['validation_normal']):,} normal images."
    )
    train_model(
        train_dataset=splits["train_normal"],
        validation_dataset=splits["validation_normal"],
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )


if __name__ == "__main__":
    main()
