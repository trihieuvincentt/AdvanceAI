from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "fashion_mnist"
NORMAL_CLASS = 7
RANDOM_SEED = 42
CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def create_one_class_splits(
    train_dataset: FashionMNIST,
    test_dataset: FashionMNIST,
    normal_class: int = NORMAL_CLASS,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Dataset]:
    targets = train_dataset.targets
    normal_indices = torch.where(targets == normal_class)[0]
    anomaly_indices = torch.where(targets != normal_class)[0]

    generator = torch.Generator().manual_seed(random_seed)
    normal_indices = normal_indices[torch.randperm(len(normal_indices), generator=generator)]
    train_size = int(len(normal_indices) * 0.8)
    train_normal_indices = normal_indices[:train_size].tolist()
    validation_normal_indices = normal_indices[train_size:].tolist()

    anomaly_indices = anomaly_indices[
        torch.randperm(len(anomaly_indices), generator=generator)
    ][: len(validation_normal_indices)]
    validation_anomaly_indices = anomaly_indices.tolist()

    validation_normal = Subset(train_dataset, validation_normal_indices)
    validation_anomaly = Subset(train_dataset, validation_anomaly_indices)
    return {
        "train_normal": Subset(train_dataset, train_normal_indices),
        "validation_normal": validation_normal,
        "validation_mixed": ConcatDataset(
            [validation_normal, validation_anomaly]
        ),
        "test_mixed": test_dataset,
    }


def main() -> None:
    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=transform,
    )

    splits = create_one_class_splits(train_dataset, test_dataset)
    test_normal_count = int((test_dataset.targets == NORMAL_CLASS).sum().item())
    test_anomaly_count = len(test_dataset) - test_normal_count
    image, label = train_dataset[0]
    print(f"Dataset directory: {DATA_ROOT}")
    print(f"Normal class: {NORMAL_CLASS} ({CLASS_NAMES[NORMAL_CLASS]})")
    print(f"Random seed: {RANDOM_SEED}")
    print(
        f"Train-normal: {len(splits['train_normal']):,} "
        f"(all {CLASS_NAMES[NORMAL_CLASS]})"
    )
    print(
        f"Validation-normal: {len(splits['validation_normal']):,} "
        f"(all {CLASS_NAMES[NORMAL_CLASS]})"
    )
    print(
        f"Validation-mixed: {len(splits['validation_mixed']):,} "
        f"({len(splits['validation_normal']):,} normal, "
        f"{len(splits['validation_normal']):,} anomaly)"
    )
    print(
        f"Test-mixed: {len(splits['test_mixed']):,} "
        f"({test_normal_count:,} normal, {test_anomaly_count:,} anomaly)"
    )
    print(f"Image shape: {tuple(image.shape)}")
    print(f"Pixel range: [{image.min().item():.1f}, {image.max().item():.1f}]")
    print(f"Sample label: {label} ({CLASS_NAMES[label]})")


if __name__ == "__main__":
    main()
