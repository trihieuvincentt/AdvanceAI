import torch
from torch import nn


class Autoencoder(nn.Module):
    """Fully connected autoencoder for 28x28 grayscale images."""

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer")

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError(
                "images must have shape (batch_size, 1, 28, 28)"
            )

        flattened = images.flatten(start_dim=1)
        latent = self.encoder(flattened)
        reconstructed = self.decoder(latent)
        return reconstructed.reshape(-1, 1, 28, 28)
