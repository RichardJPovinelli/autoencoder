"""MNIST autoencoder architecture and image-flattening utilities."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

INPUT_DIM = 28 * 28
HIDDEN_DIM = 128


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_mnist_images(images: torch.Tensor) -> torch.Tensor:
    """Reshape single or batched 28x28 MNIST tensors into shape `(B, 784)`."""
    if images.ndim == 1 and images.shape[0] == INPUT_DIM:
        flattened = images.unsqueeze(0)
    elif images.ndim == 2 and images.shape == (28, 28):
        flattened = images.reshape(1, INPUT_DIM)
    elif images.ndim == 2 and images.shape[1] == INPUT_DIM:
        flattened = images
    elif (
        images.ndim == 3
        and images.shape[1:] == (28, 28)
        or images.ndim == 4
        and images.shape[1:] == (1, 28, 28)
    ):
        flattened = images.reshape(images.shape[0], INPUT_DIM)
    else:
        raise ValueError(f"Unsupported MNIST image tensor shape: {tuple(images.shape)}")

    if flattened.shape[0] == 0:
        raise ValueError("Input batch must not be empty.")
    if not torch.all(torch.isfinite(flattened)):
        raise ValueError("Input tensor must contain only finite values.")

    return flattened.to(dtype=torch.float32)


class Autoencoder(nn.Module):
    """Symmetric fully connected autoencoder with a configurable bottleneck."""

    def __init__(self, latent_dim: int = 16) -> None:
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}.")
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, INPUT_DIM),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode input image tensor(s) into bottleneck vectors."""
        flattened = flatten_mnist_images(images)
        return self.encoder(flattened)

    def decode(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck vectors into flattened `[0, 1]` reconstructions."""
        return self.decoder(latent_vectors)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode and reconstruct input image tensor(s)."""
        return self.decode(self.encode(images))


def resolve_device(device: torch.device | None = None) -> torch.device:
    """Return the explicit device or auto-select CUDA when available."""
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_autoencoder(
    model: Autoencoder,
    train_loader: DataLoader[Any],
    epochs: int = 10,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Train the autoencoder on unlabeled images and return per-epoch mean BCE loss."""
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}.")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")

    target_device = resolve_device(device)
    model.to(target_device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses: list[float] = []
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            flattened = flatten_mnist_images(images).to(target_device)

            optimizer.zero_grad()
            reconstructed = model(flattened)
            loss = criterion(reconstructed, flattened)
            loss.backward()
            optimizer.step()

            batch_size = flattened.shape[0]
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise ValueError("train_loader produced zero samples.")
        epoch_losses.append(total_loss / sample_count)

    model.eval()
    return epoch_losses
