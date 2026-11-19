"""Two-stage MNIST digit predictor using autoencoder bottleneck features."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mnist_autoencoder import (
    Autoencoder,
    flatten_mnist_images,
    resolve_device,
    set_random_seed,
    train_autoencoder,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for MNIST autoencoder training and digit classification."""

    latent_dim: int = 16
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    seed: int = 42
    data_dir: Path = Path(__file__).resolve().parent / "data"

    def __post_init__(self) -> None:
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {self.latent_dim}.")
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}.")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}."
            )


class DigitAutoencoderPredictor:
    """Encodes MNIST digits into a bottleneck and predicts `(digit, probability)`."""

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or ExperimentConfig()
        self.device = resolve_device(device)
        set_random_seed(self.config.seed)
        self.autoencoder = Autoencoder(latent_dim=self.config.latent_dim).to(
            self.device
        )
        self.classifier: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        max_iter=400,
                        tol=1e-3,
                        random_state=self.config.seed,
                    ),
                ),
            ]
        )
        self.is_fitted = False
        self.training_losses: list[float] = []

    def _encode_loader(
        self, data_loader: DataLoader[Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        latent_batches: list[np.ndarray] = []
        label_batches: list[np.ndarray] = []
        self.autoencoder.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                flattened = flatten_mnist_images(images).to(self.device)
                latent = self.autoencoder.encoder(flattened)
                latent_batches.append(latent.cpu().numpy())
                label_batches.append(np.asarray(labels, dtype=np.int64))
        if not latent_batches:
            raise ValueError("DataLoader produced zero batches.")
        return np.concatenate(latent_batches, axis=0), np.concatenate(
            label_batches, axis=0
        )

    def fit(self, train_loader: DataLoader[Any]) -> DigitAutoencoderPredictor:
        """Train the autoencoder and fit the probabilistic classifier on training data only."""
        set_random_seed(self.config.seed)
        self.training_losses = train_autoencoder(
            model=self.autoencoder,
            train_loader=train_loader,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            device=self.device,
        )
        train_latent, train_labels = self._encode_loader(train_loader)
        self.classifier.fit(train_latent, train_labels)
        self.is_fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "DigitAutoencoderPredictor must be fitted with fit() before inference."
            )

    def predict_digit(self, image: torch.Tensor) -> tuple[int, float]:
        """Return `(predicted_digit, probability)` for a single 28x28 MNIST image tensor."""
        self._require_fitted()
        flattened = flatten_mnist_images(image)
        if flattened.shape[0] != 1:
            raise ValueError(
                f"predict_digit expects a single image, got batch size {flattened.shape[0]}."
            )

        self.autoencoder.eval()
        with torch.no_grad():
            latent = self.autoencoder.encoder(flattened.to(self.device)).cpu().numpy()

        probabilities = self.classifier.predict_proba(latent)[0]
        best_index = int(np.argmax(probabilities))
        classes = self.classifier.named_steps["classifier"].classes_
        predicted_digit = int(classes[best_index])
        probability = float(probabilities[best_index])
        return predicted_digit, probability

    def evaluate_accuracy(self, test_loader: DataLoader[Any]) -> float:
        """Evaluate top-1 digit classification accuracy on a held-out split."""
        self._require_fitted()
        test_latent, test_labels = self._encode_loader(test_loader)
        predicted_labels = self.classifier.predict(test_latent)
        return float(np.mean(predicted_labels == test_labels))


def predict_digit(
    predictor: DigitAutoencoderPredictor, image: torch.Tensor
) -> tuple[int, float]:
    """Predict `(digit, probability)` for a single image using a fitted predictor."""
    return predictor.predict_digit(image)


def format_prediction(digit: int, probability: float) -> str:
    """Format a prediction string containing only the digit and probability."""
    if not 0 <= digit <= 9:
        raise ValueError(f"digit must be in 0..9, got {digit}.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0.0, 1.0], got {probability}.")
    return f"Digit: {digit}, Probability: {probability:.4f}"


def load_mnist_datasets(data_dir: Path) -> tuple[datasets.MNIST, datasets.MNIST]:
    """Load MNIST train and test splits without leaking test samples into training."""
    should_download = not (data_dir / "MNIST").exists()
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=should_download,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=should_download,
        transform=transform,
    )
    return train_dataset, test_dataset


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for MNIST autoencoder digit prediction."""
    parser = argparse.ArgumentParser(
        description="Train an MNIST autoencoder and predict digit values with probabilities."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Autoencoder training epochs."
    )
    parser.add_argument(
        "--latent-dim", type=int, default=16, help="Bottleneck dimension."
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Adam learning rate."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory holding MNIST data.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of held-out test digits to predict and display.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run MNIST training, held-out evaluation, and sample digit predictions."""
    args = build_argument_parser().parse_args(argv)
    config = ExperimentConfig(
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    train_dataset, test_dataset = load_mnist_datasets(config.data_dir)
    train_generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=train_generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    predictor = DigitAutoencoderPredictor(config=config).fit(train_loader)
    accuracy = predictor.evaluate_accuracy(test_loader)
    print(f"Test accuracy: {accuracy:.4f}")

    sample_count = max(0, min(args.samples, len(test_dataset)))
    for idx in range(sample_count):
        image, _ = test_dataset[idx]
        digit, probability = predictor.predict_digit(image)
        print(format_prediction(digit, probability))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
