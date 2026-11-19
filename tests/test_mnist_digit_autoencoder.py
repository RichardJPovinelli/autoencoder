import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

AUTOENCODER_DIR = Path(__file__).resolve().parents[1]
if str(AUTOENCODER_DIR) not in sys.path:
    sys.path.insert(0, str(AUTOENCODER_DIR))

from mnist_autoencoder import (
    Autoencoder,
    flatten_mnist_images,
    set_random_seed,
    train_autoencoder,
)
from mnist_digit_classifier import (
    DigitAutoencoderPredictor,
    ExperimentConfig,
    format_prediction,
    predict_digit,
)


def test_autoencoder_encodes_to_configured_latent_dim_and_reconstructs() -> None:
    model = Autoencoder(latent_dim=16)

    flat_batch = torch.rand(4, 28 * 28)
    latent_from_flat = model.encode(flat_batch)
    reconstructed_from_flat = model(flat_batch)

    assert latent_from_flat.shape == (4, 16)
    assert reconstructed_from_flat.shape == (4, 28 * 28)
    assert torch.all(reconstructed_from_flat >= 0.0)
    assert torch.all(reconstructed_from_flat <= 1.0)

    image_batch = torch.rand(3, 1, 28, 28)
    latent_from_images = model.encode(image_batch)
    reconstructed_from_images = model.decode(latent_from_images)

    assert latent_from_images.shape == (3, 16)
    assert reconstructed_from_images.shape == (3, 28 * 28)


def test_autoencoder_rejects_invalid_shapes_and_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="latent_dim"):
        Autoencoder(latent_dim=0)

    with pytest.raises(ValueError, match="shape"):
        flatten_mnist_images(torch.zeros(2, 10))

    with pytest.raises(ValueError, match="empty"):
        flatten_mnist_images(torch.zeros(0, 28 * 28))

    non_finite = torch.zeros(1, 28 * 28)
    non_finite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        flatten_mnist_images(non_finite)


def test_train_autoencoder_reduces_reconstruction_loss() -> None:
    set_random_seed(42)
    synthetic_images = torch.zeros(32, 1, 28, 28)
    synthetic_images[:16, :, :14, :] = 0.9
    synthetic_images[16:, :, 14:, :] = 0.9
    synthetic_labels = torch.cat([torch.zeros(16), torch.ones(16)]).long()
    loader = DataLoader(
        TensorDataset(synthetic_images, synthetic_labels), batch_size=16, shuffle=True
    )

    model = Autoencoder(latent_dim=8)
    epoch_losses = train_autoencoder(
        model, loader, epochs=8, learning_rate=5e-3, device=torch.device("cpu")
    )

    assert len(epoch_losses) == 8
    assert epoch_losses[-1] < epoch_losses[0]


def _make_synthetic_digit_dataset(samples_per_digit: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for digit in range(10):
        for _ in range(samples_per_digit):
            image = torch.rand(1, 28, 28, generator=generator) * 0.05
            row_start = digit * 2
            image[:, row_start : row_start + 2, :] = 0.9
            images.append(image)
            labels.append(digit)
    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))


def test_predict_digit_returns_learned_digit_and_probability() -> None:
    config = ExperimentConfig(
        latent_dim=16, epochs=15, batch_size=25, learning_rate=5e-3, seed=7
    )
    predictor = DigitAutoencoderPredictor(config=config, device=torch.device("cpu"))

    with pytest.raises(RuntimeError, match="fit"):
        predictor.predict_digit(torch.zeros(28, 28))

    train_loader = DataLoader(
        _make_synthetic_digit_dataset(samples_per_digit=10, seed=7),
        batch_size=25,
        shuffle=True,
    )
    test_dataset = _make_synthetic_digit_dataset(samples_per_digit=2, seed=99)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    predictor.fit(train_loader)

    with pytest.raises(ValueError, match="single image"):
        predictor.predict_digit(torch.zeros(2, 1, 28, 28))

    sample_image, true_label = test_dataset[6]
    predicted_digit, probability = predict_digit(predictor, sample_image)

    assert type(predicted_digit) is int
    assert 0 <= predicted_digit <= 9
    assert type(probability) is float
    assert 0.0 <= probability <= 1.0
    assert predicted_digit == int(true_label.item())
    assert predictor.evaluate_accuracy(test_loader) >= 0.8


def test_format_prediction_outputs_only_digit_and_probability() -> None:
    assert format_prediction(7, 0.98765) == "Digit: 7, Probability: 0.9877"

    with pytest.raises(ValueError, match="digit"):
        format_prediction(10, 0.5)

    with pytest.raises(ValueError, match="probability"):
        format_prediction(3, 1.2)
