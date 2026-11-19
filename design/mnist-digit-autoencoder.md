# Feature Design: MNIST Digit-Predicting Autoencoder

Status: Approved
Architecture: [docs/adr/0001-two-stage-mnist-digit-autoencoder.md](../docs/adr/0001-two-stage-mnist-digit-autoencoder.md)
Requirements: Encode 28x28 MNIST images through a configurable autoencoder bottleneck (default `latent_dim=16`) and output only the predicted digit (`0`–`9`) and its associated probability (`0.0`–`1.0`).

## Behavior

1. **Autoencoder representation (`code/autoencoder/mnist_autoencoder.py`):**
   - `Autoencoder(latent_dim=16)` maps flattened 28x28 grayscale images `(B, 784)` in `[0.0, 1.0]` through `784 -> 128 -> latent_dim` (`ReLU` hidden activations) to bottleneck vectors $\mathbf{z} \in \mathbb{R}^{B \times \text{latent\_dim}}$, and decodes $\mathbf{z}$ through `latent_dim -> 128 -> 784` with a `Sigmoid` output layer back into `[0.0, 1.0]`.
   - `train_autoencoder` trains the autoencoder unsupervised with `BCELoss` and `Adam` on unlabeled training images only and returns the per-epoch mean training loss.
   - Importing `mnist_autoencoder` performs no dataset download, training, or plotting.

2. **Digit prediction & CLI (`code/autoencoder/mnist_digit_classifier.py`):**
   - `DigitAutoencoderPredictor.fit(train_loader)` runs in two stages:
     1. Trains `Autoencoder(latent_dim)` on training images only.
     2. Extracts frozen bottleneck vectors $\mathbf{z}_{\text{train}} = \text{encoder}(\mathbf{x}_{\text{train}})$ and fits a probabilistic classifier pipeline (`StandardScaler` + `MLPClassifier`) mapping $\mathbf{z}_{\text{train}} \to \{0,\dots,9\}$.
   - `predict_digit(image)` accepts one image tensor of shape `(784,)`, `(1, 784)`, `(28, 28)`, `(1, 28, 28)`, or `(1, 1, 28, 28)` and returns only `(digit, probability)` where `digit` is a Python `int` in `0..9` and `probability` is a Python `float` in `[0.0, 1.0]` equal to $\max_c P(y = c \mid \mathbf{z})$.
   - `evaluate_accuracy(test_loader)` returns top-1 classification accuracy in `[0.0, 1.0]` on a held-out split without modifying model or scaler state.
   - The CLI runner trains on the MNIST training split (`code/autoencoder/data`), reports held-out test accuracy, and prints sample predictions containing only the predicted digit and its probability (`Digit: <d>, Probability: <p:.4f>`).

## Interfaces

### `code/autoencoder/mnist_autoencoder.py`
- `INPUT_DIM: int = 28 * 28`
- `set_random_seed(seed: int) -> None`
- `flatten_mnist_images(images: torch.Tensor) -> torch.Tensor`
  - Accepts shapes `(784,)`, `(B, 784)`, `(28, 28)`, `(B, 28, 28)`, or `(B, 1, 28, 28)` with `B >= 1`.
  - Validates finite entries (`torch.isfinite`) and converts to `float32` tensor of shape `(B, 784)`.
- `class Autoencoder(nn.Module)`:
  - `__init__(self, latent_dim: int = 16) -> None`
  - `encode(self, images: torch.Tensor) -> torch.Tensor` -> `(B, latent_dim)`
  - `decode(self, latent_vectors: torch.Tensor) -> torch.Tensor` -> `(B, 784)`
  - `forward(self, images: torch.Tensor) -> torch.Tensor` -> `(B, 784)`
- `train_autoencoder(model: Autoencoder, train_loader: DataLoader[Any], epochs: int = 10, learning_rate: float = 1e-3, device: torch.device | None = None) -> list[float]`

### `code/autoencoder/mnist_digit_classifier.py`
- `@dataclass(frozen=True) class ExperimentConfig`:
  - `latent_dim: int = 16`
  - `epochs: int = 10`
  - `batch_size: int = 256`
  - `learning_rate: float = 1e-3`
  - `seed: int = 42`
  - `data_dir: Path = Path("code/autoencoder/data")`
  - `checkpoint_path: Path | None = None`
- `class DigitAutoencoderPredictor`:
  - `__init__(self, config: ExperimentConfig | None = None, device: torch.device | None = None) -> None`
  - `fit(self, train_loader: DataLoader[Any]) -> DigitAutoencoderPredictor`
  - `predict_digit(self, image: torch.Tensor) -> tuple[int, float]`
  - `evaluate_accuracy(self, test_loader: DataLoader[Any]) -> float`
  - `save_checkpoint(self, path: Path) -> None`
  - `@classmethod load_checkpoint(cls, path: Path, device: torch.device | None = None) -> DigitAutoencoderPredictor`
- `predict_digit(predictor: DigitAutoencoderPredictor, image: torch.Tensor) -> tuple[int, float]`
- `format_prediction(digit: int, probability: float) -> str` -> `f"Digit: {digit}, Probability: {probability:.4f}"`
- `main(argv: Sequence[str] | None = None) -> int`

## Data and state

- **Train/test isolation (ML hygiene):**
  - Only `train_dataset` (`train=True`, 60,000 samples) is passed to `fit()`.
  - `StandardScaler` and `MLPClassifier` inside the scikit-learn `Pipeline` are fitted solely on training latent vectors $\mathbf{z}_{\text{train}}$ and training labels $y_{\text{train}}$.
  - `test_dataset` (`train=False`, 10,000 samples) is read exclusively in `evaluate_accuracy` and sample inference with `model.eval()` and `torch.no_grad()`.
- **Reproducibility:**
  - `ExperimentConfig` captures all hyperparameters (`latent_dim`, `epochs`, `batch_size`, `learning_rate`, `seed`, `data_dir`).
  - `set_random_seed(config.seed)` seeds Python `random`, NumPy, and PyTorch before initializing weights and DataLoader shuffling.
- **Checkpoints (optional):**
  - Saved only when `--checkpoint` (or `save_checkpoint`) is explicitly requested so repository runs stay clean by default.

## Error handling

- Raise `ValueError` when:
  - `latent_dim < 1`, `epochs < 1`, `batch_size < 1`, or `learning_rate <= 0`.
  - `images` has an unsupported rank/shape that does not represent `(B, 784)` or `(B, [1,] 28, 28)`, or has zero samples (`B == 0`).
  - `images` contains `NaN` or `Inf` values.
  - `predict_digit` receives a batch with `B != 1` (single-image prediction contract).
  - `digit` is outside `0..9` or `probability` is outside `[0.0, 1.0]` in `format_prediction`.
- Raise `RuntimeError` when `predict_digit`, `evaluate_accuracy`, or `save_checkpoint` is called before `fit()`.

## Test strategy

- **Unit tests (`code/autoencoder/tests/test_mnist_digit_autoencoder.py`):**
  - Fast, deterministic tests on synthetic tensors (no 60,000-sample MNIST training during unit tests):
    1. `Autoencoder` encodes `(B, 784)` and `(B, 1, 28, 28)` tensors to `(B, latent_dim)` and reconstructs `(B, 784)` in `[0.0, 1.0]`.
    2. Input and configuration validation (`latent_dim < 1`, invalid tensor shapes, `NaN`/`Inf`, batch size != 1 on `predict_digit`, unfitted `RuntimeError`).
    3. `train_autoencoder` decreases reconstruction `BCELoss` over epochs on seeded synthetic images.
    4. `DigitAutoencoderPredictor.fit` and `predict_digit` learn distinct synthetic digit patterns and return `(digit: int, probability: float)` in `0..9` and `[0.0, 1.0]`.
    5. `format_prediction` outputs only the digit and associated probability.
- **End-to-end integration verification:**
  - Run `code/autoencoder/.venv/bin/python code/autoencoder/mnist_digit_classifier.py --epochs 10 --latent-dim 16` against local `code/autoencoder/data/MNIST` and verify held-out test accuracy (> 90% expected) and sample `(digit, probability)` predictions.

## Change log

- 2026-11-19 increment 1: Initial feature design created under ADR 0001.
