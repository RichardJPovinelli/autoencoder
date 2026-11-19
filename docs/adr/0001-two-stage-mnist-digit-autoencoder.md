# ADR 0001: Two-Stage Autoencoder and Bottleneck Classifier for MNIST Digit Prediction

Date: 2026-11-19
Status: Accepted

## Context

`code/autoencoder/dimension_reduction.py` defines a PyTorch `Autoencoder` (`784 -> 128 -> 64 -> 128 -> 784`) for MNIST dimensionality reduction, but it executes a 100-epoch training loop at module top level on import, fixes the bottleneck at 64 dimensions, and outputs only reconstructed images rather than digit predictions.

We need a Python 3.15 autoencoder pipeline in `code/autoencoder/` with a configurable smaller bottleneck (default `latent_dim=16`) that outputs only the predicted digit (`0`–`9`) and its associated probability (`0.0`–`1.0`), while leaving `code/autoencoder/dimension_reduction.py` intact and maintaining strict train/test separation and fast unit testability.

## Decision

Split the feature into two import-safe modules with a single directional dependency inside `code/autoencoder/`:

1. **`code/autoencoder/mnist_autoencoder.py`** (representation & unsupervised training boundary):
   - Owns `Autoencoder(latent_dim: int = 16)` (`784 -> 128 -> latent_dim -> 128 -> 784`), `flatten_mnist_images`, `set_random_seed`, and `train_autoencoder`.
   - Contains no top-level training or data-loading side effects on import.
   - Has zero dependency on digit labels or downstream classification.

2. **`code/autoencoder/mnist_digit_classifier.py`** (classification & prediction boundary):
   - Depends on `mnist_autoencoder.py` (never the reverse).
   - Owns `ExperimentConfig`, `DigitAutoencoderPredictor` (extracting bottleneck features $\mathbf{z} \in \mathbb{R}^{d}$ and fitting a probabilistic classifier on training features only), `predict_digit`, `format_prediction`, and the CLI entrypoint.
   - Exposes single-image prediction returning only `(digit, probability)` where `digit` is an `int` in `0..9` and `probability` is a `float` in `[0.0, 1.0]`.

3. **`code/autoencoder/dimension_reduction.py`**:
   - Remains unmodified so existing standalone usage and lecture examples are unaffected.

## Alternatives Considered

### Two-Stage Unsupervised Autoencoder + Bottleneck Classifier (Chosen) vs. Joint Multi-Task Autoencoder
- **Two-stage (chosen):** Trains the autoencoder purely on reconstruction of unlabeled training images, then fits a probabilistic classifier (`LogisticRegression` / `MLPClassifier`) on frozen bottleneck vectors $\mathbf{z}$ using training labels.
  - *Tradeoffs:* Directly matches the feature-extraction pattern in `lectures/lecture_12.2_CNNs_ResNets_and_Generative_Deep_Learning.tex` (line 422), isolates representation learning from classification, makes unit testing of encoder/decoder vs. classifier clean and fast, and allows swapping classifiers without retraining the autoencoder. Slightly lower ceiling accuracy than supervised CNNs, though typically >90% on MNIST.
- **Joint multi-task autoencoder:** Trains encoder, decoder, and a linear/MLP softmax head simultaneously under $\mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{CE}}$.
  - *Tradeoffs:* Tightly couples representation learning to label loss and mixes unsupervised and supervised objectives, complicating isolated component tests and requiring tuning of loss weighting $\lambda$.

### Bottleneck Dimensionality: `latent_dim=16` (Default) vs. `64` vs. `2`
- **`latent_dim=16` (chosen default, configurable):** Strong compression (49x from 784 pixels) while preserving stroke geometry and high linear/nonlinear digit separability.
- **`64`:** Less compression, larger feature space than needed for 28x28 grayscale digits.
- **`2`:** Useful for 2D scatter visualization, but discards detail needed for accurate separation of confusable digits (`4`/`9`, `3`/`5`/`8`).

## Consequences

- **Positive:**
  - Unidirectional dependency (`mnist_digit_classifier` -> `mnist_autoencoder`) prevents circular imports and keeps representation learning decoupled from classification.
  - Import-safe modules allow fast deterministic unit tests on small synthetic tensors.
  - `dimension_reduction.py` remains untouched and backward-compatible.
- **Negative / Constraints:**
  - Two-stage training optimizes reconstruction independently of digit boundaries, so classification accuracy depends on the bottleneck retaining sufficient class-discriminative structure at the chosen `latent_dim`.
