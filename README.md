# MNIST Digit-Predicting Autoencoder

This repository implements a two-stage PyTorch autoencoder and probabilistic bottleneck classifier for 28x28 MNIST handwritten digits (Python 3.15):

1. **Unsupervised autoencoder (`mnist_autoencoder.py`):** Compresses 784-pixel MNIST images through `784 -> 128 -> latent_dim` (default `latent_dim=16`) into a low-dimensional bottleneck vector $z$ and reconstructs the image via `latent_dim -> 128 -> 784` using binary cross-entropy loss.
2. **Bottleneck digit classifier (`mnist_digit_classifier.py`):** Extracts frozen bottleneck features $z$ from the training split, fits a standardized multi-layer perceptron (`StandardScaler` + `MLPClassifier`), and predicts only the digit value (`0`–`9`) and its associated probability (`0.0`–`1.0`) via `predict_digit(image) -> tuple[int, float]`.

## Repository structure

- `mnist_autoencoder.py` — `Autoencoder`, `flatten_mnist_images`, `set_random_seed`, and `train_autoencoder`.
- `mnist_digit_classifier.py` — `ExperimentConfig`, `DigitAutoencoderPredictor`, `predict_digit`, `format_prediction`, and the CLI runner (`--epochs`, `--latent-dim`, `--batch-size`, `--learning-rate`, `--seed`, `--data-dir`, `--samples`).
- `tests/test_mnist_digit_autoencoder.py` — `pytest` unit test suite.
- `pyproject.toml` — Project metadata and dependency declarations.

# Run the Python program

These instructions use Python 3.15 on Windows and macOS. Install Git and uv once, then follow the same clone and run steps on either platform.

## One-time setup

### Windows (PowerShell)

Open **PowerShell** from the Start menu and run these commands. Skip any tool you already have.

```powershell
winget install --id Git.Git -e --source winget
winget install --id astral-sh.uv -e --source winget
```

Close and reopen PowerShell after installing. If `winget` is unavailable, see the [Git for Windows installer](https://git-scm.com/install/win) and the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

### macOS (Terminal)

Open **Terminal** from Spotlight. If you already use Homebrew, run:

```sh
brew install git uv
```

Otherwise, if Git is missing or macOS prompts you to install developer tools, install Apple's Command Line Tools and wait for the installation to finish:

```sh
xcode-select --install
```

Then install uv using the [official installer](https://docs.astral.sh/uv/getting-started/installation/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen Terminal after installing.

## Clone and run on either platform

1. **Copy the clone URL from GitHub.** Open the repository's main page in your browser. If you are viewing this README as a separate file, click the repository name near the top of the page to return to the file list. Click the green **Code** button above the files, select **HTTPS**, and click the copy icon next to the URL. [GitHub’s illustrated cloning guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

2. **Clone the repository.** Open PowerShell on Windows or Terminal on macOS and go to the folder where you want to keep your coursework. Run the following commands to clone this repository and enter its folder:

   ```sh
   git clone https://github.com/RichardJPovinelli/autoencoder.git autoencoder
   cd autoencoder
   ```

   If you already cloned the repository, open that folder in PowerShell or Terminal and go straight to step 3.

3. From the folder containing `mnist_digit_classifier.py` and `requirements.txt`, run:

   ```sh
   uv run --python 3.15 --with-requirements requirements.txt python mnist_digit_classifier.py
   ```

   On first use, uv downloads Python 3.15 if needed and installs the packages listed in `requirements.txt`. To run the program again, open the same folder and repeat the command.

   If the program uses only the Python standard library and has no `requirements.txt`, run:

   ```sh
   uv run --python 3.15 python mnist_digit_classifier.py
   ```

## Check your setup

If you have trouble, run these commands one at a time:

```sh
git --version
uv --version
uv run --python 3.15 python --version
```

The last command should print `Python 3.15.x`, where `x` is the patch version. If the program reports that it cannot find `mnist_digit_classifier.py` or `requirements.txt`, check that you are in the assignment folder and that the filenames match those in the repository.
