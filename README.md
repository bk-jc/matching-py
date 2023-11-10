# Jarvis V2 - updated matching algorithm for CVs and jobs

The model code from Jarvis V1 is written in Go and found [here](https://bitbucket.org/talentflyxpert/csnn/src/master/).
The new version is built on a modern Python stack and highly configurable with respect to CLI parameters, including
training hyperparameters. Core functionalities include:

- Pretrained BERT model backbone for extraction embeddings (replacement of FastText)
- Three different pooling mechanisms (max, mean or CLS/attention)
- State-of-the-art training features using PyTorch Lightning
- Grid search functionality using Optuna and experiment tracking using TensorBoard
- Model artefact generation for optimised inference using ONNX

# Project setup

The project uses a minimal setup with `requirements.txt`. The CI runs python 3.10.x, but for absolute reproducibility it
is recommended to run 3.10.10, the version used for development and training.

To install this Python version, it is recommended to use [`pyenv`](https://github.com/pyenv/pyenv). You can install the
correct python version as follows:

```shell
pyenv install 3.10.10
```

To verify your installation, try running `python --version` in the root directory of the project. It should now
display `Python 3.10.10`.

From this point onwards, all you need to do is create a virtual environment and install the project:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

You can verify your installation by locally running the CI:

```shell
make test
# === X passed, Y skipped, Z warnings in 16.70s ===
```