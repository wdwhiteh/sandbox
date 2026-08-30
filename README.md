# sandbox

This is a learning sandbox for hands-on Python AI model development using
[PyTorch](https://pytorch.org/). It contains a [development container](.devcontainer/)
with PyTorch preinstalled and support for CUDA GPU acceleration — the
container attaches available NVIDIA GPUs automatically and falls back to
CPU-only operation when none are present (see
[.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) and
[.devcontainer/verify_gpus.py](.devcontainer/verify_gpus.py) for details) —
plus a worked example and tutorial series for learning how a neural network
is actually built, trained, and used.

## Example app

[pytorch_example/](pytorch_example/) is a self-contained app that shows how
a model can be created, trained, and run for inference with PyTorch:

- [train.py](pytorch_example/train.py) / [infer.py](pytorch_example/infer.py)
  — the finished version: `train.py` builds a small feed-forward neural
  network, trains it on synthetic data (using GPU acceleration when
  available), and saves the trained weights to `model.pt`; `infer.py` loads
  those weights and runs inference on new, unseen samples.
- [train1.py](pytorch_example/train1.py) through
  [train7.py](pytorch_example/train7.py) — a 7-lesson tutorial series that
  rebuilds the same model from a naive baseline up to the finished version
  above, one new idea (validation, early stopping, dropout, hyperparameter
  tuning, noise injection, CPU vs. GPU) at a time.

## How to use this repo

1. Open the repo in VS Code and reopen it in the dev container (**Reopen in
   Container**), which installs Python and PyTorch for you — see
   [.devcontainer/README.md](.devcontainer/README.md) for details.
2. Head to [pytorch_example/README.md](pytorch_example/README.md), which
   walks through the concepts (neurons, training, inference) in plain
   language and links a recommended intro video, then gives step-by-step
   instructions for running `train.py`/`infer.py` and, if you want to learn
   by experimenting, working through the `train1.py`–`train7.py` lesson
   series yourself.
