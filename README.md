# sandbox

This repository contains a [development container](.devcontainer/) for
Python AI model development using [PyTorch](https://pytorch.org/), with
support for CUDA GPU acceleration. The container attaches available NVIDIA
GPUs automatically and falls back to CPU-only operation when none are
present — see [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json)
and [.devcontainer/verify_gpus.py](.devcontainer/verify_gpus.py) for details.

## Example app

[pytorch_example/](pytorch_example/) is a simple, self-contained app that
shows how a model can be created, trained, and run for inference with
PyTorch:

- [train.py](pytorch_example/train.py) builds a small feed-forward neural
  network, trains it on synthetic data (using GPU acceleration when
  available), and saves the trained weights to `model.pt`.
- [infer.py](pytorch_example/infer.py) loads those saved weights and runs
  inference on new, unseen samples.

See [pytorch_example/README.md](pytorch_example/README.md) for setup and run
instructions.
