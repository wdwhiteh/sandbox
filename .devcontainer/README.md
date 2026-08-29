# claude-dev devcontainer

An AI development container: **Claude Code**, **Python 3.14**, and **PyTorch built
for CUDA 13.2**.

## What's in it

| Component | Version | Source |
| --- | --- | --- |
| Base image | `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` | CUDA 13.2 + cuDNN, with `nvcc` and headers |
| Python | 3.14 | deadsnakes PPA (Ubuntu 24.04 only ships 3.12) |
| PyTorch | `2.13.0+cu132` | `download.pytorch.org/whl/cu132` |
| torchvision | `0.28.0+cu132` | same index |
| Node.js | 22 LTS | NodeSource — required by the Claude Code CLI |
| Claude Code | latest | `ghcr.io/anthropics/devcontainer-features/claude-code:1.0` |

Also preinstalled: `anthropic`, `openai`, `ipykernel`.

## Using it

Open the workspace in VS Code and run **Dev Containers: Reopen in Container**.

On first start you'll be prompted to authenticate Claude Code; run `claude` in the
terminal. `~/.claude` is a named Docker volume, so that login survives rebuilds.

## The Python environment

Everything lives in a single virtualenv at `/opt/venv`, first on `PATH`. So
`python` and `pip` are unambiguously 3.14-with-PyTorch — in the terminal, in VS
Code, and in every lifecycle hook. Nothing needs activating.

This matters more than it looks. The `ghcr.io/devcontainers/features/python`
feature is deliberately **not** used: it installs its own Python at
`/usr/local/python/current` and puts it ahead of everything else on `PATH`, which
would shadow the interpreter that actually has PyTorch in it.

Add project dependencies to a top-level `requirements.txt`; `post-create.sh`
installs it into the venv when the container is created.

`torchaudio` is omitted because no cp314 wheel is published for cu132 yet. Add it
to the Dockerfile once one is.

## GPU access

`hostRequirements.gpu: "optional"` attaches all GPUs when the host has the NVIDIA
Container Toolkit, and still starts the container when it doesn't — you get a
working CPU-only session rather than a container that refuses to start.

`verify_gpus.py` runs on every container start and prints the torch version, the
CUDA version it was built against, and each visible GPU. It fails only when
PyTorch itself is broken; a CPU-only host is reported, not treated as an error.

To confirm the host side independently:

```bash
docker run --rm --gpus all nvidia/cuda:13.2.1-base-ubuntu24.04 nvidia-smi
```

CUDA 13.x needs a recent driver and drops support for pre-Turing GPUs (compute
capability < 7.5). Ampere, Ada, Hopper and Blackwell cards are all fine.

## Changing versions

The pins are `ARG`s at the top of the `Dockerfile`. If you move `CUDA_WHEEL_TAG`
off `cu132`, change the base image tag to match and update the CUDA assertion in
the build-time check — that check exists to catch a silently CPU-only or
mismatched wheel during the build instead of at first run.
