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

You don't need any prior coding experience to get this running — just follow the
steps below in order.

### 1. One-time computer setup

1. Install [Visual Studio Code](https://code.visualstudio.com/).
2. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and
   make sure it's running (look for the whale icon in your system tray / menu
   bar).
3. In VS Code, open the Extensions panel (the icon that looks like four squares
   in the left sidebar), search for **Dev Containers**, and install it.

### 2. Open this project in its container

1. In VS Code, choose **File > Open Folder...** and open this repository.
2. A pop-up should appear in the bottom-right corner asking **"Reopen in
   Container?"** — click it.
   - If no pop-up appears, press `Ctrl+Shift+P` (`Cmd+Shift+P` on a Mac) to open
     the Command Palette, type **Dev Containers: Reopen in Container**, and
     press Enter.
3. Wait for the container to build. The first build downloads and installs
   Python, PyTorch, and CUDA, so it can take several minutes — you'll see a log
   window showing progress. Later reopens are much faster since everything is
   cached.
4. When it's done, the blue status bar in the bottom-left corner of VS Code will
   read something like **"Dev Container: claude-dev"**. That means you're now
   working inside the container, with Python and PyTorch ready to go.

### 3. Sign in to Claude Code

The Claude Code panel in the sidebar and the `claude` command in the terminal
share the same login, so you only need to sign in once.

1. Open a terminal in VS Code: **Terminal > New Terminal** (or press
   `` Ctrl+` ``).
2. Type `claude` and press Enter.
3. A sign-in link will be printed — click it, or copy/paste it into your
   browser, and follow the steps to log in with your Anthropic account.
4. Once you see a confirmation, return to VS Code. You're signed in, in both the
   terminal and the Claude Code panel. `~/.claude` is a named Docker volume, so
   this login survives container rebuilds — you shouldn't need to repeat this
   step on the same machine.

## The Python environment

*This section explains what's happening under the hood. You don't need it to
run the example app — skip ahead if you're just getting started.*

There's only one Python installed in the container, living in a virtual
environment (an isolated, self-contained install of Python and its packages,
kept separate from the rest of the system) at `/opt/venv`. That location is
placed first on the `PATH` (the list of folders your terminal searches when
you type a command), so typing `python` or `pip` — in the terminal, in VS
Code, or in any setup script — always means "Python 3.14 with PyTorch already
installed." There's no separate environment to remember to activate.

This is deliberate: the standard `ghcr.io/devcontainers/features/python`
devcontainer feature is **not** used, because it would install its own,
separate Python at `/usr/local/python/current` ahead of `/opt/venv` on
`PATH` — silently replacing the `python` command with a version that doesn't
have PyTorch installed.

To add your own Python packages, list them in a top-level `requirements.txt`
file. They're installed automatically into `/opt/venv` when the container is
first created (see `post-create.sh`).

One current limitation: `torchaudio` isn't installed, because no build
compatible with Python 3.14 + CUDA 13.2 has been published yet. It can be
added to the `Dockerfile` once one is available.

## GPU access

This container works whether or not your computer has an NVIDIA GPU:

- **With a GPU** (and the NVIDIA Container Toolkit installed on the host),
  the container attaches it automatically and PyTorch uses it.
- **Without a GPU**, the container still starts normally — just without GPU
  acceleration. The [example app](../pytorch_example/) automatically falls
  back to the CPU in that case; no configuration needed.

This behavior comes from the `hostRequirements.gpu: "optional"` setting in
`devcontainer.json`.

Every time the container starts, `verify_gpus.py` runs automatically and
prints the installed PyTorch version, the CUDA version it was built against,
and any GPUs it can see. It only fails if PyTorch itself is broken — not
having a GPU is treated as a normal, expected outcome, not an error.

To check GPU access at the Docker level, independent of this project:

```bash
docker run --rm --gpus all nvidia/cuda:13.2.1-base-ubuntu24.04 nvidia-smi
```

Note: CUDA 13.x requires a recent NVIDIA driver and only supports newer GPUs
— Ampere, Ada, Hopper, and Blackwell generations (roughly RTX 30-series and
newer). Older cards aren't supported.

## Changing versions

The Python, PyTorch, and CUDA versions are pinned near the top of the
`Dockerfile` as `ARG` values (build-time settings). If you change
`CUDA_WHEEL_TAG` — to move to a newer CUDA release, for example — also update
the base image tag to match, and update the CUDA version check that runs
during the build. That check exists to catch a mismatched or accidentally
CPU-only PyTorch build immediately, rather than discovering the problem later
at runtime.
