# CUDA 13.2 + cuDNN. `devel` rather than `runtime` so nvcc and the CUDA headers are
# present for building custom kernels or compiling extensions (flash-attn, etc.).
FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Pinned for reproducible rebuilds. torch/torchvision come from PyTorch's cu132
# index; these are the versions published there with cp314 (Python 3.14) wheels.
# Note there is no torchaudio cp314 wheel for cu132 yet -- add it when there is.
ARG PYTHON_VERSION=3.14
ARG NODE_MAJOR=22
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG CUDA_WHEEL_TAG=cu132

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Ubuntu 24.04 only ships Python 3.12, so 3.14 comes from deadsnakes.
# Node is needed by the claude-code devcontainer feature, which installs the CLI
# with npm (the feature declares installsAfter node but does not provide it).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git gnupg2 sudo less nano openssh-client \
      build-essential software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-venv \
      python${PYTHON_VERSION}-dev \
 && curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# The base image already ships an `ubuntu` user at UID/GID 1000, which collides
# with the devcontainer user. Reclaim that id so bind-mounted workspace files
# stay writable from inside the container.
RUN if getent passwd ${USER_UID} >/dev/null; then \
      userdel -r "$(getent passwd ${USER_UID} | cut -d: -f1)"; \
    fi \
 && if ! getent group ${USER_GID} >/dev/null; then \
      groupadd --gid ${USER_GID} ${USERNAME}; \
    fi \
 && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME}

# A user-writable global npm prefix, so the Claude Code feature installs cleanly
# and `claude update` later works without sudo.
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=/usr/local/share/npm-global/bin:${PATH}
RUN mkdir -p ${NPM_CONFIG_PREFIX} \
 && chown -R ${USER_UID}:${USER_GID} ${NPM_CONFIG_PREFIX}

# Everything Python lives in one venv that is first on PATH, so `python` and
# `pip` are unambiguously 3.14 for the shell, VS Code, and every lifecycle hook.
# A venv also sidesteps PEP 668, which blocks pip from writing to system Python.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:${PATH}
RUN python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV}

# --index-url (not --extra-index-url) so pip cannot silently resolve the CPU-only
# build from PyPI. No `|| true` anywhere: a broken GPU stack must fail the build.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/${CUDA_WHEEL_TAG} \
      "torch==${TORCH_VERSION}+${CUDA_WHEEL_TAG}" \
      "torchvision==${TORCHVISION_VERSION}+${CUDA_WHEEL_TAG}" \
 && pip install --no-cache-dir anthropic openai ipykernel

# Catch a wrong or CPU-only wheel now, at build time, instead of at first run.
# GPU visibility itself can only be checked at runtime -- see verify_gpus.py.
RUN python -c "import torch; \
assert torch.version.cuda and torch.version.cuda.startswith('13.2'), \
  f'expected a CUDA 13.2 build, got {torch.version.cuda!r}'; \
print('built ok:', torch.__version__)" \
 && chown -R ${USER_UID}:${USER_GID} ${VIRTUAL_ENV}

# Drop the CUDA forward-compatibility driver. The base image ships
# cuda-compat-13-2 (a userspace libcuda from the 595.x branch) so that a CUDA
# 13.2 image can run on an older datacenter driver. When the container toolkit
# sees a compat libcuda newer than the host driver it prepends that directory to
# the loader cache, so `libcuda.so.1` resolves to the compat copy instead of the
# driver library the toolkit injected.
#
# That is right on bare-metal Linux and wrong under WSL2 (Docker Desktop on
# Windows), where the GPU is paravirtualised: there is no /dev/nvidia*, only
# /dev/dxg, and only the injected WSL stub libcuda can talk to it. The compat
# library loads, then fails at cuInit with CUDA_ERROR_NO_DEVICE (100) -- while
# nvidia-smi still works, because NVML resolves to the injected library. The
# symptom is a healthy-looking container where torch.cuda.is_available() is
# False and device_count() is nonzero.
#
# Removing the package takes nothing away here: forward compat is only useful
# when the host driver is older than the toolkit, and it does not function under
# WSL2 at all. Revisit if this image ever has to run on a bare-metal Linux host
# with a pre-13.2 driver.
RUN apt-get purge -y cuda-compat-13-2 \
 && rm -rf /var/lib/apt/lists/*

USER ${USERNAME}
