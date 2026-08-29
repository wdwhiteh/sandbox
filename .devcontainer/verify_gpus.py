"""Report the state of the PyTorch/CUDA stack at container start.

Exit codes are deliberate: a broken PyTorch install is a container defect and
fails (1), while "torch is fine, this host just has no GPU" is a legitimate
CPU-only session and succeeds (0).
"""

import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"FAIL: could not import torch: {exc}", file=sys.stderr)
        return 1

    print(f"torch           : {torch.__version__}")
    print(f"built for CUDA  : {torch.version.cuda}")
    print(f"cuDNN           : {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        print("gpus            : none visible (CPU-only session)")
        print(
            "note: if you expected GPUs, check that the host has the NVIDIA "
            "Container Toolkit and that the container was started with GPU access."
        )
        return 0

    count = torch.cuda.device_count()
    print(f"gpus            : {count}")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        total_gib = props.total_memory / (1024**3)
        print(
            f"  [{i}] {props.name} "
            f"(sm_{props.major}{props.minor}, {total_gib:.1f} GiB)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
