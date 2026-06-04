#!/usr/bin/env python3
"""
Cross-platform installer for FL simulation dependencies.
Run with:  python install.py
Detects CUDA version and installs PyTorch with GPU support automatically.
Requires Python 3.10+.
"""

import os
import re
import subprocess
import sys

TORCH_VERSION = "2.12.0"

# Ordered highest-first: (min CUDA major, min CUDA minor) -> wheel tag
CUDA_TO_WHEEL = [
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
]


def run(cmd: list[str]) -> None:
    print(f"  > {' '.join(cmd)}")
    subprocess.check_call(cmd)


def detect_cuda() -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
            if m:
                return int(m.group(1)), int(m.group(2))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def pick_wheel(cuda: tuple[int, int] | None) -> str | None:
    if cuda is None:
        return None
    for (maj, min_), tag in CUDA_TO_WHEEL:
        if cuda >= (maj, min_):
            return tag
    return None


def main() -> None:
    pip = [sys.executable, "-m", "pip"]
    req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")

    print("=" * 60)
    print("  FL Simulation — Dependency Installer")
    print("=" * 60)

    print("\n[1/3] Upgrading pip...")
    run(pip + ["install", "--upgrade", "pip"])

    print("\n[2/3] Installing PyTorch...")
    cuda = detect_cuda()
    wheel_tag = pick_wheel(cuda)

    if cuda:
        print(f"  Detected CUDA {cuda[0]}.{cuda[1]}")
    else:
        print("  No NVIDIA GPU detected")

    if wheel_tag:
        index_url = f"https://download.pytorch.org/whl/{wheel_tag}"
        print(f"  Installing torch=={TORCH_VERSION} with {wheel_tag}")
        run(pip + ["install", f"torch=={TORCH_VERSION}", "--index-url", index_url])
    else:
        print(f"  Installing torch=={TORCH_VERSION} (CPU-only)")
        run(pip + ["install", f"torch=={TORCH_VERSION}"])

    print("\n[3/3] Installing remaining dependencies...")
    run(pip + ["install", "-r", req])

    print("\n" + "=" * 60)
    print("  Installation complete — verifying...")
    check = subprocess.run(
        [
            sys.executable, "-c",
            "import torch; "
            "print(f'  torch      : {torch.__version__}'); "
            "avail = torch.cuda.is_available(); "
            "print(f'  CUDA       : {avail}'); "
            "print(f'  GPU        : {torch.cuda.get_device_name(0) if avail else \"none\"}')",
        ],
        capture_output=True,
        text=True,
    )
    print(check.stdout.strip())
    print("=" * 60)


if __name__ == "__main__":
    main()
