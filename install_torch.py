#!/usr/bin/env python3
"""
install_torch.py
----------------
Auto-detect the available hardware and install the correct PyTorch variant.

Supports:
  - NVIDIA CUDA  (SLURM clusters, cloud GPU instances)
  - AMD ROCm     (MI-series GPUs)
  - Apple MPS    (M-series Macs)
  - CPU-only     (laptops without a discrete GPU, CI)

Usage:
    python install_torch.py          # detect and install
    python install_torch.py --dry-run  # print the pip command without running it
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> str:
    """Run a command, return stdout (empty string on failure)."""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def detect_cuda_version() -> str | None:
    """Return the CUDA major.minor version string (e.g. '12.1'), or None."""
    # 1. nvcc
    out = _run(["nvcc", "--version"])
    m = re.search(r"release (\d+\.\d+)", out)
    if m:
        return m.group(1)

    # 2. nvidia-smi
    out = _run(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
    if m:
        return m.group(1)

    # 3. /usr/local/cuda/version.txt
    try:
        with open("/usr/local/cuda/version.txt") as f:
            m = re.search(r"(\d+\.\d+)", f.read())
            if m:
                return m.group(1)
    except FileNotFoundError:
        pass

    return None


def detect_rocm_version() -> str | None:
    """Return the ROCm major.minor version string (e.g. '6.0'), or None."""
    out = _run(["rocm-smi", "--showdriverversion"])
    m = re.search(r"(\d+\.\d+)", out)
    if m:
        return m.group(1)

    # /opt/rocm/bin/rocminfo is also reliable
    out = _run(["rocminfo"])
    if "ROCm" in out or "AMD" in out:
        m = re.search(r"ROCm.*?(\d+\.\d+)", out, re.IGNORECASE)
        if m:
            return m.group(1)
        return "unknown"

    return None


def is_apple_silicon() -> bool:
    """True when running on an Apple Silicon Mac."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# ---------------------------------------------------------------------------
# Index URL selection
# ---------------------------------------------------------------------------

# Mapping of CUDA major.minor → PyTorch wheel tag.
# Keep the closest supported tag (PyTorch doesn't build for every CUDA patch).
_CUDA_TO_WHEEL: dict[tuple[int, int], str] = {
    (12, 6): "cu126",
    (12, 4): "cu124",
    (12, 1): "cu121",
    (11, 8): "cu118",
}

_TORCH_INDEX = "https://download.pytorch.org/whl"


def cuda_wheel_tag(version_str: str) -> str:
    """Map a CUDA version string to the closest supported PyTorch wheel tag."""
    parts = version_str.split(".")
    major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    # Find the closest tag that doesn't exceed the detected version
    best: tuple[int, int] | None = None
    for (maj, min_) in sorted(_CUDA_TO_WHEEL):
        if (maj, min_) <= (major, minor):
            best = (maj, min_)
    if best is None:
        # Older CUDA — fall back to the oldest supported tag
        best = min(_CUDA_TO_WHEEL)
    return _CUDA_TO_WHEEL[best]


def rocm_wheel_tag(version_str: str) -> str:
    """Map a ROCm version string to the closest supported PyTorch wheel tag."""
    _ROCM_TO_WHEEL = {
        (6, 2): "rocm6.2",
        (6, 1): "rocm6.1",
        (6, 0): "rocm6.0",
        (5, 7): "rocm5.7",
    }
    if version_str == "unknown":
        return "rocm6.2"   # default to latest
    parts = version_str.split(".")
    major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    best: tuple[int, int] | None = None
    for (maj, min_) in sorted(_ROCM_TO_WHEEL):
        if (maj, min_) <= (major, minor):
            best = (maj, min_)
    if best is None:
        best = min(_ROCM_TO_WHEEL)
    return _ROCM_TO_WHEEL[best]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TORCH_PACKAGES = ["torch", "torchvision"]


def build_pip_command(index_url: str | None) -> list[str]:
    cmd = [sys.executable, "-m", "pip", "install"] + TORCH_PACKAGES
    if index_url:
        cmd += ["--index-url", index_url]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the install command without executing it")
    args = parser.parse_args()

    print("=== RG-CAT PyTorch installer ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print()

    index_url: str | None = None
    label: str

    # 1. ROCm takes precedence over CUDA (some AMD machines also have nvcc stubs)
    rocm = detect_rocm_version()
    if rocm is not None:
        tag = rocm_wheel_tag(rocm)
        index_url = f"{_TORCH_INDEX}/{tag}"
        label = f"AMD ROCm {rocm} → wheel tag {tag}"

    # 2. NVIDIA CUDA
    elif (cuda := detect_cuda_version()) is not None:
        tag = cuda_wheel_tag(cuda)
        index_url = f"{_TORCH_INDEX}/{tag}"
        label = f"NVIDIA CUDA {cuda} → wheel tag {tag}"

    # 3. Apple MPS
    elif is_apple_silicon():
        index_url = None   # default PyPI wheel includes MPS
        label = "Apple Silicon MPS — using default PyPI wheel"

    # 4. CPU-only fallback
    else:
        index_url = f"{_TORCH_INDEX}/cpu"
        label = "CPU-only"

    print(f"Detected hardware: {label}")
    cmd = build_pip_command(index_url)
    print(f"Install command:   {' '.join(cmd)}")
    print()

    if args.dry_run:
        print("Dry-run mode — not executing.")
        return

    print("Installing...")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("\nInstallation failed. Check the output above.", file=sys.stderr)
        sys.exit(ret.returncode)

    print("\nDone. Verify with:  python -c \"import torch; print(torch.__version__)\"")


if __name__ == "__main__":
    main()
