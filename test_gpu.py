#!/usr/bin/env python3
"""
test_gpu.py — Verify CUDA / GPU availability after environment setup.
Target: NVIDIA RTX 4060 | Driver 535.288.01 | CUDA 12.2
"""

import sys

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is not installed. Run setup.sh first.")
    sys.exit(1)

print("=" * 50)
print("  GPU / CUDA Verification")
print("=" * 50)

cuda_available = torch.cuda.is_available()
print(f"  PyTorch Version  : {torch.__version__}")
print(f"  CUDA Available   : {cuda_available}")

if cuda_available:
    gpu_name    = torch.cuda.get_device_name(0)
    cuda_ver    = torch.version.cuda
    device_count = torch.cuda.device_count()
    mem_total   = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    mem_free    = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)

    print(f"  GPU Name         : {gpu_name}")
    print(f"  CUDA Version     : {cuda_ver}")
    print(f"  Device Count     : {device_count}")
    print(f"  VRAM Total       : {mem_total:.2f} GB")
    print(f"  VRAM Free        : {mem_free:.2f} GB")

    # Quick tensor smoke-test on GPU
    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print(f"  Tensor Smoke Test: PASSED ✓ (1024×1024 matmul on GPU)")
    except Exception as e:
        print(f"  Tensor Smoke Test: FAILED ✗ — {e}")
else:
    print("  GPU Name         : None (CPU-only mode)")
    print()
    print("  WARNING: CUDA not available.")
    print("  Possible causes:")
    print("    - NVIDIA driver not installed / incompatible")
    print("    - PyTorch CPU-only wheel was installed instead of CUDA wheel")
    print("    - Running inside a container without GPU passthrough")
    print()
    print("  Re-run setup.sh to reinstall the correct CUDA wheel.")

print("=" * 50)
