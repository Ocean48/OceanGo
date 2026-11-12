import ctypes
import os
import pathlib
import platform
import sys
import torch


def _get_ram_gb() -> float:
    os_name = platform.system().lower()
    try:
        if os_name == "windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024 ** 3)
        elif hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


print("========================================")
print("OceanGo V2 Hardware & PyTorch Diagnostic")
print("========================================")
print("Platform      :", platform.system(), f"({platform.machine()}, {platform.release()})")
print("Python        :", sys.version.split()[0])
print("Torch Version :", torch.__version__)
print("Torch Location:", pathlib.Path(torch.__file__).resolve())
print("CPU Threads   :", os.cpu_count() or 1)
ram_gb = _get_ram_gb()
if ram_gb > 0:
    print(f"System RAM    : {ram_gb:.1f} GB")
print("========================================\n")

cuda_available = torch.cuda.is_available()
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

print("Hardware Acceleration Status:")
print(f" - CUDA Available (NVIDIA GPU)     : {cuda_available}")
if cuda_available:
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    bf16_ok = torch.cuda.is_bf16_supported()
    print(f"   Compiled CUDA version           : {torch.version.cuda}")
    print(f"   NVIDIA Device Name              : {torch.cuda.get_device_name(0)}")
    print(f"   Compute Capability / VRAM       : sm_{props.major}{props.minor} | {vram_gb:.1f} GB VRAM")
    print(f"   Native bfloat16 Tensor Cores    : {bf16_ok}")
    if props.major >= 12:
        try:
            _ = torch.zeros((2, 2), device="cuda", dtype=torch.bfloat16) @ torch.zeros((2, 2), device="cuda", dtype=torch.bfloat16)
            conv_test = torch.nn.Sequential(
                torch.nn.Conv2d(18, 64, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
            ).cuda()
            x_test = torch.randn((8, 18, 9, 9), device="cuda", dtype=torch.float32)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                y_test = conv_test(x_test)
            y_test.float().mean().backward()
            assert torch.isfinite(y_test).all().item(), "Non-finite Conv2d+BN output on Blackwell!"
            print("   Blackwell (RTX 50-series) Check : MatMul + Conv2d/BN bfloat16 finite verified (contiguous NCHW)!")
        except Exception as e:
            print(f"   WARNING: Blackwell sm_{props.major}{props.minor} kernel test failed ({e}).")
            print("   Please install CUDA 12.8+ PyTorch: pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128")

print(f" - MPS Available (Apple Silicon GPU): {mps_available}")

if cuda_available:
    print("\nSUCCESS: PyTorch detected NVIDIA GPU with CUDA acceleration!")
elif mps_available:
    print("\nSUCCESS: PyTorch detected Apple Silicon GPU with MPS (Metal Performance Shaders) acceleration!")
else:
    print("\nNOTE: Running on CPU mode (no GPU hardware accelerator detected).")
    current_os = platform.system().lower()
    if current_os in ("windows", "linux"):
        print("\nHow to enable NVIDIA GPU (including RTX 5060 Ti / RTX 5090 Blackwell & RTX 30/40 series):")
        print("1. Uninstall CPU torch: pip uninstall torch torchvision torchaudio -y")
        print("2. Install CUDA 12.8 torch: pip install torch --index-url https://download.pytorch.org/whl/cu128")
    elif current_os == "darwin":
        print("\nNote for macOS:")
        print("- Apple Silicon (M1/M2/M3/M4): Requires arm64 Python and PyTorch >= 2.0.")
        print("- Intel Macs: MPS is not supported on Intel Macs; CPU mode is used.")
print("========================================")
