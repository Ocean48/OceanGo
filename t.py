import sys
import pathlib
import platform
import torch

print("========================================")
print("PyTorch Device & Hardware Acceleration Test")
print("========================================")
print("Platform      :", platform.system(), f"({platform.machine()}, {platform.release()})")
print("Python        :", sys.version.split()[0])
print("Torch Version :", torch.__version__)
print("Torch Location:", pathlib.Path(torch.__file__).resolve())
print("========================================\n")

cuda_available = torch.cuda.is_available()
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

print("Hardware Acceleration Status:")
print(f" - CUDA Available (NVIDIA GPU)     : {cuda_available}")
if cuda_available:
    print(f"   Compiled CUDA version           : {torch.version.cuda}")
    print(f"   NVIDIA Device Name              : {torch.cuda.get_device_name(0)}")

print(f" - MPS Available (Apple Silicon GPU): {mps_available}")

if cuda_available:
    print("\nSUCCESS: PyTorch detected NVIDIA GPU with CUDA acceleration!")
elif mps_available:
    print("\nSUCCESS: PyTorch detected Apple Silicon GPU with MPS (Metal Performance Shaders) acceleration!")
else:
    print("\nNOTE: Running on CPU mode (no GPU hardware accelerator detected).")
    current_os = platform.system().lower()
    if current_os == "windows":
        print("\nHow to enable NVIDIA GPU on Windows:")
        print("1. Uninstall CPU torch: pip uninstall torch torchvision torchaudio -y")
        print("2. Install CUDA torch:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    elif current_os == "darwin":
        print("\nNote for macOS:")
        print("- Apple Silicon (M1/M2/M3/M4): Requires arm64 Python and PyTorch >= 2.0.")
        print("- Intel Macs: MPS is not supported on Intel Macs; CPU mode is used.")
print("========================================")
