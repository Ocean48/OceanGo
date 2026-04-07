import torch, sys, pathlib, platform
print("========================================")
print("PyTorch GPU / CUDA Diagnostic Test")
print("========================================")
print("Python        :", sys.version)
print("Torch         :", torch.__version__)
print("Compiled CUDA :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Torch location:", pathlib.Path(torch.__file__).resolve())
print("========================================\n")

if torch.cuda.is_available():
    print("SUCCESS! PyTorch can see your GPU.")
    print("Device Name  :", torch.cuda.get_device_name(0))
else:
    print("ERROR: PyTorch cannot see a compatible NVIDIA GPU.")
    print("If your computer HAS an NVIDIA graphics card, you are likely running the CPU-only version of PyTorch.")
    print("\nHOW TO FIX THIS FOR FASTER GPU TRAINING:")
    print("1. Uninstall your current Torch version:")
    print("   pip uninstall torch torchvision torchaudio -y")
    print("2. Reinstall the CUDA-enabled version from pytorch.org:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\nNote: Only NVIDIA GPUs support CUDA. AMD/Intel GPUs or basic laptops will remain on CPU mode.")
