import torch, sys, pathlib, platform
print("Torch         :", torch.__version__)
print("Compiled CUDA :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Torch location:", pathlib.Path(torch.__file__).resolve())
