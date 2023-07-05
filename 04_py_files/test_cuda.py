import torch
if torch.cuda.is_available():
    print("GPU-AVAILABLE")
else:
    print("GPU-NA")

