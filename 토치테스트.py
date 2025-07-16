import torch
print(torch.__version__)
print("CUDA 가능 여부:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "X")