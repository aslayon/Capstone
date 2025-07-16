import torch
print(torch.__version__)                 # → 2.2.2
print(torch.version.cuda)                # → 12.1
print(torch.cuda.is_available())         # → True
print(torch.cuda.get_device_name(0))     # → NVIDIA RTX 3070 Ti