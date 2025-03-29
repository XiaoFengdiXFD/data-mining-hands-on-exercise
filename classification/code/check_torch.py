import torch

# 检查 PyTorch 版本
print("PyTorch version:", torch.__version__)

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA is available.")
    # 显示可用 GPU 的数量
    print("Number of GPUs available:", torch.cuda.device_count())
    # 显示当前使用的 GPU 名称
    print("Current GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. You are using the CPU version.")
