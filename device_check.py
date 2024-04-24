import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available")
    # List all available CUDA devices
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available")