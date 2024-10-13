import torch

def SetCuda():
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backend.mps.is_available()
            else "cpu"
    )
    print(f"Using {device} device")
    return device   