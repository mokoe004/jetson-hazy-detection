import torch
import torch.nn as nn

from models import AODNet, FFANet, LCANet, LDNet, LFDNet, GCANet

def print_model_info(model):
    """
    Creates simple print statements with model information
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print(f"Model name              : {model.__class__.__name__}")
    print(f"Device                  : {device}")
    print(f"Total parameters        : {num_params:,}")
    print(f"Trainable parameters    : {num_trainable_params:,}")
    print("=" * 50)

def cfg_select_model(cfg, device: str) -> nn.Module:
    torch_device = torch.device(device)
    if cfg.model.name == "AODNet":
        model = AODNet().to(torch_device)
    elif cfg.model.name == "FFANet":
        model = FFANet().to(torch_device)
    elif cfg.model.name == "LCANet":
        model = LCANet().to(torch_device)
    elif cfg.model.name == "LDNet":
        model = LDNet().to(torch_device)
    elif cfg.model.name == "LFDNet":
        model = LFDNet().to(torch_device)
    elif cfg.model.name == "GCANet":
        model = GCANet().to(torch_device)
    else:
        print("Model from cfg file not known. Fallback to AODNet")
        model = AODNet().to(torch_device)

    return model