import torch
import torch.nn as nn

from models import AODnet, FFANet, LCANet, LDNet, LFDNet

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
    device = torch.device(device)
    if cfg.model.name == "AODnet":
        model = AODnet().to(device)
    elif cfg.model.name == "FFANet":
        model = FFANet().to(device)
    elif cfg.model.name == "LCANet":
        model = LCANet().to(device)
    elif cfg.model.name == "LDNet":
        model = LDNet().to(device)
    elif cfg.model.name == "LDFNet":
        model = LFDNet().to(device)
    else:
        print("Model from cfg file not known. Fallback to AODNet")
        model = AODnet().to(device)

    return model