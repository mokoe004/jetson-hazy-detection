import torch
import torch.nn as nn
from pathlib import Path
import pickle

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


def load_pretrained_dehazer(
    cfg,
    device: torch.device,
    checkpoint_path: str,
    project_root: str | Path | None = None,
    strict: bool = True,
    print_info: bool = True,
):
    model = cfg_select_model(cfg, device.type)
    if print_info:
        print_model_info(model)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        if project_root is None:
            project_root = Path.cwd()
        ckpt_path = (Path(project_root) / ckpt_path).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        state = torch.load(ckpt_path, map_location=device)
    except pickle.UnpicklingError:
        # PyTorch >=2.6 defaults to weights_only=True; older checkpoints may need full unpickling.
        state = torch.load(ckpt_path, map_location=device, weights_only=False)

    if cfg.model.name == "FFANet":
        model=nn.DataParallel(model)
        model.load_state_dict(state['model'])
        model.eval()
        return model, ckpt_path
    
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=strict)
    model.eval()
    return model, ckpt_path
