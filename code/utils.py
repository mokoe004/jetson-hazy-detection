import torch

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