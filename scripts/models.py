import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


def build_r2plus1d18_binary(
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Binary classifier for reduced EF (logit output).
    Expects input shape: (B, C, T, H, W)
    Returns logits shape: (B, 1)
    """
    weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
    model = r2plus1d_18(weights=weights)

    # Replace the final fully-connected layer (512 -> 1)
    in_features = model.fc.in_features  # should be 512
    if dropout and dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_features, 1),
        )
    else:
        model.fc = nn.Linear(in_features, 1)

    if freeze_backbone:
        # Freeze everything...
        for p in model.parameters():
            p.requires_grad = False
        # ...then unfreeze the head
        for p in model.fc.parameters():
            p.requires_grad = True

    return model


def unfreeze_layer4_and_head(model: nn.Module) -> None:
    """
    For "Stage 2" fine-tuning: unfreeze layer4 + fc only.
    """
    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze layer4 (last residual block) + fc head
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True


@torch.no_grad()
def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)