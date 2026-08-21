import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.r = r
        self.scaling = alpha / r
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        target_device = base_layer.weight.device
        if target_device.type in ["meta", "cpu"]:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(torch.zeros(r, in_features, device=target_device, dtype=target_dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=target_device, dtype=target_dtype))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, **kwargs):
        base_out = self.base_layer(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling


def inject_lora(model, target_modules=None, r=16, alpha=32, dropout=0.05):
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]

    replaced_count = 0
    for name, module in model.named_children():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            replaced_count += 1
        else:
            replaced_count += inject_lora(module, target_modules, r=r, alpha=alpha, dropout=dropout)
    return replaced_count


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        count = param.numel() if param.numel() > 0 else getattr(param, "ds_numel", 0)
        all_params += count
        if param.requires_grad:
            trainable_params += count
    ratio = 100 * trainable_params / all_params if all_params else 0.0
    print(f"Trainable params: {trainable_params:,d} || All params: {all_params:,d} || Ratio: {ratio:.4f}%")
