
import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A minimal LoRA wrapper for nn.Linear.
    Implements: y = xW^T + b + scale * (x A^T B^T)
    where A: (r, in), B: (out, r)

    Notes:
    - Base linear weights are frozen by default.
    - LoRA weights are initialized as in common practice: A ~ Kaiming, B = 0.
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scale = self.alpha / max(self.r, 1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Freeze base weights by default
        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base.in_features
        out_f = base.out_features

        if self.r > 0:
            self.A = nn.Parameter(torch.empty(self.r, in_f))
            self.B = nn.Parameter(torch.empty(out_f, self.r))
            # Init
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r <= 0:
            return y
        # (B, *, in) @ (in, r) @ (r, out)
        lora = self.dropout(x) @ self.A.t() @ self.B.t()
        return y + self.scale * lora


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    setattr(parent, name, new_module)


def apply_lora_to_linear_layers(
    module: nn.Module,
    target_keywords: Tuple[str, ...] = ("qkv", "proj"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> int:
    """Recursively replace nn.Linear layers whose attribute name contains target keywords with LoRALinear."""
    replaced = 0
    for child_name, child in list(module.named_children()):
        # Recurse first
        replaced += apply_lora_to_linear_layers(child, target_keywords, r, alpha, dropout)

        if isinstance(child, nn.Linear) and any(k in child_name for k in target_keywords):
            _replace_module(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
    return replaced


def mark_only_lora_trainable(module: nn.Module) -> None:
    """Ensure only LoRA params are trainable (and any non-frozen modules you explicitly unfreeze elsewhere)."""
    for n, p in module.named_parameters():
        if "A" in n or "B" in n:
            p.requires_grad = True
        else:
            # keep as is (base might already be frozen)
            pass


