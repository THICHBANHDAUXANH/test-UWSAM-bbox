
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmdet.registry import MODELS

# segment_anything must be installed in the environment
from segment_anything import sam_model_registry

from .sam_lora import apply_lora_to_linear_layers


@MODELS.register_module()
class SAMImageEncoderLoRABackbone(nn.Module):
    """SAM image encoder (ViT) as an MMDet backbone, with LoRA adapters.

    Paper-faithful intent (UWSAM teacher):
    - Use SAM ViT-Huge image encoder as teacher backbone.
    - Apply LoRA to attention linears (qkv/proj).
    - Freeze base weights; train LoRA + downstream heads.

    Output:
    - A single feature map tensor [B, 256, H/16, W/16] (SAM embedding).
      Returned as a tuple to match MMDet backbone conventions.
    """
    def __init__(
        self,
        sam_type: str = "vit_h",
        sam_checkpoint: str = "",
        img_size: int = 1024,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_targets: Tuple[str, ...] = ("qkv", "proj"),
        freeze_base: bool = True,
    ):
        super().__init__()
        assert sam_type in sam_model_registry, f"Unknown sam_type={sam_type}. Available: {list(sam_model_registry.keys())}"
        if not sam_checkpoint:
            raise ValueError("sam_checkpoint must be provided (e.g., sam_vit_h_4b8939.pth)")

        sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
        self.image_encoder: nn.Module = sam.image_encoder
        self.img_size = img_size

        replaced = apply_lora_to_linear_layers(
            self.image_encoder,
            target_keywords=lora_targets,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        # Freeze base weights (recommended for LoRA)
        if freeze_base:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            # Unfreeze LoRA params
            for n, p in self.image_encoder.named_parameters():
                if n.endswith(".A") or n.endswith(".B"):
                    p.requires_grad = True

        # Store how many layers were patched (useful for debugging/logging)
        self.num_lora_layers = replaced

    def forward(self, x: torch.Tensor):
        # SAM expects input already normalized to its pixel_mean/std and padded to square (1024).
        # MMDet data_preprocessor should handle normalization; padding is handled by pad_size_divisor,
        # but make sure your pipeline produces 1024x1024 or padded appropriately.
        emb = self.image_encoder(x)  # [B, 256, H/16, W/16]
        # Convert to channel-last to match the project's hidden-state format
        emb = emb.permute(0, 2, 3, 1).contiguous()  # [B, H/16, W/16, 256]
        return (emb,)

