
from __future__ import annotations
from typing import Optional, Tuple, List

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from mmdet.registry import MODELS

# segment_anything must be installed in the environment
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam

from safetensors import safe_open
from safetensors.torch import save_file


class _LoRA_qkv(nn.Module):
    """LoRA wrapper for SAM's qkv linear layer.
    
    In SAM it is implemented as:
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    
    This wrapper adds LoRA adapters to Q and V (not K).
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        r: int,
        alpha: int,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.r = int(r)
        self.alpha = int(alpha)
        self.scale = self.alpha / max(self.r, 1)

    def forward(self, x):
        qkv = self.qkv(x)  # B, H, W, 3*dim
        new_q = self.linear_b_q(self.linear_a_q(x)) * self.scale
        new_v = self.linear_b_v(self.linear_a_v(x)) * self.scale
        # Add LoRA outputs to Q and V portions
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        return qkv


@MODELS.register_module()
class SAMImageEncoderLoRABackbone(nn.Module):
    """SAM image encoder (ViT) as an MMDet backbone, with LoRA adapters.

    Uses the Sam_LoRA approach:
    - Apply LoRA only to Q and V in attention layers (not K, not proj).
    - Freeze base SAM weights; train only LoRA parameters.
    - LoRA weights initialized: A ~ Kaiming uniform, B = 0.

    Output:
    - A single feature map tensor [B, 256, H/16, W/16] in NCHW format.
      Returned as a tuple to match MMDet backbone conventions.
    """
    
    def __init__(
        self,
        sam_type: str = "vit_h",
        sam_checkpoint: str = "",
        img_size: int = 1024,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_layer: Optional[List[int]] = None,
        freeze_base: bool = True,
    ):
        super().__init__()
        assert sam_type in sam_model_registry, f"Unknown sam_type={sam_type}. Available: {list(sam_model_registry.keys())}"
        if not sam_checkpoint:
            raise ValueError("sam_checkpoint must be provided (e.g., sam_vit_h_4b8939.pth)")

        # Build SAM model and extract image encoder
        sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
        self.image_encoder: nn.Module = sam.image_encoder
        self.img_size = img_size
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        # Determine which layers to apply LoRA
        num_blocks = len(self.image_encoder.blocks)
        if lora_layer is not None:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(num_blocks))  # All layers by default

        # Storage for LoRA weights (for save/load utilities)
        self.w_As: List[nn.Linear] = []
        self.w_Bs: List[nn.Linear] = []

        # Freeze all base encoder parameters first
        if freeze_base:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # Apply LoRA surgery to attention blocks
        for t_layer_i, blk in enumerate(self.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            
            # Create LoRA matrices for Q
            w_a_linear_q = nn.Linear(dim, lora_r, bias=False)
            w_b_linear_q = nn.Linear(lora_r, dim, bias=False)
            
            # Create LoRA matrices for V
            w_a_linear_v = nn.Linear(dim, lora_r, bias=False)
            w_b_linear_v = nn.Linear(lora_r, dim, bias=False)
            
            # Store references for save/load
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            
            # Replace qkv with LoRA wrapper
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                r=lora_r,
                alpha=lora_alpha,
            )

        # Initialize LoRA weights
        self.reset_parameters()
        
        # Store count for debugging
        self.num_lora_layers = len(self.lora_layer)
    def reset_parameters(self) -> None:
        """Initialize LoRA weights: A with Kaiming uniform, B with zeros."""
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def save_lora_parameters(self, filename: str) -> None:
        """Save LoRA parameters to a file."""
        assert filename.endswith(".pth") or filename.endswith(".pt"), "Use .pth or .pt extension"

        num_layer = len(self.w_As)
        state_dict = {}
        for i in range(num_layer):
            state_dict[f"w_a_{i:03d}"] = self.w_As[i].weight
            state_dict[f"w_b_{i:03d}"] = self.w_Bs[i].weight if i < len(self.w_Bs) else None
        
        # Filter out None values
        state_dict = {k: v for k, v in state_dict.items() if v is not None}
        torch.save(state_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        """Load LoRA parameters from a file."""
        state_dict = torch.load(filename, map_location="cpu")
        
        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            if saved_key in state_dict:
                w_A_linear.weight = Parameter(state_dict[saved_key])

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            if saved_key in state_dict:
                w_B_linear.weight = Parameter(state_dict[saved_key])

    def load_state_dict(self, state_dict, strict: bool = False):
        """Override to allow loading with mismatched keys (e.g., LoRA vs non-LoRA checkpoints).
        
        By default uses strict=False to skip unexpected/missing keys gracefully.
        """
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward pass through SAM image encoder with LoRA.
        
        Args:
            x: Input tensor [B, 3, H, W], should be normalized and padded to 1024x1024.
        
        Returns:
            Tuple of embedding tensor [B, 256, H/16, W/16] in NCHW format.
        """
        # SAM image encoder returns [B, 256, H/16, W/16] (NCHW format)
        emb = self.image_encoder(x)
        return (emb,)

