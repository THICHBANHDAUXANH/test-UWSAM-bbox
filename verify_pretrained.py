"""
Verification script to check if SAM pretrained weights are loaded correctly.
This script verifies:
1. Keys are loaded without mismatches
2. Weights are pretrained (not random init)
3. LoRA is applied correctly
4. Forward pass works
5. [NEW] Direct comparison with HuggingFace weights
6. [NEW] Missing/unexpected keys check
"""

import sys
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_pretrained_weights():
    """Verify that SAM pretrained weights are correctly loaded."""
    
    from mmdet.registry import MODELS
    from transformers import SamModel
    import project.our.our_model
    
    print("=" * 70)
    print("SAM Pretrained Weights Verification")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Load HuggingFace SAM (ground truth)
    # =========================================================================
    print("\n[Step 1] Loading HuggingFace SAM as ground truth...")
    hf_model = SamModel.from_pretrained('facebook/sam-vit-huge')
    hf_state_dict = hf_model.vision_encoder.state_dict()
    print(f"  HuggingFace vision encoder keys: {len(hf_state_dict)}")
    
    # =========================================================================
    # Step 2: Build our backbone
    # =========================================================================
    print("\n[Step 2] Building USISSamVisionEncoder...")
    backbone_cfg = dict(
        type='USISSamVisionEncoder',
        hf_pretrain_name='facebook/sam-vit-huge',
        extra_config=dict(image_size=1024, output_hidden_states=True),
        peft_config=dict(
            peft_type='LORA',
            r=16,
            target_modules=['qkv'],
            lora_alpha=32,
            lora_dropout=0.05,
            bias='none',
        ),
        init_cfg=dict(type='Pretrained', checkpoint=None),
    )
    backbone = MODELS.build(backbone_cfg)
    our_state_dict = backbone.state_dict()
    print(f"  Our backbone keys: {len(our_state_dict)}")
    
    # =========================================================================
    # Step 3: Direct weight comparison (Method A)
    # =========================================================================
    print("\n[Step 3] Direct weight comparison with HuggingFace...")
    print("  Comparing critical tensors (max_abs_diff should be ~0):")
    
    # Key mapping: HF key → Our key (with PEFT prefix)
    keys_to_compare = [
        ('pos_embed', 'vision_encoder.base_model.model.pos_embed'),
        ('patch_embed.projection.weight', 'vision_encoder.base_model.model.patch_embed.projection.weight'),
        ('patch_embed.projection.bias', 'vision_encoder.base_model.model.patch_embed.projection.bias'),
        ('layers.0.attn.qkv.weight', 'vision_encoder.base_model.model.layers.0.attn.qkv.base_layer.weight'),
        ('layers.0.attn.qkv.bias', 'vision_encoder.base_model.model.layers.0.attn.qkv.base_layer.bias'),
        ('layers.31.attn.qkv.weight', 'vision_encoder.base_model.model.layers.31.attn.qkv.base_layer.weight'),
        ('neck.conv1.weight', 'vision_encoder.base_model.model.neck.conv1.weight'),
    ]
    
    all_match = True
    for hf_key, our_key in keys_to_compare:
        if hf_key in hf_state_dict and our_key in our_state_dict:
            hf_tensor = hf_state_dict[hf_key]
            our_tensor = our_state_dict[our_key]
            
            if hf_tensor.shape == our_tensor.shape:
                max_diff = (hf_tensor - our_tensor).abs().max().item()
                status = "✓" if max_diff < 1e-6 else "✗"
                print(f"  {status} {hf_key}")
                print(f"      max_abs_diff: {max_diff:.2e}")
                if max_diff >= 1e-6:
                    all_match = False
            else:
                print(f"  ! {hf_key} - shape mismatch: HF {hf_tensor.shape} vs Ours {our_tensor.shape}")
                all_match = False
        else:
            if hf_key not in hf_state_dict:
                print(f"  ! {hf_key} - not found in HuggingFace")
            if our_key not in our_state_dict:
                print(f"  ! {our_key} - not found in our model")
    
    if all_match:
        print("\n  ✅ All compared weights EXACTLY MATCH HuggingFace!")
    else:
        print("\n  ⚠️ Some weights differ from HuggingFace!")
    
    # =========================================================================
    # Step 4: Missing/Unexpected keys check (Method B)
    # =========================================================================
    print("\n[Step 4] Checking missing/unexpected keys...")
    
    # Get keys from our model (excluding PEFT additions)
    our_base_keys = set()
    for key in our_state_dict.keys():
        # Remove PEFT prefix to get base key
        if 'base_model.model.' in key:
            base_key = key.split('base_model.model.')[1]
            # Remove .base_layer for LoRA wrapped layers
            base_key = base_key.replace('.base_layer.', '.')
            # Skip LoRA-specific keys
            if 'lora_' not in base_key:
                our_base_keys.add(base_key)
    
    hf_keys = set(hf_state_dict.keys())
    
    # Compare
    missing_in_ours = hf_keys - our_base_keys
    unexpected_in_ours = our_base_keys - hf_keys
    
    print(f"  HuggingFace keys: {len(hf_keys)}")
    print(f"  Our base keys: {len(our_base_keys)}")
    print(f"  Missing keys (in HF but not in ours): {len(missing_in_ours)}")
    print(f"  Unexpected keys (in ours but not in HF): {len(unexpected_in_ours)}")
    
    if missing_in_ours:
        print(f"\n  Missing keys (first 5):")
        for k in list(missing_in_ours)[:5]:
            print(f"    - {k}")
    
    if unexpected_in_ours:
        print(f"\n  Unexpected keys (first 5):")
        for k in list(unexpected_in_ours)[:5]:
            print(f"    - {k}")
    
    if not missing_in_ours and not unexpected_in_ours:
        print("  ✅ No missing or unexpected keys!")
    
    # =========================================================================
    # Step 5: LoRA verification
    # =========================================================================
    print("\n[Step 5] Verifying LoRA layers...")
    lora_a_keys = [k for k in our_state_dict.keys() if 'lora_A' in k]
    lora_b_keys = [k for k in our_state_dict.keys() if 'lora_B' in k]
    
    print(f"  LoRA_A keys: {len(lora_a_keys)}")
    print(f"  LoRA_B keys: {len(lora_b_keys)}")
    
    if lora_b_keys:
        lora_b_weight = our_state_dict[lora_b_keys[0]]
        if lora_b_weight.abs().max() < 1e-6:
            print("  ✓ LoRA_B initialized to zeros (correct)")
        
    # =========================================================================
    # Step 6: Forward pass
    # =========================================================================
    print("\n[Step 6] Running forward pass...")
    backbone.eval()
    random_input = torch.randn(1, 3, 1024, 1024)
    
    with torch.no_grad():
        output = backbone(random_input)
    
    print(f"  Input shape: {random_input.shape}")
    print(f"  Output shape: {output.last_hidden_state.shape}")
    print("  ✓ Forward pass completed successfully")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  • SAM ViT-H loaded from: facebook/sam-vit-huge")
    print(f"  • Weight comparison with HF: {'MATCH' if all_match else 'DIFFER'}")
    print(f"  • Missing keys: {len(missing_in_ours)}")
    print(f"  • Unexpected keys: {len(unexpected_in_ours)}")
    print(f"  • LoRA layers: {len(lora_a_keys)}")
    print(f"  • Forward pass: OK")
    print("=" * 70)
    
    return all_match and len(missing_in_ours) == 0


if __name__ == "__main__":
    try:
        success = verify_pretrained_weights()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
