"""
Test script to demonstrate key mismatch between:
1. segment-anything GitHub checkpoint (sam_vit_h_4b8939.pth)
2. Our HuggingFace-based model

This will download the checkpoint and show the key differences.
"""

import sys
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def download_segment_anything_checkpoint():
    """Download SAM ViT-H checkpoint from segment-anything GitHub."""
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    save_path = "sam_vit_h_4b8939.pth"
    
    if os.path.exists(save_path):
        print(f"  Checkpoint already exists: {save_path}")
        return save_path
    
    print(f"  Downloading from: {url}")
    print(f"  This may take a while (~2.4GB)...")
    
    urllib.request.urlretrieve(url, save_path)
    print(f"  Downloaded to: {save_path}")
    return save_path


def test_mismatch():
    """Test key mismatch between segment-anything and HuggingFace."""
    
    print("=" * 70)
    print("KEY MISMATCH TEST: segment-anything vs HuggingFace")
    print("=" * 70)
    
    # Step 1: Download segment-anything checkpoint
    print("\n[Step 1] Downloading segment-anything checkpoint...")
    ckpt_path = download_segment_anything_checkpoint()
    
    # Step 2: Load segment-anything checkpoint
    print("\n[Step 2] Loading segment-anything checkpoint...")
    sa_state_dict = torch.load(ckpt_path, map_location='cpu')
    print(f"  Total keys: {len(sa_state_dict)}")
    
    # Step 3: Get image_encoder keys
    sa_image_encoder_keys = [k for k in sa_state_dict.keys() if k.startswith('image_encoder')]
    print(f"  Image encoder keys: {len(sa_image_encoder_keys)}")
    print("\n  Sample segment-anything keys:")
    for k in sa_image_encoder_keys[:10]:
        print(f"    - {k}")
    
    # Step 4: Load HuggingFace SAM
    print("\n[Step 3] Loading HuggingFace SAM...")
    from transformers import SamModel
    hf_model = SamModel.from_pretrained('facebook/sam-vit-huge')
    hf_state_dict = hf_model.vision_encoder.state_dict()
    print(f"  HuggingFace vision encoder keys: {len(hf_state_dict)}")
    print("\n  Sample HuggingFace keys:")
    for k in list(hf_state_dict.keys())[:10]:
        print(f"    - {k}")
    
    # Step 5: Compare key formats
    print("\n[Step 4] KEY FORMAT COMPARISON:")
    print("=" * 70)
    
    # Remove 'image_encoder.' prefix from segment-anything keys
    sa_keys_stripped = set()
    for k in sa_image_encoder_keys:
        stripped = k.replace('image_encoder.', '')
        sa_keys_stripped.add(stripped)
    
    hf_keys = set(hf_state_dict.keys())
    
    # Find differences
    common = sa_keys_stripped & hf_keys
    only_in_sa = sa_keys_stripped - hf_keys
    only_in_hf = hf_keys - sa_keys_stripped
    
    print(f"\n  Common keys: {len(common)}")
    print(f"  Only in segment-anything: {len(only_in_sa)}")
    print(f"  Only in HuggingFace: {len(only_in_hf)}")
    
    if only_in_sa:
        print(f"\n  Keys ONLY in segment-anything (first 20):")
        for k in list(only_in_sa)[:20]:
            print(f"    - {k}")
    
    if only_in_hf:
        print(f"\n  Keys ONLY in HuggingFace (first 20):")
        for k in list(only_in_hf)[:20]:
            print(f"    - {k}")
    
    # Step 6: Show key name differences
    print("\n[Step 5] KEY NAMING DIFFERENCES:")
    print("=" * 70)
    print("""
  segment-anything format     ->  HuggingFace format
  -------------------------       ------------------
  blocks.0.norm1.weight       ->  layers.0.layer_norm1.weight
  blocks.0.attn.qkv.weight    ->  layers.0.attn.qkv.weight
  patch_embed.proj.weight     ->  patch_embed.projection.weight
  neck.0.weight               ->  neck.conv1.weight
    """)
    
    # Step 7: Try to load segment-anything into HuggingFace model
    print("\n[Step 6] TESTING: Load segment-anything into HuggingFace model...")
    print("=" * 70)
    
    # Create a new HuggingFace vision encoder 
    from transformers import SamConfig
    from transformers.models.sam.modeling_sam import SamVisionEncoder
    
    config = SamConfig.from_pretrained('facebook/sam-vit-huge').vision_config
    hf_encoder = SamVisionEncoder(config)
    
    # Extract image_encoder state dict from segment-anything
    sa_image_encoder_state = {}
    for k, v in sa_state_dict.items():
        if k.startswith('image_encoder.'):
            new_key = k.replace('image_encoder.', '')
            sa_image_encoder_state[new_key] = v
    
    # Try to load
    missing_keys, unexpected_keys = hf_encoder.load_state_dict(sa_image_encoder_state, strict=False)
    
    print(f"\n  Missing keys (in HF model but not in checkpoint): {len(missing_keys)}")
    print(f"  Unexpected keys (in checkpoint but not in HF model): {len(unexpected_keys)}")
    
    if missing_keys:
        print(f"\n  Missing keys (first 20):")
        for k in missing_keys[:20]:
            print(f"    - {k}")
    
    if unexpected_keys:
        print(f"\n  Unexpected keys (first 20):")
        for k in unexpected_keys[:20]:
            print(f"    - {k}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print("""
  RESULT: MISMATCH DETECTED!
  
  segment-anything checkpoint CANNOT be directly loaded into HuggingFace model
  because the key names are different.
  
  This is why we use HuggingFace's from_pretrained() instead of loading
  the checkpoint manually - HuggingFace handles the key conversion.
        """)
    else:
        print("  All keys match!")
    
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_mismatch()
        exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
