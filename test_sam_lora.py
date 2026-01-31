"""
Test script to verify regular SAM VisionEncoder with LoRA from HuggingFace.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_regular_sam_lora():
    """Test regular SamVisionEncoder with LoRA"""
    print("=" * 60)
    print("Testing Regular HuggingFace SAM VisionEncoder with LoRA")
    print("=" * 60)
    
    from transformers import SamModel, SamConfig
    from peft import get_peft_config, get_peft_model
    
    model_name = "facebook/sam-vit-huge"
    
    # Step 1: Load SAM model from HuggingFace
    print(f"\n1. Loading SAM from HuggingFace: {model_name}")
    sam_model = SamModel.from_pretrained(model_name)
    vision_encoder = sam_model.vision_encoder
    print(f"   ✓ Loaded SamVisionEncoder")
    print(f"   Type: {type(vision_encoder).__name__}")
    
    # Step 2: Apply LoRA
    print(f"\n2. Applying LoRA to SamVisionEncoder...")
    peft_config = get_peft_config({
        "peft_type": "LORA",
        "r": 16,
        "target_modules": ["qkv"],
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "inference_mode": False,
    })
    lora_model = get_peft_model(vision_encoder, peft_config)
    print("   ✓ LoRA applied successfully!")
    lora_model.print_trainable_parameters()
    
    # Step 3: Forward pass with random data
    print(f"\n3. Testing forward pass with random data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    lora_model = lora_model.to(device)
    lora_model.eval()
    
    random_input = torch.randn(1, 3, 1024, 1024).to(device)
    print(f"   Input shape: {random_input.shape}")
    
    with torch.no_grad():
        output = lora_model(random_input)
    
    print(f"   ✓ Forward pass successful!")
    print(f"   Output type: {type(output).__name__}")
    if hasattr(output, 'last_hidden_state'):
        print(f"   last_hidden_state shape: {output.last_hidden_state.shape}")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        test_regular_sam_lora()
        exit(0)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
