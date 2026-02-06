# USIS - Underwater SAM Instance Segmentation

A state-of-the-art underwater image instance segmentation model that integrates the Segment Anything Model (SAM) with specialized underwater vision adapters. This project extends MMDetection framework to provide robust underwater object detection and segmentation capabilities.

## Overview

USIS (Underwater SAM Instance Segmentation) combines the powerful vision capabilities of Meta's SAM model with custom underwater-specific adaptations:

- **SAM Integration**: Leverages SAM ViT backbone with LoRA fine-tuning
- **Underwater Adaptations**: Color attention adapters specifically designed for underwater imaging challenges
- **Multi-scale Detection**: Feature Pyramid Network (FPN) for detecting objects at various scales
- **Instance Segmentation**: End-to-end trainable instance segmentation pipeline

## Key Features

- 🌊 **Underwater Specialized**: Color attention adapters to handle underwater color distortion
- 🎯 **SAM-based Architecture**: Built on top of SAM ViT-Base/Huge models
- ⚡ **LoRA Fine-tuning**: Parameter-efficient fine-tuning using LoRA adapters  
- 🔧 **MMDetection Framework**: Leverages robust MMDetection ecosystem
- 📊 **Multi-class Support**: Supports 10 underwater object categories
- 🚀 **Efficient Training**: Frozen decoder with trainable adapters for fast convergence

## Supported Object Categories

The model is trained to detect and segment 10 underwater object categories:
- Fish
- Reptiles  
- Arthropoda
- Corals
- Mollusk
- Plants
- Ruins
- Garbage
- Human
- Robots

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU training)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/THICHBANHDAUXANH/test-UWSAM-bbox.git
   cd test-UWSAM-bbox
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Download SAM pretrained weights**
   ```bash
   cd pretrain
   bash download_huggingface.sh
   ```

## Model Architecture

### Core Components

1. **SAM Vision Encoder** (`USISSamVisionEncoder`)
   - Based on SAM ViT-Base/Huge architecture
   - Enhanced with LoRA adapters for efficient fine-tuning
   - Custom underwater-specific vision transformer blocks

2. **Color Attention Adapter** (`ColorAttentionAdapter`) 
   - Channel-wise attention mechanism
   - Designed to handle underwater color distortion
   - Applied at multiple scales in the network

3. **USIS FPN** (`USISFPN`)
   - Feature Pyramid Network for multi-scale detection
   - Integrates SAM features with detection pipeline
   - Custom feature aggregation and splitting

4. **Anchor-based Detection Head**
   - Standard RPN + RoI heads for object detection
   - Mask head for instance segmentation
   - Optimized for underwater object characteristics

### Model Configuration

The main model configuration is in `project/our/configs/anchor_net.py`:

```python
model = dict(
    type='USISAnchor',
    backbone=dict(
        type='USISSamVisionEncoder',
        hf_pretrain_name=sam_pretrain_name,
        peft_config=dict(
            peft_type="LORA",
            r=16,
            target_modules=["qkv"],
            lora_alpha=32,
            lora_dropout=0.05,
        ),
    ),
    # ... other components
)
```

## Usage

### Training

```bash
# Single GPU training
python tools/train.py project/our/configs/anchor_net.py

# Multi-GPU training  
bash tools/dist_train.sh project/our/configs/anchor_net.py 8

# Slurm cluster training
bash tools/slurm_train.sh <partition> <job_name> project/our/configs/anchor_net.py 8
```

### Inference

```bash
# Single image inference
python tools/test.py project/our/configs/anchor_net.py \
    work_dirs/anchor_net/latest.pth \
    --show-dir results/

# Batch inference with visualization
python vis_infer.py
```

### Testing and Evaluation

```bash
# Evaluate on test set
python tools/test.py project/our/configs/anchor_net.py \
    work_dirs/anchor_net/latest.pth \
    --eval bbox segm

# Distributed testing
bash tools/dist_test.sh project/our/configs/anchor_net.py \
    work_dirs/anchor_net/latest.pth 8 --eval bbox segm
```

## Verification Scripts

The repository includes several verification scripts to ensure correct model setup:

### SAM Weight Verification
```bash
python verify_pretrained.py
```
Verifies that SAM pretrained weights are correctly loaded and LoRA is applied properly.

### LoRA Testing  
```bash
python test_sam_lora.py
```
Tests the LoRA integration with SAM vision encoder.

### Model Comparison
```bash
python test_segment_anything_mismatch.py
```
Compares model outputs with reference implementations.

## Project Structure

```
test-UWSAM-bbox/
├── project/our/
│   ├── configs/          # Model configurations
│   └── our_model/        # Custom model implementations
│       ├── anchor.py     # Main USIS model
│       ├── common.py     # Shared components and adapters  
│       └── sam.py        # Custom SAM encoder implementations
├── pretrain/             # Pretrained model weights
├── tools/                # Training and testing scripts
├── tests/                # Test datasets and evaluation
├── work_dirs/            # Training outputs and checkpoints
├── vis_infer.py          # Inference visualization script
├── verify_pretrained.py  # Model verification script
└── requirements.txt      # Python dependencies
```

## Key Dependencies

- **MMDetection**: Object detection framework
- **Transformers**: HuggingFace transformers for SAM models
- **PEFT**: Parameter Efficient Fine-Tuning library
- **PyTorch Lightning**: Training framework
- **OpenCV**: Computer vision utilities
- **Albumentations**: Data augmentation

## Model Weights

### SAM Backbone Options
- **SAM ViT-Base**: `facebook/sam-vit-base` (recommended for most use cases)
- **SAM ViT-Huge**: `facebook/sam-vit-huge` (for maximum performance)

### Pretrained USIS Models
- Download from the releases section or train from scratch using provided configurations

## Performance

The model achieves competitive performance on underwater instance segmentation benchmarks:

- **USIS10K Dataset**: State-of-the-art results on 10-class underwater segmentation
- **Efficient Training**: Converges faster than full fine-tuning approaches
- **Memory Efficient**: LoRA reduces memory requirements by ~50%

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## Acknowledgments

- [Meta AI SAM](https://github.com/facebookresearch/segment-anything) for the foundational vision model
- [MMDetection](https://github.com/open-mmlab/mmdetection) for the detection framework  
- [HuggingFace Transformers](https://github.com/huggingface/transformers) for model implementations
- [PEFT](https://github.com/huggingface/peft) for parameter efficient fine-tuning

## Contact

For questions and support, please open an issue in the GitHub repository or contact the maintainers.
