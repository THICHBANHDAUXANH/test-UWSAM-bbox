_base_ = ['anchor_net.py']

# Train UWSAM model on UIIS10K (10 classes) using SAM ViT-B + LoRA adapters.
# This uses the native segment_anything checkpoint format (.pth).

work_dir = './work_dirs/UIIS10KDataset/vit_b_lora'

## ---------------------- MODEL ----------------------

crop_size = (1024, 1024)
num_classes = 10

# SAM ViT-B checkpoint (native format for backbone)
sam_checkpoint = '/home/anhbd/PROJECT CV OLP 2025/checkpoints/sam_vit_b_01ec64.pth'

# HuggingFace SAM ViT-B (for mask decoder config and shared embedding)
sam_hf_pretrain_name = '/home/anhbd/PROJECT CV OLP 2025/test-UWSAM-bbox/pretrain/sam-vit-base'
sam_hf_ckpt_path = '/home/anhbd/PROJECT CV OLP 2025/test-UWSAM-bbox/pretrain/sam-vit-base/pytorch_model.bin'

# Override model for ViT-B
model = dict(
    decoder_freeze=True,
    shared_image_embedding=dict(
        _delete_=True,  # Clear inherited config from base
        type='USISSamPositionalEmbedding',
        hf_pretrain_name=sam_hf_pretrain_name,
    ),
    backbone=dict(
        _delete_=True,
        type='SAMImageEncoderLoRABackbone',
        sam_type='vit_b',  # ViT-Base
        sam_checkpoint=sam_checkpoint,
        img_size=1024,
        lora_r=4,  # LoRA rank (Sam_LoRA default)
        lora_layer=None,  # Apply to all transformer blocks
        freeze_base=True,
    ),
    # For ViT-B, hidden dim is 768 (not 1280 like ViT-H)
    # But the neck output is 256 after SAM's neck, so this should work
    adapter=None,
    neck=dict(
        type='USISFPN',
        feature_aggregator=None,
        feature_spliter=dict(
            type='USISSimpleFPNHead',
            backbone_channel=256, 
            in_channels=[64, 128, 256],
            out_channels=256,
            num_outs=3,
            norm_cfg=dict(type='LN2d', requires_grad=True)),
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(
            mask_decoder=dict(
                _delete_=True,  # Clear inherited config from base
                type='USISSamMaskDecoder',
                hf_pretrain_name=sam_hf_pretrain_name,
            ),
        ),
    ),
    train_cfg=dict(rcnn=dict(mask_size=crop_size)),
)

## ---------------------- Dataset (UIIS10K) ----------------------

dataset_type = 'CocoDataset'
data_root = '/home/anhbd/PROJECT CV OLP 2025/data/uiiis10k/UIIS10K/'

metainfo = dict(
    classes=(
        'fish',
        'reptiles',
        'arthropoda',
        'corals',
        'mollusk',
        'plants',
        'ruins',
        'garbage',
        'human',
        'robots',
    )
)

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(
        type='Pad',
        size=crop_size,
        pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0)),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/multiclass_train.json',
        data_prefix=dict(img='img/'),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/multiclass_test.json',
        data_prefix=dict(img='img/'),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    ann_file=data_root + 'annotations/multiclass_test.json',
    format_only=False,
)
test_evaluator = val_evaluator

## ---------------------- Optimizer & Training Loop ----------------------

max_epochs = 10
base_lr = 0.0001

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

find_unused_parameters = True

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[15, 21],
        gamma=0.1
    )
]

optim_wrapper = dict(
    type='OptimWrapper',  # Use standard wrapper instead of AMP to avoid dtype mismatch
    accumulative_counts=2,  # Gradient accumulation to simulate batch_size=2
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05,
    )
)

auto_scale_lr = dict(enable=False, base_batch_size=1)
