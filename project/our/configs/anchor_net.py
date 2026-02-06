## ---------------------- DEFAULT_SETTING ----------------------

default_scope = 'mmdet'
custom_imports = dict(imports=['project.our.our_model'], allow_failed_imports=False)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=3, max_keep_ckpts=1, save_best=['coco/bbox_mAP', 'coco/segm_mAP'],
                    rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

## ---------------------- MODEL ----------------------

crop_size = (1024, 1024)

batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
    batch_augments=batch_augments
)

num_classes = 10  # UIIS10K: fish, reptiles, arthropoda, corals, mollusk, plants, ruins, garbage, human, robots
pointset_point_num = 5  # per pointset point

# SAM ViT-Base model - using local pretrain checkpoint
sam_pretrain_name = "/home/duy-anh/projects/test-UWSAM-bbox/pretrain/sam-vit-base"
sam_pretrain_ckpt_path = "/home/duy-anh/projects/test-UWSAM-bbox/pretrain/sam-vit-base/pytorch_model.bin"

# model settings
model = dict(
    type='USISAnchor',
    data_preprocessor=data_preprocessor,
    decoder_freeze=True,
    shared_image_embedding=dict(
        type='USISSamPositionalEmbedding',
        hf_pretrain_name=sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=sam_pretrain_ckpt_path),
    ),
    backbone=dict(
        type='USISSamVisionEncoder',
        hf_pretrain_name=sam_pretrain_name,
        extra_config=dict(output_hidden_states=True),
        init_cfg=dict(type='Pretrained', checkpoint=sam_pretrain_ckpt_path),
        peft_config=dict(
            peft_type="LORA",
            r=16,
            target_modules=["qkv"],
            lora_alpha=32,
            lora_dropout=0.05,
        ),
    ),
    # Apply color adapter on SAM image embeddings (post SamVisionNeck, C=256)
    adapter=None,
    last_layer_adapter=dict(
        type='LastLayerColorAdapter',
        embed_dim=256,
        mlp_ratio=0.25,
    ),
    neck=dict(
        type='USISFPN',
        # Removed feature_aggregator - directly use 256-dim image embeddings
        feature_aggregator=None,
        feature_spliter=dict(
            type='USISSimpleFPNHead',
            backbone_channel=256,
            in_channels=[64, 128, 256, 256],
            out_channels=256,
            num_outs=5,
            norm_cfg=dict(type='LN2d', requires_grad=True)),
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            
            scales=[4, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='USISPrompterAnchorRoIPromptHead',
        with_extra_pe=True,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='USISPrompterAnchorMaskHead',
            mask_decoder=dict(
                type='USISSamMaskDecoder',
                hf_pretrain_name=sam_pretrain_name,
                init_cfg=dict(type='Pretrained', checkpoint=sam_pretrain_ckpt_path)),
            in_channels=256,
            roi_feat_size=14,
            per_pointset_point=pointset_point_num,
            with_sincos=True,
            multimask_output=False,
            class_agnostic=True,
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        )
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=crop_size,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)
    )
)

## ---------------------- DATASET ----------------------

data_root = '/home/duy-anh/projects/uiis10k-baselines/datasets/uiis10k/UIIS10K'

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

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='img/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='img/test/'),
        test_mode=True,
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}/annotations/test.json',
    metric=['bbox', 'segm'],
    format_only=False,
)

test_evaluator = val_evaluator

## ---------------------- TRAINING SCHEDULE ----------------------

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=5e-2),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=50),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[15, 21],
        gamma=0.1),
]

auto_scale_lr = dict(enable=False, base_batch_size=16)

