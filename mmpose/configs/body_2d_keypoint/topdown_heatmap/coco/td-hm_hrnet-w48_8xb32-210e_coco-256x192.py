_base_ = ['../../../_base_/default_runtime.py']

# runtime
#train_cfg = dict(max_epochs=210, val_interval=10)
train_cfg = dict(max_epochs=120, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,  # learning rate
))


# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
                       'pretrain_models/hrnet_w48-8ef0771d.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=48,
        out_channels=17,  # TODO: Update to match number of keypoints (should be 17)
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'TinyCocoDataset'
data_mode = 'topdown'
data_root = r'D:\TU\7_Semester\Bachelorarbeit\code\Pose-Estimation-ToF\training'

#data_root = 'D:/TU/7_Semester/Bachelorarbeit/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/coco_tiny/'  # TODO: update to dataset root directory
#data_root = 'D:/TU/7_Semester/test/'

work_dir = r'D:\TU\7_Semester\Bachelorarbeit\mmpose\work_dirs\td-hm_hrnet'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=32,  # TODO: Adjust based on your dataset size and available GPU memory
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='train.json',  # TODO: Update with train annotations
        data_prefix=dict(img='images/'),  # TODO: Update with training image directory
        pipeline=train_pipeline,
    ))

test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='test.json',
        data_prefix=dict(img='images/'),  # Testing image directory
        test_mode=True,
        pipeline=test_pipeline,
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='val.json',  # TODO: Update with validation annotations
        data_prefix=dict(img='images/'),  # TODO: Update with validation image directory
        test_mode=True,
        pipeline=val_pipeline,
    )
)

# evaluators
val_evaluator = dict(
    type='PCKAccuracy')

test_evaluator = dict(
    type='PCKAccuracy',
)

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='PCK', rule='greater'),
    visualization=dict(
        type='PoseVisualizationHook',
        show=False,  # Set to True to show the visualizations interactively
        out_dir=r'D:\TU\7_Semester\Bachelorarbeit\mmpose\test_results'  # Output directory for visualized images
    )
)
