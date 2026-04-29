# model settings  mobilenet_v4.py














model_cfg = dict(
    backbone=dict(
        type='MobileNetV4',
        #layers=[
        #     # LayerName, OutputSize, Kernel/Stride, Channels, Comments
        #     dict(type='Conv2D', kernel_size=3, stride=2, in_channels=3, out_channels=32),     # Conv2D-3x3x32
        #     dict(type='FusedIB', kernel_size=3, stride=2, in_channels=32, out_channels=64),   # FusedIB-3x3
        #     dict(type='ExtraDW', kernel_size=5, stride=2, in_channels=64, out_channels=128),  # ExtraDW-5x5
        #     dict(type='IB', kernel_size=3, stride=1, in_channels=128, out_channels=128, num=2), # IB-3x3 (2x)
        #     dict(type='Attention', in_channels=128, out_channels=128),                        # Attention Module
        #     dict(type='ConvNext', kernel_size=3, stride=1, in_channels=128, out_channels=384), # ConvNext-3x3-384
        #     dict(type='ExtraDW', kernel_size=3, stride=2, in_channels=384, out_channels=256),  # ExtraDW-3x3
        #     dict(type='IB', kernel_size=3, stride=1, in_channels=256, out_channels=256),       # IB-3x3
        # ],
        # avg_pool=dict(kernel_size=7, stride=1, in_channels=256),
        # conv1x1=dict(in_channels=256, out_channels=512),
        # global_avg_pool=True,
        # out_channels=512
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5))
)

# data normalization
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

# validation pipeline
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# data settings
data_cfg = dict(
    batch_size=32,
    num_workers=4,
    train=dict(
        pretrained_flag=True,
        pretrained_weights='',
        freeze_flag=False,
        freeze_layers=('backbone',),
        epoches=150,
    ),
    test=dict(
        ckpt='',
        metrics=['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options=dict(
            topk=(1, 5),
            thrs=None,
            average_mode='none'
        )
    )
)

# optimizer
optimizer_cfg = dict(
    type='AdamW',
    lr=0.0008,
    #momentum=0.9,
    weight_decay=0.0001
)

# learning rate schedule
lr_config = dict(type='CosineAnnealingLrUpdater', min_lr=1e-6)









# knowledge distillation config
kd_cfg = dict(
    enable=True,               # 是否开启KD
    teacher='densenet121',    # 
    teacher_weights=None,      # 若为None则加载ImageNet预训练
    alpha=0.5,                 # KD loss比例
    T=8.0,                     # 蒸馏温度
    freeze_teacher=True        # 是否冻结teacher参数
)






