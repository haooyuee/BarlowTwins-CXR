#   Use different random seeds to train to calculate Confidence Interval

#   CUDA_VISIBLE_DEVICES=1 nohup python tools/train.py Mytrain/config_vindr_resnet50_1217.py --seed 42 --deterministic > LOG_tmp.txt &
import os
os.environ['WANDB_DATA_DIR'] = os.getcwd() + "./wandb/"
os.environ['WANDB_CACHE_DIR'] = os.getcwd() + "./wandb/.cache/"
#"/data/home/haoyue/LOCALIZATION/mmdetection./wandb/.cache/"
Name = 'seed45'
Group = '_1217coslr_resnet50_nihpr_224'
_base_ = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
#checkpoint_file = 'resnet50_nih_1110.pth' 
checkpoint_file = 'resnet50_nih_1102.pth' 
#checkpoint_file = 'torchvision://resnet50'

model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
        #init_cfg = dict(type='Constant', layer=['Conv1d', 'Conv2d', 'Linear'], val=1)
        #init_cfg = None
        ),
    roi_head=dict(
        bbox_head=dict(num_classes=14)))

classes = ("Aortic_enlargement", "Atelectasis", 
               "Calcification", "Cardiomegaly", 
               "Consolidation", "ILD", "Infiltration", 
               "Lung_Opacity", "Nodule_Mass", "Other_lesion", 
               "Pleural_effusion", "Pleural_thickening", 
               "Pneumothorax", "Pulmonary_fibrosis")

dataset_type = 'CocoDataset'
data_root = '/data/public_data/vinBigData/train/'
annotation_root = '/data/home/haoyue/data/mycoco/vinbigdata-coco-1205/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(224, 224),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            #dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=32,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(
        type=dataset_type,
        img_prefix=data_root,
        classes=classes,
        ann_file=annotation_root + 'train_annotations_1205.json',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        img_prefix=data_root,
        classes=classes,
        ann_file=annotation_root + 'valid_annotations_1205.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root,
        classes=classes,
        ann_file=annotation_root + 'test_annotations_1205.json',
        pipeline=test_pipeline)
        )

#workflow = [('train', 1),('val', 1)]

work_dir = './Mytrain/'+ Group + '/' + Name

runner = dict(
    type='EpochBasedRunner', 
    max_epochs=50)
    

optimizer = dict(
    type='SGD',  # Optimizer: Stochastic Gradient Descent
    lr=0.005,  # Base learning rate
    momentum=0.9,  # SGD with momentum
    weight_decay=0.0001)  # Weight decay
optimizer_config = dict(grad_clip=None)  # Configuration for gradient clipping, set to None to disable

#lr_config = dict(
#    policy='step',  # Use multi-step learning rate strategy during training
#    warmup='linear',  # Use linear learning rate warmup
#    warmup_iters=500,  # End warmup at iteration 500
#    warmup_ratio=0.001,  # Coefficient for learning rate warmup
#    step=[50, 65],  # Learning rate decay at which epochs
#    gamma=0.2)  # Learning rate decay coefficient


lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',  # 使用余弦退火策略
    warmup='linear',           # 使用线性预热
    warmup_iters=500,          # 在第500次迭代结束预热
    warmup_ratio=0.001,        # 预热时的学习率为基础学习率的这个比例
    min_lr=0.0001                   # 学习率退火到的最小值
)

evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')

checkpoint_config = dict(interval=1, max_keep_ckpts=1)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={
                'project': 'mmdetection',
                'group': Group,
                'name': Name
             },
             interval=200,
             log_checkpoint=False,
             log_checkpoint_metadata=False,
             num_eval_images=0,
             commit = False)
    ])
