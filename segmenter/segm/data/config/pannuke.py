import LoadImageFromNpy

# dataset settings
dataset_type = "PanNukeDataset"
data_root = "/home/labs/testing/class54/project_data"

# NEEDS TO BE CHANGED
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# NEEDS TO BE CHANGED

crop_size = (256, 256)
max_ratio = 4
train_pipeline = [
    dict(type="LoadImageFromNpy"),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    # dict(type="Resize", img_scale=(512 * max_ratio, 512), ratio_range=(0.5, 2.0)),
    # dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type="RandomFlip", prob=0.5),
    # dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    # dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
val_pipeline = [
    dict(type="LoadImageFromNpy"),
    # dict(
    #     type="MultiScaleFlipAug",
    #     img_scale=(512 * max_ratio, 512),
    #     flip=False,
    #     transforms=[
    #         dict(type="Resize", keep_ratio=True),
    #         dict(type="RandomFlip"),
    #         dict(type="Normalize", **img_norm_cfg),
    #         dict(type="ImageToTensor", keys=["img"]),
    #         dict(type="Collect", keys=["img"]),
    #     ],
    # ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"])
]
test_pipeline = [
    dict(type="LoadImageFromNpy"),
    # dict(
    #     type="MultiScaleFlipAug",
    #     img_scale=(512 * max_ratio, 512),
    #     flip=False,
    #     transforms=[
    #         dict(type="Resize", keep_ratio=True),
    #         dict(type="RandomFlip"),
    #         dict(type="Normalize", **img_norm_cfg),
    #         dict(type="ImageToTensor", keys=["img"]),
    #         dict(type="Collect", keys=["img"]),
    #     ],
    # ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="Fold_1/images",
        ann_dir="Fold_1/masks",
        pipeline=train_pipeline,
    ),
    trainval=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=["Fold_1/images", "Fold_2/images"],
        ann_dir=["Fold_1/masks", "Fold_2/masks"],
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="Fold_2/images",
        ann_dir="Fold_2/masks",
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="Fold_3/images",
        ann_dir="Fold_3/masks",
        pipeline=test_pipeline,
    ),
)
