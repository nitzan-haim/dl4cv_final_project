
# dataset settings
dataset_type = "PannukeDataset"
data_root = "/home/labs/testing/class54/project_data"
# data_root = "/content/gdrive/MyDrive/DL4CV/final_project/data"

# NEEDS TO BE CHANGED
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)
# NEEDS TO BE CHANGED

crop_size = (256, 256)
max_ratio = 1 
train_pipeline = [
    dict(type="LoadImageFromNpy", to_float32=True),
    dict(type="LoadAnnotationsPannuke"),
    dict(type="ResizePannuke", img_scale=(256 * max_ratio, 256), ratio_range=(1,1)),
    dict(type="RandomCropPannuke", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlipPannuke", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="NormalizePannuke", **img_norm_cfg),
    dict(type="PadPannuke", size=crop_size, pad_val=0),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectPannuke", keys=["img", "gt_semantic_seg"]),
]
val_pipeline = [
    dict(type="LoadImageFromNpy",to_float32=True),
    dict(
         type="MultiScaleFlipAugPannuke",
         img_scale=(256 * max_ratio, 256),
         flip=False,
         transforms=[
             dict(type="ResizePannuke", keep_ratio=True),
             dict(type="RandomFlipPannuke"),
             dict(type="NormalizePannuke", **img_norm_cfg),
             dict(type="ImageToTensorPannuke", keys=["img"]),
             dict(type="CollectPannuke", keys=["img"]),
         ],
     ),
]
test_pipeline = [
    dict(type="LoadImageFromNpy",to_float32=True),
     dict(
         type="MultiScaleFlipAugPannuke",
         img_scale=(256 * max_ratio, 256),
         flip=False,
         transforms=[
             dict(type="ResizePannuke", keep_ratio=True),
             dict(type="RandomFlipPannuke"),
             dict(type="NormalizePannuke", **img_norm_cfg),
             dict(type="ImageToTensorPannuke", keys=["img"]),
             dict(type="CollectPannuke", keys=["img"]),
         ],
     ),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
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
