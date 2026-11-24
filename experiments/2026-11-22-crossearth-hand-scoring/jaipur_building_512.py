# Inference-only config: CrossEarth DINOv2 + Mask2Former, 2 classes (background, building),
# for the WHU-Building checkpoints. Absolute path to the DINOv2 weights so any cwd works.
_base_ = [
    "./_base_/default_runtime.py",
    "./_base_/models/CrossEarth_dinov2_mask2former.py",
]
model = dict(
    backbone=dict(init_cfg=dict(type="Pretrained",
        checkpoint="/home/23ucs715/crossearth_work/CrossEarth/checkpoints/dinov2_converted.pth")),
    decode_head=dict(num_classes=2, loss_cls=dict(class_weight=[1.0, 1.0, 0.1])),
)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    dict(type="PackSegInputs"),
]
