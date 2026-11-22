# Self-training CrossEarth (DINOv2 + Reins + Mask2Former) on Jaipur pseudo-labels.
# Starts from the WHU-Building-A2S checkpoint; labels come from the averaged A2S+S2A teacher
# (scripts/make_ce_pseudolabels.py), uncertain pixels = 255 (ignored). No Jaipur label is used.
# Place in CrossEarth/configs/. Override iterations / lr with --cfg-options.
_base_ = [
    "./_base_/default_runtime.py",
    "./_base_/models/CrossEarth_dinov2_mask2former.py",
]
HOME = "/home/23ucs715"
crop_size = (512, 512)
max_iters = 6000

model = dict(
    backbone=dict(init_cfg=dict(
        type="Pretrained", checkpoint=f"{HOME}/crossearth_work/CrossEarth/checkpoints/dinov2_converted.pth")),
    decode_head=dict(num_classes=2, loss_cls=dict(class_weight=[1.0, 1.0, 0.1])),
)
load_from = f"{HOME}/crossearth_work/CrossEarth/checkpoints/WHU-Building-A2S.pth"
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer")

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=(512, 512), ratio_range=(0.75, 1.5), keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.9),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
train_dataloader = dict(
    batch_size=2, num_workers=4, persistent_workers=True, pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(type="JaipurPseudoDataset", data_root=f"{HOME}/btp/data/jaipur_ce_pseudo",
                 data_prefix=dict(img_path="img", seg_map_path="lbl"), pipeline=train_pipeline),
)
val_dataloader = None
val_evaluator = None
val_cfg = None
test_dataloader = None
test_evaluator = None
test_cfg = None

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(type="AdamW", lr=2e-5, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={"norm": dict(decay_mult=0.0), "query_embed": embed_multi, "level_embed": embed_multi,
                     "learnable_tokens": embed_multi, "reins.scale": embed_multi},
        norm_decay_mult=0.0),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, by_epoch=False, begin=0, end=200),
    dict(type="PolyLR", eta_min=0, power=0.9, begin=200, end=max_iters, by_epoch=False),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters)
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=3000, max_keep_ckpts=1, save_last=True),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)
