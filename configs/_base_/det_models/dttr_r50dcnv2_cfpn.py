model = dict(
    type='DTTR',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='CFPN', in_channels=[256, 512, 1024, 2048], lateral_channels=256),
    bbox_head=dict(
        type='TdbHead',
        in_channels=2048,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=True),
        # postprocessor=dict(type='DBPostprocessor', text_repr_type='quad', unclip_ratio=1.7, mask_thr=0.3)),
        postprocessor=dict(type='DBPostprocessor', text_repr_type='poly', unclip_ratio=1.9, mask_thr=0.3, epsilon_ratio=0.002)),
    train_cfg=None,
    test_cfg=None)
