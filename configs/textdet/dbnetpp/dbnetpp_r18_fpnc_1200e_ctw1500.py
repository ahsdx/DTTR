_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_sgd_1200e.py',
    '../../_base_/det_models/dbnetpp_r18_fpnc.py',
    '../../_base_/det_datasets/ctw1500.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r18 = {{_base_.train_pipeline_r18}}
test_pipeline_1333_736 = {{_base_.test_pipeline_1333_736}}
# 下次要在跑一次这个，ctw1500的是1024的
# test_pipeline_4068_1024 = {{_base_.test_pipeline_1024_1024}}

optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001)
total_epochs = 1200
checkpoint_config = dict(interval=100)
# load_from = 'checkpoints/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_r18),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_1333_736),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_1333_736))

evaluation = dict(
    interval=100,
    metric='hmean-iou',
    save_best='0_hmean-iou:hmean',
    rule='greater')
