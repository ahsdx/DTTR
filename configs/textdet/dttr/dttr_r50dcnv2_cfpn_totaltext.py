_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_sgd_1200e.py',
    '../../_base_/det_models/dttr_r50dcnv2_cfpn.py',
    '../../_base_/det_datasets/totaltext.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]


optimizer = dict(type='SGD', lr=0.007 / 2, momentum=0.9, weight_decay=0.0001)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=1800)
checkpoint_config = dict(interval=50)


train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
test_pipeline_800_800 = {{_base_.test_pipeline_800_800}}

load_from = 'experiments/dttr_r50_synthtext/iter_100000.pth'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_r50dcnv2),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_800_800),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_800_800))

evaluation = dict(
    interval=50,
    metric='hmean-iou',
    save_best='0_hmean-iou:hmean',
    rule='greater')
