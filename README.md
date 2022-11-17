# DTTR: DETECTING TEXT WITH TRANSFORMERS

## Introduction

We use [MMOCR](https://github.com/open-mmlab/mmocr) to implement DTTR model. DTTR is a CNN-transformer hybrid text detection moder, achieving 0.5% H-mean improvements and 20.0% faster in inference speed than the SOTA model with a backbone of ResNet-50 on MMOCR. MMOCR is an open-source toolbox based on PyTorch and mmdetection for text detection, text recognition, and the corresponding downstream tasks including key information extraction. It is part of the [OpenMMLab](https://openmmlab.com/) project.

The main branch works with **PyTorch 1.6+**.

## Installation

MMOCR depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmocr.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/ahsdx/DTTR.git
cd DTTR
pip3 install -e .
```

## Get Started

Please see [Getting Started](https://mmocr.readthedocs.io/en/latest/getting_started.html) for the basic usage of MMOCR.

## Testing

### Demo

Download trained model [Baidu Drive](https://pan.baidu.com/s/1dDMcijm5PDxG2Pt392eLHQ)(download code: 3jd4):

```
dttr_r50dcnv2_icdar2015.pth
```

Then, place it in the models directory

Run the model inference with a single image. Here is an example:

```bash
python mmocr/utils/ocr.py demo/img_89.jpg --output demo/det_out.jpg --det DTTR_r50 --recog None --export demo/ --det_ckpt models/dttr_r50dcnv2_icdar2015.pth
```

The results can be find in `demo/det_out.jpg`.

### Evaluate the performance

Run the following command: 

```bash
python tools/test.py configs/textdet/dttr/dttr_r50dcnv2_cfpn_1200e_icdar2015.py models/dttr_r50dcnv2_icdar2015.pth --eval hmean-iou
```

The results should be as follows:

`{'0_hmean-iou:recall': 0.8343765045739047, '0_hmean-iou:precision': 0.8988589211618258, '0_hmean-iou:hmean': 0.86541822721598}`

