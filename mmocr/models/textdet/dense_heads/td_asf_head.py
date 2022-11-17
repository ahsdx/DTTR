# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmcv.cnn.utils import weight_init

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.runner import BaseModule, Sequential
from mmcv.cnn import Conv2d, Linear, build_activation_layer

from mmocr.models.builder import HEADS
from mmocr.models.textdet.utils.visualize import visualize_feature_map_sum
from .head_mixin import HeadMixin
from mmocr.models.textdet.utils import (FFN, build_positional_encoding,
                                build_transformer)
from mmocr.models.textdet.utils import visual, visualize_feature_map_sum


@HEADS.register_module()
class TD_ASF_Head(HeadMixin, BaseModule):
    """The class for DBNet head.

    This was partially adapted from https://github.com/MhLiao/DB

    Args:
        in_channels (int): The number of input channels of the db head.
        with_bias (bool): Whether add bias in Conv2d layer.
        downsample_ratio (float): The downsample ratio of ground truths.
        loss (dict): Config of loss for dbnet.
        postprocessor (dict): Config of postprocessor for dbnet.
    """

    def __init__(
            self,
            in_channels,
            with_bias=False,
            downsample_ratio=1.0,
            loss=dict(type='DBLoss'),
            postprocessor=dict(type='DBPostprocessor', text_repr_type='quad'),
            init_cfg=[
                dict(type='Kaiming', layer='Conv'),
                dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4)
            ],
            transformer=dict(
                     type='Transformer',
                     embed_dims=256,
                     num_heads=8,
                     num_encoder_layers=0,
                     num_decoder_layers=6,
                     feedforward_channels=2048,
                     dropout=0.1,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN'),
                     num_fcs=2,
                     pre_norm=False,
                     return_intermediate_dec=False),
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=128,
                normalize=True),
            train_cfg=None,
            test_cfg=None,
            **kwargs):
        old_keys = ['text_repr_type', 'decoding_type']
        for key in old_keys:
            if kwargs.get(key, None):
                postprocessor[key] = kwargs.get(key)
                warnings.warn(
                    f'{key} is deprecated, please specify '
                    'it in postprocessor config dict. See '
                    'https://github.com/open-mmlab/mmocr/pull/640'
                    ' for details.', UserWarning)
        BaseModule.__init__(self, init_cfg=init_cfg)
        HeadMixin.__init__(self, loss, postprocessor)

        assert isinstance(in_channels, int)

        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.downsample_ratio = downsample_ratio

        self.embed_dims = transformer['embed_dims']

        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)

        # self.query_embedding = nn.Embedding(2 + 4, self.embed_dims)
        self.query_embedding = nn.Embedding(32, self.embed_dims)
        # self.query_embedding = nn.Embedding(32 + 4, self.embed_dims)
        # 前面32个query表示probability query set(16)和threshold query set(16)，后面4个query表示自适应权重融合的query
        # self.mask_embed = MLP(self.embed_dims, self.embed_dims, self.embed_dims, 1) # MaskFormer那篇论文是3层MLP，集装箱
        self.mask_embed = MLP(self.embed_dims, self.embed_dims, self.embed_dims, 3)

        # self.Upsample = Sequential(
        #     nn.ConvTranspose2d(2, 2, 2, 2),
        #     nn.BatchNorm2d(2), nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(2, 2, 2, 2)
        # )
        self.Upsample = Sequential(
            nn.ConvTranspose2d(32, 8, 2, 2),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 2, 2, 2)
        )

    def diff_binarize(self, prob_map, thr_map, k):
        return torch.reciprocal(1.0 + torch.exp(-k * (prob_map - thr_map)))

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): Shape (batch_size, hidden_size, h, w).

        Returns:
            Tensor: A tensor of the same shape as input.
        """
        # print(inputs[-1][0].shape)
        shareFeats, x = inputs # weight_map[0].shape = [64, H/4, W/4]. len(weight_map) = 4
        # print(shareFeats.shape, x.shape)
        masks = None
        if masks is None:
             # 以后设置为one试一试，paddle里面设置的就是为one，而不是zero。我觉得可能会学习吧
            masks = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        query_embedding = self.query_embedding.weight
        x = self.input_proj(x)
        # print(pos_embed.shape, query_embedding.shape)

        # outs_dec: [nb_dec, bs, num_query, embed_dim] (num_query=2 ;nb_dec为层数，类似于多阶段输出，这里应该为1)
        outs_dec, _ = self.transformer(x, masks, query_embedding, pos_embed)
        mask_embed = self.mask_embed(outs_dec.squeeze(0))

        # weight_embed = mask_embed[:, -4:, :]
        # pt_embed = mask_embed[:, :-4, :] # (pt denotes probability and threshold)
        pt_embed = mask_embed

        # weights = torch.einsum("bqc,bchw->bqhw", weight_embed, shareFeats)
        # visualize_feature_map_sum(weight_maps)
        # visual(weight_maps)
        # for i, weight_map in enumerate(weight_maps):
        #     weight_maps[i] = torch.mul(weight_map, nn.Sigmoid()(weights[:, i:(i+1), :, :]))
        # shareFeats = torch.cat(weight_maps, dim=1)
        # visual([shareFeats])
        # visualize_feature_map_sum([shareFeats])
        B, _, h, w = shareFeats.shape
        out = torch.einsum("bqc,bchw->bqhw", pt_embed, shareFeats)
        # out = mask_embed @ shareFeats.flatten(2)
        # out = out.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
        out = self.Upsample(out)
        # out = out.reshape(B, 2, h*4, w*4)
        # out = F.interpolate(out, size=(h*4, h*4), mode="bilinear", align_corners=False,)
        # print(out.shape)
        # prob_map = out[:, 0:1, :, :]
        # thr_map = out[:, 1:, :, :]
        prob_map = nn.Sigmoid()(out[:, 0:1, :, :])
        thr_map = nn.Sigmoid()(out[:, 1:, :, :])
        binary_map = self.diff_binarize(prob_map, thr_map, k=50)
        # visual([binary_map])
        # visual([prob_map, thr_map, binary_map])
        outputs = torch.cat((prob_map, thr_map, binary_map), dim=1)

        return outputs


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
