'''
# -*- coding: utf-8 -*-
# @Project   : afa
# @File      : net_cam2d.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2023/2/16 11:06

# @Desciption: inherit from the model_attn_aff.py, return the feature map(_x1, _x2, _x3, _x4)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
from . import mix_transformer
import numpy as np



class CamNet(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None,):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/'+backbone+'.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict,)

        if pooling=="gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling=="gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes-1, kernel_size=1, bias=False)


    def get_param_groups(self):

        param_groups = [[], [], [], []] # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)

        return param_groups


    def forward(self, x, cam_only=False):
        # _attns
        # stage 1-2: (2,1,256,256)
        # stage 3-4: (2,2,256,256)
        # stage 5-6: (2,5,256,256)
        # stage 7-8: (2,8,1024,1024)

        # _x1: (B,64,128,128)
        # _x2: (B,128,64,64)
        # _x3: (B,320,32,32)
        # _x4: (B,512,32,32)
        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x


        cls_x4 = self.pooling(_x4,(1,1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes-1)

        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cls_x4, cam_s4 #,None#, attn_pred


        return cls_x4, _attns, _x4




if __name__=="__main__":

    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    cam_net = CamNet('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    cam_net._param_groups()
    dummy_input = torch.rand(2,3,512,512)
    cam_net(dummy_input)