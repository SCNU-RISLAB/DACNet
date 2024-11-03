# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

import models.vgg_ as models
from models.cycle_mlp import *

model_path = ['models/pretrained_models/CycleMLP_B1.pth',
              'models/pretrained_models/CycleMLP_B2.pth',
              'models/pretrained_models/CycleMLP_B3.pth']

class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list):
        out = []

        if self.return_interm_layers:
            xs = tensor_list
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                out.append(xs)

        else:
            xs = self.body(tensor_list)
            out.append(xs)
        return out


class Backbone_VGG(BackboneBase_VGG):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, return_interm_layers: bool):
        if name == 'vgg16_bn':
            backbone = models.vgg16_bn(pretrained=True)
        elif name == 'vgg16':
            backbone = models.vgg16(pretrained=True)
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


def Backbone_CycleMLP(name: str, return_interm_layers: bool):
    """CycleMLP backbone."""
    if name == 'CycleMLP_B1_feat':
        backbone = CycleMLP_B1_feat()
        save_model = torch.load(model_path[0])
        model_dict = backbone.state_dict()
        state_dict = {k: v for k, v in save_model['model'].items() if k in model_dict.keys()}
        print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        model_dict.update(state_dict)
    elif name == 'CycleMLP_B2_feat':
        backbone = CycleMLP_B2_feat()
        save_model = torch.load(model_path[1])
        model_dict = backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        model_dict.update(state_dict)
    elif name == 'CycleMLP_B3_feat':
        backbone = CycleMLP_B3_feat()
        save_model = torch.load(model_path[2])
        model_dict = backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        model_dict.update(state_dict)
    return backbone



def build_backbone(args):
    if args.backbone in ['vgg16_bn' , 'vgg16']:
        backbone = Backbone_VGG(args.backbone, True)
    elif args.backbone in  ['CycleMLP_B1_feat', 'CycleMLP_B2_feat', 'CycleMLP_B3_feat', 'CycleMLP_B4_feat', 'CycleMLP_B5_feat']:
        backbone = Backbone_CycleMLP(args.backbone, False)
    return backbone

if __name__ == '__main__':
    Backbone_VGG('vgg16', True)