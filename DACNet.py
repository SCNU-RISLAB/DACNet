import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher_crowd

import numpy as np
import time

import cv2
from functools import reduce


# the network frmawork of the regression branch
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 2)


# the network frmawork of the classification branch
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)  # [bz,c,w,h] -> [bz,w,h,c]

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)  # [bz, w*h*n_anchor_points, n_cls]


# generate the reference points in grid layout
def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points


# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points


# this class generate all reference points on all pyramid levels
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2 ** p, row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))



class FSBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low):
        self.transposed_conv = nn.ConvTranspose2d(in_channels=in_channels_high, out_channels=in_channels_low, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.ca_module = ChannelAttention(in_channels_high) #

    def forward(self, high_level_feature, low_level_feature):
        upsampled_high_level_feature = self.transposed_conv(high_level_feature)
        upsampled_high_level_feature = F.interpolate(upsampled_high_level_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=True)
        attention_weights = self.ca_module(upsampled_high_level_feature)
        low_level_feature_adjusted = low_level_feature[:, :high_level_feature.shape[1], :, :]
        filtered_low_level_feature = low_level_feature_adjusted * attention_weights
        output_feature = upsampled_high_level_feature + filtered_low_level_feature
        return output_feature





class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(Decoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]



class Decoder1(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=[256, 256, 256]):
        super(Decoder1, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        # self.P5_1 = nn.Conv2d(C5_size, C5_size+C5_size, kernel_size=1, stride=1, padding=0)
        # self.P5_2 = nn.Conv2d(C5_size+C5_size, C4_size, kernel_size=3, stride=1, padding=1)
        self.P5_1 = nn.Conv2d(C5_size, C5_size + C5_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(C5_size + C5_size, feature_size[0], kernel_size=3, stride=1, padding=1)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        # add P5 elementwise to C4
        # self.P4_1 = nn.Conv2d(C4_size+C4_size, C4_size, kernel_size=1, stride=1, padding=0)
        # self.P4_2 = nn.Conv2d(C4_size, C3_size, kernel_size=3, stride=1, padding=1)
        self.P4_1 = nn.Conv2d(C4_size + feature_size[0], C4_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(C4_size, feature_size[1], kernel_size=3, stride=1, padding=1)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        # add P4 elementwise to C3
        # self.P3_1 = nn.Conv2d(C3_size+C3_size, C3_size, kernel_size=1, stride=1, padding=0)
        # self.P3_2 = nn.Conv2d(C3_size, C3_size//2, kernel_size=3, stride=1, padding=1)
        self.P3_1 = nn.Conv2d(C3_size + feature_size[1], C3_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(C3_size, feature_size[2], kernel_size=3, stride=1, padding=1)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_x = self.P5_2(P5_x)
        P5_out = P5_x  # [512, 8, 8]
        P5_upsampled_x = self.P5_upsampled(P5_x)

        C4 = torch.cat([C4, P5_upsampled_x], 1)
        P4_x = self.P4_1(C4)
        P4_x = self.P4_2(P4_x)
        P4_out = P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)

        C3 = torch.cat([C3, P4_upsampled_x], 1)
        P3_x = self.P3_1(C3)
        P3_x = self.P3_2(P3_x)
        P3_out = P3_x

        return [P3_out, P4_out, P5_out]





class Decoder2(nn.Module):
    def __init__(self, feat_in_size=[64, 64, 64, 64], feature_size=[64, 64, 64, 64]):
        super(Decoder2, self).__init__()
        self.P5_1 = nn.Conv2d(feat_in_size[-1], feat_in_size[-1] + feat_in_size[-1], kernel_size=3, stride=1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(feat_in_size[-1] + feat_in_size[-1])
        self.act5_1 = nn.ReLU()
        self.P5_2 = nn.Conv2d(feat_in_size[-1] + feat_in_size[-1], feature_size[0], kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(feature_size[0])
        self.act5_2 = nn.ReLU()
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        #
        self.P4_1 = nn.Conv2d(feat_in_size[-2] + feature_size[0], feat_in_size[-2], kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(feat_in_size[-2])
        self.act4_1 = nn.ReLU()
        self.P4_2 = nn.Conv2d(feat_in_size[2], feature_size[1], kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(feature_size[1])
        self.act4_2 = nn.ReLU()
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        #
        self.P3_1 = nn.Conv2d(feat_in_size[-3] + feature_size[1], feat_in_size[-3], kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(feat_in_size[-3])
        self.act3_1 = nn.ReLU()
        self.P3_2 = nn.Conv2d(feat_in_size[-3], feature_size[2], kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(feature_size[2])
        self.act3_2 = nn.ReLU()
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        #
        self.P2_1 = nn.Conv2d(feat_in_size[-4] + feature_size[2], feat_in_size[-4], kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(feat_in_size[-4])
        self.act2_1 = nn.ReLU()
        self.P2_2 = nn.Conv2d(feat_in_size[-4], feature_size[3], kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(feature_size[3])
        self.act2_2 = nn.ReLU()

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_x = self.bn5_1(P5_x)
        P5_x = self.act5_1(P5_x)
        P5_x = self.P5_2(P5_x)
        P5_x = self.bn5_2(P5_x)
        P5_x = self.act5_2(P5_x)
        P5_out = P5_x  # 16x
        P5_upsampled_x = self.P5_upsampled(P5_x)


        C4 = torch.cat([C4, P5_upsampled_x], 1)
        P4_x = self.P4_1(C4)
        P4_x = self.bn4_1(P4_x)
        P4_x = self.act4_1(P4_x)
        P4_x = self.P4_2(P4_x)
        P4_x = self.bn4_2(P4_x)
        P4_x = self.act4_2(P4_x)
        P4_out = P4_x  # 8x
        P4_upsampled_x = self.P4_upsampled(P4_x)




        C3 = torch.cat([C3, P4_upsampled_x], 1)
        P3_x = self.P3_1(C3)
        P3_x = self.bn3_1(P3_x)
        P3_x = self.act3_1(P3_x)
        P3_x = self.P3_2(P3_x)
        P3_x = self.bn3_2(P3_x)
        P3_x = self.act3_2(P3_x)
        P3_out = P3_x  # 4x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        



        C2 = torch.cat([C2, P3_upsampled_x], 1)
        P2_x = self.P2_1(C2)
        P2_x = self.bn2_1(P2_x)
        P2_x = self.act2_1(P2_x)
        P2_x = self.P2_2(P2_x)
        P2_x = self.bn2_2(P2_x)
        P2_x = self.act2_2(P2_x)
        P2_out = P2_x  # 2x
             
        return [P2_out, P3_out, P4_out, P5_out]


    
  


  
 ##ACFM adative Fine-gine focus
class ACFM(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        '''

        super(ACFM,self).__init__()
        ##--------------------------ksize=1-------------------------------------##
        internal_neurons=out_channels
        self.fc1_1 = nn.Conv2d(in_channels=in_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2_1 = nn.Conv2d(in_channels=internal_neurons, out_channels=in_channels, kernel_size=1, stride=1, bias=True)

        ##---------------------------ksize=3-------------------------------------##
        self.fc1_3 = nn.Conv2d(in_channels=in_channels, out_channels=internal_neurons, kernel_size=3, padding=1, stride=1, bias=True)
        self.fc2_3 = nn.Conv2d(in_channels=internal_neurons, out_channels=in_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.input_channels = in_channels        
  
        
        d=max(in_channels//r,L)   # 计算从向量C降维到 向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
   
        ####split操作
        self.conv=nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))

        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        output=[input,input]

        ##k=1的权重
        x1 = F.adaptive_avg_pool2d(input, output_size=(1, 1))
        #print('x:', x.shape)
        y1 = self.fc1_3(x1)
        x1 = self.fc1_1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2_1(x1)
        x1 = torch.sigmoid(x1)
  
        x2 = F.adaptive_max_pool2d(input, output_size=(1, 1))
        y2 = self.fc1_3(x2)
        #print('x:', x.shape)
        x2 = self.fc1_1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2_1(x2)
        x2 = torch.sigmoid(x2)
        
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        
        ##k=3的权重
        y1 = F.relu(y1, inplace=True)
        y1 = self.fc2_3(y1)
        y1 = torch.sigmoid(y1)
 
        y2 = F.relu(y2, inplace=True)
        y2 = self.fc2_3(y2)
        y2 = torch.sigmoid(y2)
        
        y = y1 + y2
        y = y.view(-1, self.input_channels, 1, 1)
        
        attention_weight1 = x  # 第一个注意力权重张量  
        attention_weight2 = y  # 第二个注意力权重张量  
        # 使用torch.cat在第二个维度上拼接两个张量  
        a_b = torch.cat((attention_weight1, attention_weight2), dim=1) 
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]  
        
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块 [[batch_size,1,out_channels,1],[batch_size,1,out_channels,1]
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) #   [[batch_size,out_channels,1,1],[batch_size,out_channels,1,1]
        V=list(map(lambda x,y:x*y,output,a_b)) # 逐元素相乘[batch_size,out_channels,H,W] * [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]
        V=reduce(lambda x,y:x+y,V) # 逐元素相加  [batch_size,out_channels,H,W] + [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        return V    # [batch_size,out_channels,H,W]

    


class DACNet(nn.Module):
    def __init__(self, backbone, row=2, line=2, return_interm_layers=True):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        # the number of all anchor points
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)  # 256
        self.classification = ClassificationModel(num_features_in=256, \
                                                  num_classes=self.num_classes, \
                                                  num_anchor_points=num_anchor_points)

        self.anchor_points = AnchorPoints(pyramid_levels=[3, ], row=row, line=line)

        self.return_interm_layers = return_interm_layers
        if self.return_interm_layers:
            self.fpn = Decoder2(feat_in_size=[128, 256, 512, 512], feature_size=[64, 64, 64, 64])


        #---------------  ACFM---------
        self.ACFM_att_P1 = ACFM(64,64)
        self.ACFM_att_P2 = ACFM(64,64)
        self.ACFM_att_P3 = ACFM(64,64)
        self.ACFM_att_P4 = ACFM(64,64)
        

    def forward(self, samples: NestedTensor):
        # get the backbone features
        print('features:', samples.shape)
        features = self.backbone(samples)


        features_fpn = self.fpn([features[0], features[1], features[2], features[3]])
        # features_fpn = features

        batch_size = features[0].shape[0]
        print('batch_size:',batch_size)

        # ---------------------fea[0:3]--------------------------
        features_fpn[0] = F.adaptive_avg_pool2d(features_fpn[0], output_size=features_fpn[2].size()[2:])
        features_fpn[1] = F.adaptive_avg_pool2d(features_fpn[1], output_size=features_fpn[2].size()[2:])
        features_fpn[3] = F.upsample_nearest(features_fpn[3], size=features_fpn[2].size()[2:])

        # #--------------------ACFM——————————————-----
        features_att_fpn_P1 = self.ACFM_att_P1(features_fpn[0])
        features_att_fpn_P2 = self.ACFM_att_P2(features_fpn[1])
        features_att_fpn_P3 = self.ACFM_att_P3(features_fpn[2])
        features_att_fpn_P4 = self.ACFM_att_P4(features_fpn[3])


        att_map = [features_att_fpn_P1, features_att_fpn_P2, features_att_fpn_P3, features_att_fpn_P4]
        # concat
        feat_fuse_reg = torch.cat([features_att_fpn_P1, features_att_fpn_P2, features_att_fpn_P3, features_att_fpn_P4],
                                  1)  #
        feat_fuse_cls = torch.cat([features_att_fpn_P1, features_att_fpn_P2, features_att_fpn_P3, features_att_fpn_P4],
                                  1)  #


        feat_att_reg = feat_fuse_reg
        feat_att_cls = feat_fuse_cls

        regression = self.regression(feat_att_reg) * 100  # 8x
        classification = self.classification(feat_att_cls)
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        # decode the points as prediction
        output_coord = regression + anchor_points
        output_class = classification
        out = {'pred_logits': output_class, 'pred_points': output_coord, 'att_map': att_map}
        # out = {'pred_logits': output_class, 'pred_points': output_coord}
        return out






class SetCriterion_Crowd(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, aux_kwargs, radius):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            radius: auxiliary radius for local density estimation
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.radius = radius  # 新增的辅助半径参数

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        if 'loss_aux' in self.weight_dict:
            self.aux_mode = False
        else:
            self.aux_mode = True
            self.aux_number = aux_kwargs['AUX_NUMBER']
            self.aux_range = aux_kwargs['AUX_RANGE']
            self.aux_kwargs = aux_kwargs['AUX_kwargs']

    def compute_auxiliary_radius(self, points):
        """ Calculate the auxiliary radius, based on the coordinates of the input points.
        Parameters:
        points: The predicted point coordinates
        Returns:
        radius: This is the computed radius
        "" "
        # calculate the local radius using Euclidean distance
        计算辅助半径, 基于输入点的坐标.
        Parameters:
            points: 预测的点坐标
        Returns:
            radius: 计算得到的半径
        """
        # 使用欧几里得距离计算局部半径
        dists = torch.cdist(points, points)  # 计算所有点对之间的距离
        radius = (dists < self.radius).sum(dim=1)  # 计算在半径内的点数
        return radius

    def estimate_local_density(self, points):
        """ 根据辅助半径估计局部密度.
        Parameters:
            points: 预测的点坐标
        Returns:
            density: 局部密度估计
        Estimate the local density from the auxiliary radius.
        Parameters:
        points: The predicted point coordinates
        Returns:
        density: Local density estimation
        """
        radius_counts = self.compute_auxiliary_radius(points)
        density = radius_counts / (torch.pi * self.radius**2)  # 假设圆形区域
        return density

    def forward(self, outputs, targets, show=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points'], 'offset': outputs['offset']}
        indices1 = self.matcher(output1, targets)

        # 估计局部密度
        pred_points = outputs['pred_points']
        local_density = self.estimate_local_density(pred_points)  # 计算局部密度

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.weight_dict.keys():
            if loss == 'loss_ce':
                losses.update(self.loss_labels(output1, targets, indices1, num_boxes))
            elif loss == 'loss_points':
                losses.update(self.loss_points(output1, targets, indices1, num_boxes))
            elif loss == 'loss_aux':
                out_auxs = output1['aux']
                losses.update(self.loss_auxiliary(out_auxs, targets, show))
            else:
                raise KeyError('do you really want to compute {} loss?'.format(loss))

        print(losses)
        return losses







