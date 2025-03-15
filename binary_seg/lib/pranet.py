import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from lib.nn import SynchronizedBatchNorm2d
from torchvision.utils import save_image

from lib.pvtv2 import pvt_v2_b2
from .Res2Net_v1b import res2net50_v1b_26w_4s

BatchNorm2d = SynchronizedBatchNorm2d


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )
def show_tensor(tensor_list):
    falg=np.random.choice(range(10000000))
    for i, tensor in enumerate(tensor_list):
        array=np.array(tensor.detach().to('cpu'))
        np.savetxt('tensor_values'+str(i+falg)+'.txt', array, fmt='%0.6f')


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) # ReLU activation is not used here, so in the PraNet part, structures like ra4_conv2 need to be wrapped with F.relu() afterward.

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module): # A modified Receptive Field Block (RFB) to enhance the expressive capability of features.
    # Multi-scale features + fusion + residual.
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

    
class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel,num_class):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5_fg = nn.Conv2d(3*channel, num_class, 1) # V2 newly added, for foreground
        self.conv5_bg = nn.Conv2d(3*channel, num_class, 1) # V2 newly added, for background

        


    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x_fg = self.conv5_fg(x)
        x_bg = self.conv5_bg(x)

        return x_fg,x_bg



class PVT_PraNet_V2(nn.Module):
    # res2net based encoder decoder
    def __init__(self,channel=32,num_class=3,sem_downsample=1,use_softmax=True):
        super(PVT_PraNet_V2, self).__init__()
        self.idx=range(10)
        self.num_class=num_class
        self.sem_downsample=sem_downsample
        self.use_softmax=use_softmax

         # ------conv block to convert single channel to 3 channels------
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # ---- PVT_V2_B2 Backbone ----
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        # ---- Res2Net Backbone ----
        # self.backbone = res2net50_v1b_26w_4s(pretrained=True)

        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)
        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel,self.num_class)
        # ---- DSRA3 ----
        self.ra4_conv1 = BasicConv2d(512, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)

        self.ra4_conv5_fg = BasicConv2d(256, num_class, kernel_size=1) # V2 newly added, for foreground
        self.ra4_conv5_bg = BasicConv2d(256, num_class, kernel_size=1) # V2 newly added, for background
 
        # ---- DSRA2 ---- 
        self.ra3_conv1 = BasicConv2d(320, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.ra3_conv4_fg = BasicConv2d(64, num_class, kernel_size=3, padding=1) # V2 newly added, for foreground
        self.ra3_conv4_bg = BasicConv2d(64, num_class, kernel_size=3, padding=1) # V2 newly added, for background

        # ---- DSRA1 ----
        self.ra2_conv1 = BasicConv2d(128, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.ra2_conv4_fg = BasicConv2d(64, num_class, kernel_size=3, padding=1) # V2 newly added, for foreground
        self.ra2_conv4_bg = BasicConv2d(64, num_class, kernel_size=3, padding=1) # V2 newly added, for background

       
    def forward(self, x,segSize=None):
        if x.size()[1] == 1:
            x = self.conv(x)
        # ----Encoder stage1~4----
        x1,x2,x3,x4=self.backbone(x)    # Obtain feature maps from four different layers（H/4 W/4 64    H/8 W/8 128  H/16 W/416 320  H/32 W/32 512）
        
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32

        # Feature fusion + upsampling to the original image size, resulting in lateral_map_5, a coarse segmentation prediction map.
        ra5_feat_fg,ra5_feat_bg = self.agg1(x4_rfb, x3_rfb, x2_rfb) # _,_,H/8,W/8
        lateral_map_5_fg = F.interpolate(ra5_feat_fg, scale_factor=8/self.sem_downsample, mode='bilinear')    # 修改 NOTES: Sup-1 (bs, 3, 44, 44) -> (bs, 3, 352, 352)
        lateral_map_5_bg = F.interpolate(ra5_feat_bg, scale_factor=8/self.sem_downsample, mode='bilinear')

        # ---- DSRA3 ----
        crop_4_fg = F.interpolate(ra5_feat_fg, scale_factor=0.25, mode='bilinear') # _,_,H/32,W/32
        crop_4_bg = F.interpolate(ra5_feat_bg, scale_factor=0.25, mode='bilinear') # _,_,H/32,W/32
 
        x = self.ra4_conv1(x4)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))

        ra4_feat_fg = self.ra4_conv5_fg(x) # new   _,3,H/32,W/32
        ra4_feat_bg = self.ra4_conv5_bg(x) # new   _,3,H/32,W/32

        if self.use_softmax:
            ra4_feat_fg=ra4_feat_fg+ra4_feat_fg.mul(nn.functional.softmax(crop_4_fg-crop_4_bg,dim=1))
        else:
            ra4_feat_fg=ra4_feat_fg+ra4_feat_fg.mul(crop_4_fg-crop_4_bg)

        lateral_map_4_fg = F.interpolate(ra4_feat_fg, scale_factor=32/self.sem_downsample, mode='bilinear')
        lateral_map_4_bg=F.interpolate(ra4_feat_bg, scale_factor=32/self.sem_downsample, mode='bilinear')


        # ---- DSRA2 ----
        crop_3_fg = F.interpolate(ra4_feat_fg, scale_factor=2, mode='bilinear')
        crop_3_bg = F.interpolate(ra4_feat_bg, scale_factor=2, mode='bilinear')

        x = self.ra3_conv1(x3)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))

        ra3_feat_fg = self.ra3_conv4_fg(x)
        ra3_feat_bg = self.ra3_conv4_bg(x)

        if self.use_softmax:
            ra3_feat_fg=ra3_feat_fg+ra3_feat_fg.mul(nn.functional.softmax(crop_3_fg-crop_3_bg,dim=1))
        else:
            ra3_feat_fg=ra3_feat_fg+ra3_feat_fg.mul(crop_3_fg-crop_3_bg)

        lateral_map_3_fg = F.interpolate(ra3_feat_fg, scale_factor=16/self.sem_downsample, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)
        lateral_map_3_bg = F.interpolate(ra3_feat_bg, scale_factor=16/self.sem_downsample, mode='bilinear')

        # ---- DSRA1 ----
        crop_2_fg = F.interpolate(ra3_feat_fg, scale_factor=2, mode='bilinear')
        crop_2_bg = F.interpolate(ra3_feat_bg, scale_factor=2, mode='bilinear')
        
        x = self.ra2_conv1(x2)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))

        ra2_feat_fg = self.ra2_conv4_fg(x)
        ra2_feat_bg = self.ra2_conv4_bg(x)

        if self.use_softmax:
            ra2_feat_fg=ra2_feat_fg+ra2_feat_fg.mul(nn.functional.softmax(crop_2_fg-crop_2_bg,dim=1))
        else:
            ra2_feat_fg=ra2_feat_fg+ra2_feat_fg.mul(crop_2_fg-crop_2_bg)
   
        lateral_map_2_fg = F.interpolate(ra2_feat_fg, scale_factor=8/self.sem_downsample, mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        lateral_map_2_bg = F.interpolate(ra2_feat_bg, scale_factor=8/self.sem_downsample, mode='bilinear')

        return lateral_map_2_fg,lateral_map_3_fg,lateral_map_4_fg,lateral_map_5_fg,lateral_map_2_bg,lateral_map_3_bg,lateral_map_4_bg,lateral_map_5_bg

    
    
    
class PraNet_V2(nn.Module):
    # res2net based encoder decoder
    def __init__(self,channel=32,num_class=3,sem_downsample=1,use_softmax=True):
        super(PraNet_V2, self).__init__()
        self.idx=range(10)
        self.num_class=num_class
        self.sem_downsample=sem_downsample
        self.use_softmax=use_softmax

         # ------conv block to convert single channel to 3 channels------
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # ---- PVT_V2_B2 Backbone ----
        # self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        # path = './models/pvt_v2_b2.pth'
        # save_model = torch.load(path)
        # model_dict = self.backbone.state_dict()
        # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # self.backbone.load_state_dict(model_dict)
        
        # ---- Res2Net Backbone ----
        self.backbone = res2net50_v1b_26w_4s(pretrained=True)

        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel,self.num_class)
        # ---- DSRA3 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)

        self.ra4_conv5_fg = BasicConv2d(256, num_class, kernel_size=1) # V2 newly added, for foreground
        self.ra4_conv5_bg = BasicConv2d(256, num_class, kernel_size=1) # V2 newly added, for background
 
        # ---- DSRA2 ---- 
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.ra3_conv4_fg = BasicConv2d(64, num_class, kernel_size=3, padding=1) # V2 newly added, for foreground
        self.ra3_conv4_bg = BasicConv2d(64, num_class, kernel_size=3, padding=1) # V2 newly added, for background

        # ---- DSRA1 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.ra2_conv4_fg = BasicConv2d(64, num_class, kernel_size=3, padding=1) # V2 newly added, for foreground
        self.ra2_conv4_bg = BasicConv2d(64, num_class, kernel_size=3, padding=1) # V2 newly added, for background



    def forward(self, x,segSize=None):
        # ----Encoder stage1----
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)      # bs, 64, 88, 88
        x1 = self.backbone.layer1(x)      # bs, 256, 88, 88
        # ----Encoder stage2----
        x2 = self.backbone.layer2(x1)     # bs, 512, 44, 44
        # ----Encoder stage3----
        x3 = self.backbone.layer3(x2)     # bs, 1024, 22, 22
        # ----Encoder stage4----
        x4 = self.backbone.layer4(x3)     # bs, 2048, 11, 11

        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32

        # Feature fusion + upsampling to the original image size, resulting in lateral_map_5, a coarse segmentation prediction map.
        ra5_feat_fg,ra5_feat_bg = self.agg1(x4_rfb, x3_rfb, x2_rfb) # _,_,H/8,W/8
        lateral_map_5_fg = F.interpolate(ra5_feat_fg, scale_factor=8/self.sem_downsample, mode='bilinear')
        lateral_map_5_bg = F.interpolate(ra5_feat_bg, scale_factor=8/self.sem_downsample, mode='bilinear')

        # ---- DSRA3 ----
        crop_4_fg = F.interpolate(ra5_feat_fg, scale_factor=0.25, mode='bilinear') # _,_,H/32,W/32
        crop_4_bg = F.interpolate(ra5_feat_bg, scale_factor=0.25, mode='bilinear') # _,_,H/32,W/32
        

        x = self.ra4_conv1(x4)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))

        ra4_feat_fg = self.ra4_conv5_fg(x) # new   _,3,H/32,W/32
        ra4_feat_bg = self.ra4_conv5_bg(x) # new   _,3,H/32,W/32

        if self.use_softmax:
            ra4_feat_fg=ra4_feat_fg+ra4_feat_fg.mul(nn.functional.softmax(crop_4_fg-crop_4_bg,dim=1))
        else:
            ra4_feat_fg=ra4_feat_fg+ra4_feat_fg.mul(crop_4_fg-crop_4_bg)

        lateral_map_4_fg = F.interpolate(ra4_feat_fg, scale_factor=32/self.sem_downsample, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)
        lateral_map_4_bg=F.interpolate(ra4_feat_bg, scale_factor=32/self.sem_downsample, mode='bilinear')


        # ---- DSRA2 ----
        crop_3_fg = F.interpolate(ra4_feat_fg, scale_factor=2, mode='bilinear')
        crop_3_bg = F.interpolate(ra4_feat_bg, scale_factor=2, mode='bilinear')

        x = self.ra3_conv1(x3)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))

        ra3_feat_fg = self.ra3_conv4_fg(x)
        ra3_feat_bg = self.ra3_conv4_bg(x)

        if self.use_softmax:
            ra3_feat_fg=ra3_feat_fg+ra3_feat_fg.mul(nn.functional.softmax(crop_3_fg-crop_3_bg,dim=1))
            # ra3_feat_fg=ra3_feat_fg+crop_3_fg
        else:
            ra3_feat_fg=ra3_feat_fg+ra3_feat_fg.mul(crop_3_fg-crop_3_bg)
            # ra3_feat_fg=ra3_feat_fg+crop_3_fg

        lateral_map_3_fg = F.interpolate(ra3_feat_fg, scale_factor=16/self.sem_downsample, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)
        lateral_map_3_bg = F.interpolate(ra3_feat_bg, scale_factor=16/self.sem_downsample, mode='bilinear')


        # ---- DSRA1 ----
        crop_2_fg = F.interpolate(ra3_feat_fg, scale_factor=2, mode='bilinear')
        crop_2_bg = F.interpolate(ra3_feat_bg, scale_factor=2, mode='bilinear')

        x = self.ra2_conv1(x2)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))

        ra2_feat_fg = self.ra2_conv4_fg(x)
        ra2_feat_bg = self.ra2_conv4_bg(x)

        if self.use_softmax:
            ra2_feat_fg=ra2_feat_fg+ra2_feat_fg.mul(nn.functional.softmax(crop_2_fg-crop_2_bg,dim=1))
            # ra2_feat_fg=ra2_feat_fg+crop_2_fg
        else:
            ra2_feat_fg=ra2_feat_fg+ra2_feat_fg.mul(crop_2_fg-crop_2_bg)
            # ra2_feat_fg=ra2_feat_fg+crop_2_fg
   
        lateral_map_2_fg = F.interpolate(ra2_feat_fg, scale_factor=8/self.sem_downsample, mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        lateral_map_2_bg = F.interpolate(ra2_feat_bg, scale_factor=8/self.sem_downsample, mode='bilinear')

        return lateral_map_2_fg,lateral_map_3_fg,lateral_map_4_fg,lateral_map_5_fg,lateral_map_2_bg,lateral_map_3_bg,lateral_map_4_bg,lateral_map_5_bg


if __name__ == '__main__':
    ras = PVT_PraNet_V2().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)
