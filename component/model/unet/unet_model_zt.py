# full assembly of the sub-parts to form the complete net
# This vision was modified by xiaooru.zheng in 2021.06.04

import torch
import torch.nn as nn
from .unet_parts import double_down_max,unet_up

class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        # Input：256*256*in_ch
        #        256*256*out_ch*2
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x
class UNet(nn.Module):
    def __init__(self,  n_classes, in_channels=3, pretrained = False, ignore_index=-1, weight=None):
        super(UNet, self).__init__()
        print("CHANNEL:",in_channels)
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.inc= double_conv(in_channels, 64)
        self.down1 =double_down_max(64, 128)
        self.down2 =double_down_max(128, 256)
        self.down3 =double_down_max(256, 512)
        self.down4 =double_down_max(512, 1024)
        self.up1 = unet_up(1024, 512)
        self.up2 = unet_up(512, 256)
        self.up3 = unet_up(256, 128)
        self.up4 = unet_up(128, 64)
        # self.outc = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        self.outc = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0)
        )

    def forward(self,x,labels=None, th=1.0):
        #images=x
        #print(x.shape)
        # = x.size()
        x = self.inc(x)    # 256*256*3 # 256*256*64
        x1 = x
        #print('x1',x1.shape)
        x = self.down1(x)
        x2 = x # 128*128*64 #128*128*128 # 128*128*128
        #print('x2',x2.shape)
        x=self.down2(x)
        x3 = x # 64*64*128 #64*64*256 #64*64*256
        #print('x3',x3.shape)
        x=self.down3(x)
        x4 = x # 32*32*256 #32*32*512 #32*32*512
        #print('x4',x4.shape)
        x= self.down4(x) # 16*16*1024 #16*16*1024 # 16*16*1024
        #print('x5',x.shape)
        x = self.up1(x, x4) #16*16*1024->32*32*512+32*32*512 # 32*32*512
        #print('UP1',x.shape)
        x = self.up2(x, x3)  #32*32*512->64*64*256+64*64*256 # 64*64*256
        #print('UP2',x.shape)
        x = self.up3(x, x2) #64*64*256->128*128*128+128*128*128 #128*128*128
        #print('UP3',x.shape)
        x = self.up4(x, x1) #128*128*128->256*256*64+256*256*64 #256*256*64
        #print('UP4',x.shape)
        x = self.outc(x)    #256*256*8 (6,8,256,256)
        #print('out', x.shape)
        #x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)
        # if labels is not None:
        #     #total_valid_pixel 抛去被背景值的有效像素，从标签中来
        #     # labels=torch.where(labels.data==255,2,labels)
        #     losses, total_valid_pixel = self.mceloss(x, labels, th=th)
        #     return x, losses, total_valid_pixel
        # else:
        #     return x
        return x
