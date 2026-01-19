# sub-parts of the U-Net model
# unet 20210604 modified by Xiaorou.Zheng

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class double_down_max(nn.Module):
    def __init__(self, in_ch, out_ch):
        # Input 256*256*in_ch
        # 128*128*in_ch
        # 128*128*out_ch
        super(double_down_max, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x

class unet_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(unet_up, self).__init__()
        # 1024 256
        # 512 512
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.Conv2d(in_ch, in_ch//2, kernel_size=2, padding=0),
                                    nn.ReLU(inplace=True))
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=4, stride=2)
        #1024 #256
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1 16*16*1024
        # x2 32*32*512
        x1 = self.up(x1) #16*16*1024—>32*32*512
        # input is CHW
        diffY = int(x2.size()[2] - x1.size()[2])
        diffX = int(x2.size()[3] - x1.size()[3])
        pad=(diffX // 2, diffX - diffX//2,diffY // 2, diffY - diffY//2)
        # 塑造将x1塑造成x2
        x1 = F.pad(x1, pad)
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # 将两个张量拼接在一起
        # 32*32*1024
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
class downcat(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(downcat, self).__init__()
        self.downcat = nn.MaxPool2d(2)
    
        self.conv = double_conv(in_ch, out_ch)
        
    def forward(self,x1,x2):
        #print(x1.shape)
        x1 = self.downcat(x1)
        #print(x1.shape)
         # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        #print(x.shape)
        x = self.conv(x)
        return x
        
        
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
