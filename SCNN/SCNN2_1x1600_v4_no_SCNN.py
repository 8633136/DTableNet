import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#TODO
# init_nn.ConvTransposed

class R_SCNN(nn.Module):
    # 行方向分支的第一个SCNN，结果为H*W/8*C
    def __init__(self, ms_ks=9):
        """
        Argument
            ms_ks: kernel size in message passing conv
            消息传递conv中的内核大小
        """
        super(R_SCNN, self).__init__()
        self.conv_l_r = nn.Conv2d(256, 256, (1, ms_ks), padding=(0, ms_ks // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1)
        self.convv2 = nn.Conv2d(in_channels=200, out_channels=1, kernel_size=1, padding=0, bias=False)
        # self.relu = nn.ReLU()
        self.conv_up2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu = nn.ReLU()
        self.convt1 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False)
        self.convt2 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False)
        self.convt3 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=4,stride=2,padding=1,bias=False)
    def forward(self, p2_r):
        # p2_r = self.r_scnn(x=p2_r, conv=self.conv_l_r, reverse=False)
        # p2_r = self.bn1(p2_r)
        # p2_r = self.relu(p2_r)
        #
        # p2_r = self.r_scnn(x=p2_r, conv=self.conv_l_r, reverse=True)
        # p2_r = self.bn1(p2_r)
        # p2_r = self.relu(p2_r)

        # p2_r = self.upsample(p2_r)  # to  H/2*W/32*64
        p2_r = self.convt1(p2_r)  # to  H/2*W/32*64
        p2_r = self.conv_up2(p2_r)
        # p2_r = self.upsample(p2_r)  # to H*W/32*64
        p2_r = self.convt2(p2_r)  # to H*W/32*64
        p2_r = self.conv(p2_r)
        # p2_r = self.upsample(p2_r)  # H*W/32*1
        p2_r = self.convt3(p2_r)  # H*W/32*1
        p2_r = torch.transpose(p2_r,dim0=1,dim1=3)  # 2*H*1*W/32
        p2_r = self.convv2(p2_r)  # H*1*1
        # p2_r = self.bn2(p2_r)
        p2_r = torch.sigmoid(p2_r)

        return p2_r

    def r_scnn(self, x, conv, reverse=False):
        b, c, h, w = x.shape
        slices = [x[:, :, :, i:(i + 1)] for i in range(w)]
        dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        out = out[::-1]
        return torch.cat(out, dim=dim)
    def f_conv2(self,p2_r):
        in_channels = p2_r.size()[1]
        # in_channels = nn.ModuleList(in_channels)
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=1, kernel_size=1, padding=0, bias=False)
        return self.conv2(p2_r)
    def upsample(self, x):
        _, _, H, W = x.size()
        return F.interpolate(x, size=(2* H, W), mode='bilinear', align_corners=True)


class C_SCNN(nn.Module):
    # 列方向分支的第一个SCNN，结果为H/8*W*C
    def __init__(self, ms_ks=9):
        """
        Argument
            ms_ks: kernel size in message passing conv
            消息传递conv中的内核大小
        """
        super(C_SCNN, self).__init__()
        self.conv_u_d = nn.Conv2d(256, 256, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1)
        self.convv2 = nn.Conv2d(in_channels=200, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.conv_up2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)

        self.relu = nn.ReLU()
        # self.relu = nn.ELU()
        self.convt1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1,
                                         bias=False)
        self.convt2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1,
                                         bias=False)
        self.convt3 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1,
                                         bias=False)

    def forward(self, p2_c):
        # p2_c = self.c_scnn(x=p2_c, conv=self.conv_u_d, reverse=False)
        # p2_c = self.bn1(p2_c)
        # p2_c = self.relu(p2_c)
        # f1 = p2_c
        # p2_c = self.c_scnn(x=p2_c, conv=self.conv_u_d, reverse=False)
        # p2_c = self.bn1(p2_c)
        # p2_c = self.relu(p2_c)
        # f2 = p2_c
        # p2_c = self.upsample(p2_c)
        p2_c = self.convt1(p2_c)
        p2_c = self.conv_up2(p2_c)
        # p2_c = self.upsample(p2_c)  # H/32*W/2*64
        p2_c = self.convt2(p2_c)
        p2_c = self.conv(p2_c)  # H/32*W/2*1
        # p2_c = self.upsample(p2_c) # H/32*W*1
        p2_c = self.convt3(p2_c)
        p2_c = torch.transpose(p2_c,dim0=1,dim1=2)  # B,H/32,1,W
        p2_c = self.convv2(p2_c)  # B,1,1,W
        p2_c = torch.sigmoid(p2_c)

        return p2_c
    def f_conv2(self,p2_c):
        in_channels = p2_c.size()[1]
        conv2 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0, bias=False)
        return conv2(p2_c)

    def c_scnn(self, x, conv, reverse=False):
        b, c, h, w = x.shape
        slices = [x[:, :, i:(i + 1), :] for i in range(h)]
        dim = 2
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        out = out[::-1]
        return torch.cat(out, dim=dim)

    def upsample(self, x):
        _, _, H, W = x.size()
        return F.interpolate(x, size=(H, 2*W), mode='bilinear', align_corners=True)
