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
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.conv_up2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        # self.relu = nn.SELU()

    def forward(self, p2_r):
        p2_r = self.r_scnn(x=p2_r, conv=self.conv_l_r, reverse=False)
        p2_r = self.bn1(p2_r)
        p2_r = self.relu(p2_r)

        p2_r = self.r_scnn(x=p2_r, conv=self.conv_l_r, reverse=True)
        p2_r = self.bn1(p2_r)
        p2_r = self.relu(p2_r)

        p2_r = self.upsample(p2_r)  # to  H/2*W/32*64
        p2_r = self.conv_up2(p2_r)
        p2_r = self.upsample(p2_r)  # to H*W/32*64
        p2_r = self.conv(p2_r)
        p2_r = self.upsample(p2_r)  # H*W/32*1
        p2_r = torch.transpose(p2_r,dim0=1,dim1=3)  # 2*H*1*W/32
        p2_r = self.conv2(p2_r)  # H*1*1
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
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.conv_up2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)

        self.relu = nn.ReLU()
        # self.relu = nn.SELU()

    def forward(self, p2_c):
        p2_c = self.c_scnn(x=p2_c, conv=self.conv_u_d, reverse=False)
        p2_c = self.bn1(p2_c)
        p2_c = self.relu(p2_c)
        f1 = p2_c
        p2_c = self.c_scnn(x=p2_c, conv=self.conv_u_d, reverse=False)
        p2_c = self.bn1(p2_c)
        p2_c = self.relu(p2_c)
        f2 = p2_c
        p2_c = self.upsample(p2_c)
        p2_c = self.conv_up2(p2_c)
        p2_c = self.upsample(p2_c)  # H/32*W/2*64
        p2_c = self.conv(p2_c)  # H/32*W/2*1
        p2_c = self.upsample(p2_c) # H/32*W*1
        p2_c = torch.transpose(p2_c,dim0=1,dim1=2)  # B,H/32,1,W
        p2_c = self.conv2(p2_c)  # B,1,1,W
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
class mid2(nn.Module):
    def __init__(self, ms_ks=9):
        super(mid2, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.conv1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=7,stride=4,padding=2)
        self.conv2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=5,padding=2,stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.convt1 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=4,padding=0)
        self.convt1.weight.data = self.bilinear_kernel(in_channels=128,out_channels=64,kernel_size=4)
        self.convt2 = nn.ConvTranspose2d(in_channels=64,out_channels=1,kernel_size=4,padding=1,stride=2)
        self.convt2.weight.data = self.bilinear_kernel(in_channels=64,out_channels=1,kernel_size=4)
        self.relu = nn.SELU()
    def forward(self,p2):
        p2_m = self.maxpool(p2)
        p2_m = self.conv1(p2_m)  # ->1/16  7*7conv
        f1 = p2_m
        p2_m = self.relu(p2_m)
        p2_m = self.conv2(p2_m)  # ->1/16,5*5conv
        f2 = p2_m
        p2_m = self.relu(p2_m)
        p2_m = self.bn1(p2_m)
        p2_m = self.convt1(p2_m)  # ->1/4,4*4conv
        f3 = p2_m
        p2_m = p2_m + p2
        p2_m = self.convt2(p2_m)  #->1/2,4*4conv
        # p2_m = self.relu(p2_m)
        # return p2_m
        p2_m = torch.sigmoid(p2_m)
        return [p2_m,f3,f2,f1]

    def bilinear_kernel(self,in_channels, out_channels, kernel_size):
        '''
        return a bilinear filter tensor
        '''
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

        # 上半部分是生成一层双线性插值核
        # 下面是拉伸
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        # 赋值到每一层
        for i in range(in_channels):
            for j in range(out_channels):
                # weight[i,j,:,:] = np.zeros((kernel_size,kernel_size),dtype=np.float32)
                weight[i,j,:,:] = filt

        # weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)
class Mid(nn.Module):
    def __init__(self,ms_ks=9):
        super(Mid,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv_l_r = nn.Conv2d(64, 64, (1, ms_ks), padding=(0, ms_ks // 2), bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_u_d = nn.Conv2d(64, 64, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False)

        self.convt1=nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=2,stride=2,padding=0,bias=False)
        self.convt2=nn.ConvTranspose2d(in_channels=64,out_channels=1,kernel_size=2,stride=2,padding=0,bias=False)
        self.bn=nn.BatchNorm2d(1)

    def forward(self,p2):
        p2_m=self.conv1(p2)
        p2_m=self.r_scnn(p2_m,conv=self.conv_l_r,reverse=False)
        p2_m=self.bn1(p2_m)
        p2_m=self.relu(p2_m)

        p2_m=self.r_scnn(p2_m,conv=self.conv_l_r,reverse=True)
        p2_m = self.bn1(p2_m)
        p2_m = self.relu(p2_m)

        p2_m=self.conv2(p2_m)
        p2_m=self.bn1(p2_m)
        p2_m=self.relu(p2_m)

        p2_m = self.c_scnn(p2_m, conv=self.conv_u_d, reverse=False)
        p2_m = self.bn1(p2_m)
        p2_m = self.relu(p2_m)

        p2_m = self.c_scnn(p2_m, conv=self.conv_u_d, reverse=True)
        p2_m = self.bn1(p2_m)
        p2_m = self.relu(p2_m)

        p2_m=self.convt1(p2_m)
        p2_m=self.bn1(p2_m)
        p2_m=self.relu(p2_m)

        p2_m=p2_m+p2

        p2_m=self.convt1(p2_m)
        p2_m = self.bn1(p2_m)
        p2_m = self.relu(p2_m)

        p2_m=self.convt2(p2_m)
        p2_m = self.bn(p2_m)
        p2_m = torch.sigmoid(p2_m)

        return p2_m

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
# class mid(nn.Module):
#     def __init__(self):
#         super(mid, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=64,out_channels=256,stride=1,padding=0,kernel_size=2)
#         self.bn1 = nn.BatchNorm2d(256)
#
#         self.conv2 = nn.Conv2d(in_channels=256,out_channels=64,stride=1,padding=1,kernel_size=2)
#         self.bn2 = nn.BatchNorm2d(64)
#
#         self.convt1 = nn.ConvTranspose2d(in_channels=64,out_channels=256,stride=2,kernel_size=2,padding=0)
#
#         self.convt2 = nn.ConvTranspose2d(in_channels=256,out_channels=1,stride=2,kernel_size=2,padding=0)
#         self.relu = nn.ReLU()
#     def upsample(self, x):
#         _, _, H, W = x.size()
#         return F.interpolate(x, size=(2 * H, 2 * W), mode='bilinear', align_corners=True)
#     def forward(self, p2):
#        p2_m = self.conv1(p2)
#        p2_m = self.bn1(p2_m)
#        p2_m = self.relu(p2_m)
#
#        p2_m = self.conv2(p2_m)
#        p2_m = self.bn2(p2_m)
#        p2_m = self.relu(p2_m)
#
#        p2_m = self.convt1(p2_m)
#
#        p2_m = self.convt2(p2_m)
#        p2_m = torch.sigmoid(p2_m)
#
#        return p2_m
