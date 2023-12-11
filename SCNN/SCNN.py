import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.bn=nn.BatchNorm2d(1)
        # self.bn.momentum=

    def forward(self, p2_r):
        p2_r = self.r_scnn(x=p2_r, conv=self.conv_l_r, reverse=False)
        p2_r = self.r_scnn(x=p2_r, conv=self.conv_l_r, reverse=True)
        p2_r = self.upsample(p2_r)
        p2_r = self.conv(p2_r)
        p2_r=self.bn(p2_r)
        p2_r=torch.sigmoid(p2_r)

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

    def upsample(self, x):
        _, _, H, W = x.size()
        return F.interpolate(x, size=(4 * H, 4 * W), mode='bilinear', align_corners=True)


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
        self.conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, p2_c):
        p2_c = self.c_scnn(x=p2_c, conv=self.conv_u_d, reverse=False)
        p2_c = self.c_scnn(x=p2_c, conv=self.conv_u_d, reverse=False)
        p2_c = self.upsample(p2_c)
        p2_c = self.conv(p2_c)
        p2_c = self.bn(p2_c)
        p2_c = torch.sigmoid(p2_c)

        return p2_c

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
        return F.interpolate(x, size=(4 * H, 4 * W), mode='bilinear', align_corners=True)
