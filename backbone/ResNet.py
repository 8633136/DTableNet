import torch.nn as nn
import torch
import torch.nn.functional as F
from .swinT import SwinT
# from model.DCN import DeformConv2d
# 针对18/34层网络的conv2_x，conv3_x，conv4_x，conv5_x的系列卷积层
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,DCN=False):
        super(BasicBlock, self).__init__()
        if DCN:
            self.conv1 = DeformConv2d(inc=in_channel, outc=out_channel,
                                   kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.relu = nn.ReLU()
            self.conv2 = DeformConv2d(inc=out_channel, outc=out_channel,
                                   kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channel)
            self.downsample = downsample
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                   kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                   kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channel)
            self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# 针对50/101/152层网络的conv2_x，conv3_x，conv4_x，conv5_x的系列卷积层
class Bottleneck_resnet(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,DCN=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.conv1 = SwinT(in_channels=in_channel,out_channels=out_channel,input_resolution=(200,200),num_heads=8,window_size=7)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel,input_size,stride=1, downsample=None,DCN=None):
        super(Bottleneck, self).__init__()
        # input_size = 200
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.conv1 = SwinT(in_channels=in_channel,out_channels=out_channel,input_resolution=(input_size,input_size),num_heads=8,window_size=7)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=stride, bias=False, padding=1)
        self.conv2 = SwinT(in_channels=out_channel,out_channels=out_channel,input_resolution=(input_size,input_size),num_heads=8,window_size=7)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        # self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
        #                        kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.conv3 = SwinT(in_channels=in_channel,out_channels=out_channel,input_resolution=(input_size,input_size),num_heads=8,window_size=7)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
# 网络ResNet
class ResNet(nn.Module):
    def __init__(self, block, blocks_num):
        # block有两种：BasicBlock针对18/34层网络，Bottleneck针对50/101/152层网络
        # blocks_num是一个列表，表示conv2_x，conv3_x，conv4_x，conv5_x分别对应的卷积层个数
        super(ResNet, self).__init__()

        self.in_channel = 64
        self.expansion=block.expansion
        self.out_channels = 256
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
													padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0],input_size=224, stride=1)   # conv2_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1],input_size=112, stride=2,DCN=False)  # conv3_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2],input_size=56, stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3],input_size=28 , stride=2)  # conv5_x

        #C5-->P5

        self.toplayer = nn.Conv2d(512*block.expansion, 64*block.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5=nn.BatchNorm2d(64*block.expansion)

        # C4-->P4
        self.latlayer1 = nn.Conv2d(256*block.expansion, 64*block.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64*block.expansion)
        # self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # C3-->P3
        self.latlayer2 = nn.Conv2d(128*block.expansion, 64*block.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64*block.expansion)
        # self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # C2-->P2
        self.latlayer3 = nn.Conv2d(64*block.expansion, 64*block.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64*block.expansion)
        self.smooth3 = nn.Conv2d(64*block.expansion, 64*block.expansion, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=64*block.expansion, out_channels=256, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)


    # 构建conv2_x，conv3_x，conv4_x，conv5_x卷积层
    def _make_layer(self, block, channel, block_num,input_size, stride=1,DCN=False):
        downsample = None  # 设定不是虚线,downsample不为None即是虚线
        # 网络结构中虚线路径的设定，只有18/34层网络的conv2_x不执行if语句
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1,
                                                        stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        # conv2_x，conv3_x，conv4_x，conv5_x的第一个卷积层
        # 第一层是虚线路径，传入downsample，因为两个block里面默认downsample = None
        layers.append(block(self.in_channel, channel,input_size, downsample=downsample, stride=stride,DCN=DCN))
        self.in_channel = channel * block.expansion
        # conv2_x，conv3_x，conv4_x，conv5_x每个系列的剩余卷积层，均为实线
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel,input_size))

        return nn.Sequential(*layers)

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.conv1(x)  # conv1
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)  # maxpool

        c2 = self.layer1(c1)  # conv2_x
        c3 = self.layer2(c2)  # conv3_x
        c4 = self.layer3(c3)  # conv4_x
        c5 = self.layer4(c4)  # conv5_x

        p5=self.toplayer(c5)
        p5=self.bn5(p5)
        p5=self.relu(p5)

        p4=self.upsample_add(p5,self.latlayer1(c4))
        p4=self.bn4(p4)
        p4=self.relu(p4)

        p3 = self.upsample_add(p4, self.latlayer2(c3))
        p3 = self.bn3(p3)
        p3 = self.relu(p3)

        p2 = self.upsample_add(p3, self.latlayer3(c2))
        p2 = self.bn2(p2)
        p2 = self.relu(p2)

        p2=self.conv2(p2)
        p2 = self.bn6(p2)
        p2 = self.relu(p2)
        p_all = {
            '0':p2,
            '1':p3,
            '2':p4,
            '3':p5
        }

        return p_all


# def resnet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])


# def resnet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])
#
# if __name__ == '__main__':
#     a=torch.rand(1,3,800,800)
#     net=resnet50()
#     b=net(a)
#     print(b.shape)
