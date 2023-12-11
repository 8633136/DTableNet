import os
import time
import json
import numpy as np
import torch
import copy
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from network_files import FasterRCNN_predict as FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2,resnet18_fpn_backbone
from draw_box_utils import draw_objs
from network_files.transform import GeneralizedRCNNTransform as GTransform
import transforms as my_transforms
from val_teds.get_score_from_boxs import Postprocess as Box_Postprocess
from val_teds.get_score_line_V2 import Postprocess as Line_Postprocess
from val_teds.src.metric import TEDS
from tqdm import tqdm
import shutil
import torch.nn as nn
from thop import profile
def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    # from train
    # # # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


if __name__ == '__main__':
    model = create_model(num_classes=3)
    input = torch.randn(1,3,800,800)
    flops,params = profile(model,inputs=(input,))
    print("flops:",flops)
    print("params:",params)