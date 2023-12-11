import random

import torch
from torchvision.transforms import functional as F
import numpy as np
import cv2
from Frcnn_out import extend_anno
import torchvision.transforms.functional as TF
class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        # target['file_name'] = image.filename.split("/")[-1]
        image = F.to_tensor(image)
        return image, target

class resize_img(object):
    def __init__(self,resize_f):
        self.resize_f = resize_f
    def __call__(self,img,target):
        img,target = self.resize_f.resize(img,target)
        return img,target
class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target
class My_padding(object):
    def __init__(self):
        pass
    def __call__(self,img, target):
        _,h,w = img.shape
        target['scnn_ori_shape']= np.array([h,w])
        if h >w:
            width = h-w
            p2d=[0, 0,width,0] #[L,U,R,D]
        else:
            width = w-h
            p2d=[0,0,0,width]
        img_pad=F.pad(img,p2d,0,'constant')
        _,h,w = img_pad.shape
        assert h==w
        return img_pad, target
class generate_scnn_target(object):
    def __init__(self):
        pass
    def __call__(self, img,target):
        imgshape = img.size()
        assert imgshape[1]==imgshape[2]==800
        if "labels" not in target.keys():
            return img,target
        labels = target['labels'].numpy()
        boxes = target['boxes'].numpy()
        col_box = []
        row_box = []
        for label,box in zip(labels,boxes):
            if label==3:
                row_box.append(box)
            if label ==4:
                col_box.append(box)
        ori_shape = target['scnn_ori_shape']
        _,h,w = img.shape
        h=800
        w=800
        assert h==w==800
        maxbl=h/max(ori_shape)
        resize_shape = np.array(ori_shape)*maxbl
        target['scnn_ori_shape'] = torch.tensor(target['scnn_ori_shape'])
        html= extend_anno(row_boxes=row_box,col_boxes=col_box,imgshape=resize_shape,extend_pix=8)
        separator_row=html['separator_row']
        separator_col = html['separator_col']
        thr_row = np.zeros((2*h,1))
        thr_col = np.zeros((1,2*w))
        for row in separator_row:
            up = 2*max(int(row[0]),0)
            dowm = 2*min(int(row[1]),h)
            thr_row[up:dowm,:] = 1
        for col in separator_col:
            left = 2*max(int(col[0]),0)
            right = 2*min(int(col[1]),w)
            thr_col[:,left:right] = 1
        target['row'] = torch.tensor(thr_row,dtype=torch.float32).reshape(1,2*h,1)
        target['col'] = torch.tensor(thr_col,dtype=torch.float32).reshape(1,1,2*w)
        target['row_mask'] = torch.ones((1,2*h,1),dtype=torch.float32)
        target['col_mask'] = torch.ones((1,1,2*w),dtype=torch.float32)
        labels = target['labels'].numpy()
        label_id = np.where(labels<3)
        target['labels'] = target['labels'][label_id]
        target['boxes'] = target['boxes'][label_id]
        target['area'] = target['area'][label_id]
        target['iscrowd'] = target['iscrowd'][label_id]
        return img,target
class Normalize:
    """Normalize a tensor image with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    # def __init__(self):
    #     pass
    def __call__(self, img,targets):
        img = TF.normalize(img,self.mean, self.std)
        return img,targets