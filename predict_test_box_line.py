import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2,resnet18_fpn_backbone
from draw_box_utils import draw_objs
from network_files.transform import GeneralizedRCNNTransform as GTransform
import transforms as my_transforms
from val_teds.get_score_from_boxs import Postprocess as Box_Postprocess
from val_teds.get_score_line_no_merge import Postprocess as Line_Postprocess
from val_teds.src.metric import TEDS
from tqdm import tqdm
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
    backbone = resnet18_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model
# def model_val(img_batch):


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=4)
    # blank = ' '
    # print('-' * 90)
    # print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
    #       + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
    #       + ' ' * 3 + 'number' + ' ' * 3 + '|')
    # print('-' * 90)
    # num_para = 0
    # type_size = 1  # 如果是浮点数就是4
    # for index, (key, w_variable) in enumerate(model.named_parameters()):
    #     if len(key) <= 30:
    #         key = key + (30 - len(key)) * blank
    #     shape = str(w_variable.shape)
    #     if len(shape) <= 40:
    #         shape = shape + (40 - len(shape)) * blank
    #     each_para = 1
    #     for k in w_variable.shape:
    #         each_para *= k
    #     num_para += each_para
    #     str_num = str(each_para)
    #     if len(str_num) <= 10:
    #         str_num = str_num + (10 - len(str_num)) * blank
    #
    #     print('| {} | {} | {} |'.format(key, shape, str_num))
    # print('-' * 90)
    # print('The total number of parameters: ' + str(num_para))
    # print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    # print('-' * 90)
    # total = sum([param.nelement() for param in model.parameters()])
    #
    # print("Number of parameter: %.2fM" % (total / 1e6))
    # exit(0)
    # load train weights
    weights_path = "./work511/resNetFpn-model-1.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './table_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}
    img_folder = "/home/li/Desktop/disk2/datasets/pubtabnet/pubtabnet_TSRNet/val"
    img_list = os.listdir(img_folder)
    img_list = tqdm(img_list)
    score_list = []
    batch_size = 10
    num_now = 0
    img_batch = []
    filename_list = []
    gt_val_file = "/home/li/Desktop/disk2/wy/TableRe/TSR4/model/gtVal_1212.json"
    box_post = Box_Postprocess("thr_file",
                               row_thr=0.4, col_thr=0.6,
                               merboxs=None,
                               teds=TEDS(n_jobs=1),
                               gt_val_file=gt_val_file,
                               min_width=4,
                               save_file='./Val_data.json',
                               iou=0.5)
    line_post = Line_Postprocess(
                            "thr_file",
                            row_thr=0.4, col_thr=0.6,
                            DBPost_merge=[],
                            teds=TEDS(n_jobs=1),
                            gt_val_file=gt_val_file,
                            min_width=8,
                            save_file='./Val_data.json',
                            iou=0.7)
    for id,img in enumerate(img_list):
        imgname = img
        img_path = os.path.join(img_folder,img)
        # load image
        original_img = Image.open(img_path)
        # img = transforms.ToTensor(original_img)
        # from pil image to tensor, do not normalize image
        data_transform = my_transforms.Compose([my_transforms.ToTensor(),
                                         my_transforms.My_padding(),
                                         my_transforms.resize_img(GTransform(min_size=800,max_size=800)),
                                         # transforms.RandomHorizontalFlip(0.5)
                                         my_transforms.generate_scnn_target()])
        img,traget = data_transform(original_img,{})
        # expand batch dimension
        if num_now==0:
            img_batch = torch.unsqueeze(img, dim=0)
            filename_list.append(imgname)
            num_now +=1
        elif num_now<batch_size:
            img = torch.unsqueeze(img, dim=0)
            shape = img.size()
            assert shape[2] == shape[3]==800
            img_batch =torch.cat([img_batch,img],0)
            filename_list.append(imgname)
            num_now +=1
        if num_now>=batch_size or id >= len(img_list)-1:
            num_now =0
        else:
            continue
        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions,_,p2_r,p2_c = model(img_batch.to(device))
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))
            for one_id,pred in enumerate(predictions):
                predict_boxes = pred["boxes"].to("cpu").numpy()
                predict_classes = pred["labels"].to("cpu").numpy()
                predict_scores = pred["scores"].to("cpu").numpy()

                # get teds
                box_thresh = 0.5
                merge_boxs = []
                row_boxs = []
                col_boxs = []
                head_box = []
                for pid,label in enumerate(predict_classes.tolist()):
                    if label==1 and predict_scores[pid]>=box_thresh:
                        merge_boxs.append(predict_boxes[pid])
                    elif label ==2 and predict_scores[pid]>=box_thresh:
                        head_box.append(predict_boxes[pid])
                    elif label ==3 and predict_scores[pid]>=box_thresh:
                        row_boxs.append(predict_boxes[pid])
                    elif label ==4 and predict_scores[pid]>=box_thresh:
                        col_boxs.append(predict_boxes[pid])
                # score = box_post.Get_score_out(row_boxs=row_boxs,col_boxs=col_boxs,merge_boxs=merge_boxs,
                #                    head_box=head_box,filename=imgname,imgshape=[])
                row = (255 * p2_r[one_id]).to('cpu', torch.uint8).detach().numpy()
                col = (255 * p2_c[one_id]).to('cpu', torch.uint8).detach().numpy()
                score = line_post.Get_score_out(thr_row=row,thr_col=col,merge_boxs=merge_boxs,
                                                filename=filename_list[one_id],imgshape=[],
                                                head_box=head_box)
                score_list.append(score)
                if len(score_list)%100==1:
                    print('ave_score:',sum(score_list)/len(score_list))
            filename_list = []
            continue


            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
            drow_img = img.cpu().clone()
            drow_img = drow_img.squeeze(0)
            drow_img = transforms.ToPILImage()(drow_img)
            plot_img = draw_objs(drow_img,
                                 predict_boxes,
                                 predict_classes,
                                 predict_scores,
                                 category_index=category_index,
                                 box_thresh=0.5,
                                 line_thickness=4,
                                 font='arial.ttf',
                                 font_size=7)
            # plt.imshow(plot_img)
            # plt.show()
            # 保存预测的图片结果
            plot_img.save(os.path.join("./pred_result",imgname))

    print('ave_score:', sum(score_list) / len(score_list))
if __name__ == '__main__':
    main()
