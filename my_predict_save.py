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
# def model_val(img_batch):


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1")
    # device = 'cpu'
    print("using {} device.".format(device))

    # create model

    model = create_model(num_classes=3)
    # weights_path = "./work705_adjust_datasets/resNetFpn-model-15.pth"
    # weights_path = "./710_serve_test/dence/resNetFpn-model-24.pth"
    # weights_path = "./710_serve_test/resNetFpn-model-20.pth"
    # weights_path = "./716_serve_test/resNetFpn-model-22.pth"
    weights_path = "./731_serve_test_convt/resNetFpn-model-36.pth"
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
    # img_folder = "/home/li/Desktop/disk2/datasets/pubtabnet/val_img_no_error"
    # img_folder = "/home/li/Desktop/disk2/datasets/pubtabnet/voc_dence_no800_38791_689/JPEGImgages_val"
    # img_folder = "/home/li/Desktop/disk2/datasets/pubtabnet/voc_dence/JPEGImgages_val_dence"
    # img_folder = "/home/li/Desktop/disk2/datasets/pubtabnet/coco_pubtabnet_new2/train2017"
    img_list = os.listdir(img_folder)
    img_list = tqdm(img_list)
    score_list = []
    batch_size = 16
    num_now = 0
    img_batch = []
    filename_list = []
    gt_val_file = "/home/li/Desktop/disk2/wy/TableRe/TSR4/model/gtVal_1212_text.json"
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
                            row_thr=0.65, col_thr=0.7,
                            DBPost_merge=[],
                            teds=TEDS(n_jobs=4,structure_only=True),
                            gt_val_file=gt_val_file,
                            min_width_row=10,min_width_col=12,
                            save_file='./Val_data.json',
                            iou=0.7)
    box_thresh = 0.9
    # save low score img
    save_low_score_folder = "./A814_epoch36_save"
    save_json_file=os.path.join(save_low_score_folder,'save_thr.json')
    save_json_data={}
    save_score_dict = {}
    save_score_file = os.path.join(save_low_score_folder,'save_score_814_epoch36_65710129.json')
    save_img_txt = os.path.join(save_low_score_folder,'low_score.txt')
    img_name_list = []
    low_score_thr = 1.0
    low_score_folder_list=['0.8-0.6','0.6-0.4','0.4-0.0']
    low_score_thr_list = [0.6,0.4,0.0]
    for low_folder in low_score_folder_list:
        if not os.path.exists(os.path.join(save_low_score_folder,low_folder)):
            os.makedirs(os.path.join(save_low_score_folder,low_folder))
    model.eval()  # 进入验证模式
    if not os.path.exists(save_low_score_folder):
        os.makedirs(save_low_score_folder)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            m.training = True
    low_score_list = []
    data_transform = my_transforms.Compose([my_transforms.ToTensor(),
                                            # my_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            my_transforms.My_padding(),
                                            my_transforms.resize_img(GTransform(min_size=800, max_size=800)),
                                            # transforms.RandomHorizontalFlip(0.5)
                                            my_transforms.generate_scnn_target()])
    for id,img in enumerate(img_list):
        imgname = img
        img_path = os.path.join(img_folder,img)
        # load image
        original_img = Image.open(img_path)
        # img = transforms.ToTensor(original_img)
        # from pil image to tensor, do not normalize image

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

        # import copy
        # model2 = copy.deepcopy(model)
        # model2.train()
        with torch.no_grad():
            predictions,_,p2_r,p2_c = model(img_batch.to(device))
            for one_id,pred in enumerate(p2_c):
                # get teds
                merge_boxs = []
                row_boxs = []
                col_boxs = []
                head_box = []
                pred = predictions[one_id]
                predict_boxes = pred["boxes"].to("cpu").numpy()
                predict_classes = pred["labels"].to("cpu").numpy()
                predict_scores = pred["scores"].to("cpu").numpy()


                for pid,label in enumerate(predict_classes.tolist()):
                    if label==1 and predict_scores[pid]>=box_thresh:
                        merge_boxs.append(predict_boxes[pid])
                    elif label ==2 and predict_scores[pid]>=box_thresh:
                        head_box.append(predict_boxes[pid])
                    elif label ==3 and predict_scores[pid]>=box_thresh:
                        row_boxs.append(predict_boxes[pid])
                    elif label ==4 and predict_scores[pid]>=box_thresh:
                        col_boxs.append(predict_boxes[pid])
                row = (255 * p2_r[one_id]).to('cpu', torch.int16).detach().numpy()
                col = (255 * p2_c[one_id]).to('cpu', torch.int16).detach().numpy()
                score,point_row,point_col,headnum,_ = line_post.Get_score_out(thr_row=row,thr_col=col,merge_boxs=merge_boxs,
                                                filename=filename_list[one_id],imgshape=[],
                                                head_box=head_box)
                score_list.append(score)
                save_score_dict[filename_list[one_id]] = score
                if len(score_list) % 100 == 1:
                    print('ave_score:', sum(score_list) / len(score_list))
                    # f = open(save_json_file, 'w', encoding='utf-8')
                    # f.write(json.dumps(save_json_data))
                    # f.close()
                    f = open(save_score_file, 'w', encoding='utf-8')
                    f.write(json.dumps(save_score_dict))
                    f.close()
                # save_json_data[filename_list[one_id]] = {
                #     'predict_boxes':predict_boxes.tolist(),
                #     'predict_classes':predict_classes.tolist(),
                #     'predict_scores':predict_scores.tolist(),
                #     'row':row.tolist(),
                #     'col':col.tolist()
                # }

                continue
                if score<low_score_thr:
                    low_score_list.append(score)
                    img_name_list.append(filename_list[one_id])
                    den_path = os.path.join(save_low_score_folder,filename_list[one_id])
                    ori_path = os.path.join(img_folder,filename_list[one_id])
                    # shutil.copyfile(ori_ppath,den_path)
                    #save it and boxes
                    # drow_img = img_batch.cpu().clone()[one_id]
                    # drow_img = drow_img.squeeze(0)
                    # drow_img = transforms.ToPILImage()(drow_img)
                    # plot_img = draw_objs(drow_img,
                    #                      predict_boxes,
                    #                      predict_classes,
                    #                      predict_scores,
                    #                      category_index=category_index,
                    #                      box_thresh=0.5,
                    #                      line_thickness=4,
                    #                      font='arial.ttf',
                    #                      font_size=7)
                    # plt.imshow(plot_img)
                    # plt.show()
                    # 保存预测的图片结果
                    for id,score_thr in enumerate(low_score_thr_list):
                        if score_thr<score:
                            folder = os.path.join(save_low_score_folder,low_score_folder_list[id])
                    drow_img = img_batch[one_id].cpu().clone()
                    drow_img = drow_img.squeeze(0)
                    # img_np = np.array(plot_img)
                    expand_x = 50
                    expand_y = 50
                    save_img = np.zeros((3,800+expand_x,800+expand_y))
                    save_img = torch.tensor(save_img)
                    # b = save_img[:,:800, :800]
                    save_img[:,:800, :800] = drow_img
                    drow_img = transforms.ToPILImage()(save_img)
                    # svae data
                    ori_drow_img = copy.deepcopy(drow_img)
                    ori_predict_boxes = predict_boxes
                    ori_predict_classes = predict_classes
                    ori_predict_scores = predict_scores
                    point_row = np.array(point_row)
                    point_col = np.array(point_col)
                    len_row = point_row.shape[0]
                    len_col = point_col.shape[0]
                    row_box = np.zeros((len_row,4))
                    col_box = np.zeros((len_col,4))

                    row_box[:,1] = point_row[:,0]
                    row_box[:,2] = row_box[:,2]+800+expand_x/2
                    row_box[:,3] = point_row[:,1]
                    col_box[:,0] = point_col[:,0]
                    col_box[:,2] = point_col[:,1]
                    col_box[:,3] = col_box[:,3]+800+expand_y/2
                    predict_boxes = np.append(predict_boxes,row_box,axis=0)
                    predict_boxes = np.append(predict_boxes,col_box,axis=0)
                    row_class = np.ones((1,len_row),dtype=np.int16)*3
                    col_class = np.ones((1,len_col),dtype=np.int16)*4
                    predict_classes = np.append(predict_classes,row_class)
                    predict_classes = np.append(predict_classes,col_class)
                    row_score = np.ones((1, len_row))
                    col_score = np.ones((1, len_col))
                    predict_scores = np.append(predict_scores,row_score)
                    predict_scores = np.append(predict_scores,col_score)
                    plot_img = draw_objs(drow_img,
                                         predict_boxes,
                                         predict_classes,
                                         predict_scores,
                                         category_index=category_index,
                                         box_thresh=0.5,
                                         line_thickness=2,
                                         font='arial.ttf',
                                         font_size=5)
                    plot_img2 = draw_objs(ori_drow_img,
                                         ori_predict_boxes,
                                         ori_predict_classes,
                                         ori_predict_scores,
                                         category_index=category_index,
                                         box_thresh=0.5,
                                         line_thickness=2,
                                         font='arial.ttf',
                                         font_size=5)
                    img_save = Image.new('RGB',(2*800+expand_x,800+expand_y),(255,0,0))
                    img_save.paste(plot_img,(0,0))
                    img_save.paste(plot_img2,(800+expand_x,0))
                    prefix = str(round(score,2))+'_h'+str(headnum)+"_r"+str(len_row)+'_c'+str(len_col)+'_'

                    # plot_img.save(os.path.join(folder,prefix+filename_list[one_id]))
                    img_save.save(os.path.join(folder,prefix+filename_list[one_id]))
            filename_list = []
            continue
    # f = open(save_json_file, 'w', encoding='utf-8')
    # f.write(json.dumps(save_json_data))
    # f.close()
    f = open(save_score_file, 'w', encoding='utf-8')
    f.write(json.dumps(save_score_dict))
    f.close()
    print('ave_score:', sum(score_list) / len(score_list))
    # print("ave_low_score:",sum(low_score_list)/len(low_score_list))
def save_low_img(score,low_score_list,img,imgname):
    pass
def my_val(model,device):
    # read class_indict
    label_json_path = './table_classes.json'
    # gt_val_file = "/home/li/Desktop/disk2/wy/TableRe/TSR4/model/gtVal_1212.json"
    gt_val_file = "../gtVal_1212.json"
    img_folder = "../val"
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}
    img_list = os.listdir(img_folder)
    img_list = tqdm(img_list)
    score_list = []
    batch_size = 4
    num_now = 0
    img_batch = []
    filename_list = []
    line_post = Line_Postprocess(
        "thr_file",
        row_thr=0.75, col_thr=0.85,
        DBPost_merge=[],
        teds=TEDS(n_jobs=1),
        gt_val_file=gt_val_file,
        min_width_row=12, min_width_col=14,
        save_file='./Val_data.json',
        iou=0.7)
    box_thresh = 0.9
    model.eval()  # 进入验证模式
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            m.training = True
    data_transform = my_transforms.Compose([my_transforms.ToTensor(),
                                            # my_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            my_transforms.My_padding(),
                                            my_transforms.resize_img(GTransform(min_size=800, max_size=800)),
                                            # transforms.RandomHorizontalFlip(0.5)
                                            my_transforms.generate_scnn_target()])
    for id, img in enumerate(img_list):
        imgname = img
        img_path = os.path.join(img_folder, img)
        # load image
        original_img = Image.open(img_path)
        # img = transforms.ToTensor(original_img)
        # from pil image to tensor, do not normalize image

        img, traget = data_transform(original_img, {})
        # expand batch dimension
        if num_now == 0:
            img_batch = torch.unsqueeze(img, dim=0)
            filename_list.append(imgname)
            num_now += 1
        elif num_now < batch_size:
            img = torch.unsqueeze(img, dim=0)
            shape = img.size()
            assert shape[2] == shape[3] == 800
            img_batch = torch.cat([img_batch, img], 0)
            filename_list.append(imgname)
            num_now += 1
        if num_now >= batch_size or id >= len(img_list) - 1:
            num_now = 0
        else:
            continue
        with torch.no_grad():
            predictions, _, p2_r, p2_c = model(img_batch.to(device))
            for one_id, pred in enumerate(p2_c):
                # get teds
                merge_boxs = []
                row_boxs = []
                col_boxs = []
                head_box = []
                pred = predictions[one_id]
                predict_boxes = pred["boxes"].to("cpu").numpy()
                predict_classes = pred["labels"].to("cpu").numpy()
                predict_scores = pred["scores"].to("cpu").numpy()

                for pid, label in enumerate(predict_classes.tolist()):
                    if label == 1 and predict_scores[pid] >= box_thresh:
                        merge_boxs.append(predict_boxes[pid])
                    elif label == 2 and predict_scores[pid] >= box_thresh:
                        head_box.append(predict_boxes[pid])
                    elif label == 3 and predict_scores[pid] >= box_thresh:
                        row_boxs.append(predict_boxes[pid])
                    elif label == 4 and predict_scores[pid] >= box_thresh:
                        col_boxs.append(predict_boxes[pid])
                row = (255 * p2_r[one_id]).to('cpu', torch.int16).detach().numpy()
                col = (255 * p2_c[one_id]).to('cpu', torch.int16).detach().numpy()
                score, point_row, point_col = line_post.Get_score_out(thr_row=row, thr_col=col, merge_boxs=merge_boxs,
                                                                      filename=filename_list[one_id], imgshape=[],
                                                                      head_box=head_box)
                score_list.append(score)
                if len(score_list) % 100 == 1:
                    print('ave_score:', sum(score_list) / len(score_list))
    print('ave_score:', sum(score_list) / len(score_list))
    return  sum(score_list) / len(score_list)
if __name__ == '__main__':
    main()
