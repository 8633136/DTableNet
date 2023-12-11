import os
import json
import shutil
import tqdm
def move_dence_img(dence_data,dence_thr,ori_img_folder,save_folder,Save_no_dence=True):
    if Save_no_dence:
        dence_path = os.path.join(save_folder,"dence")
        no_dence_path = os.path.join(save_folder,'no_dence')
        if not os.path.exists(no_dence_path):
            os.makedirs(no_dence_path)
    else:
        dence_path = save_folder
    if not os.path.exists(dence_path):
        os.makedirs(dence_path)
    dence_data = tqdm.tqdm(dence_data)
    for data in dence_data:
        if data['dence_rate']>=dence_thr:
            ori_path = os.path.join(ori_img_folder,data['imgname'])
            den_path = os.path.join(dence_path,data['imgname'])
            shutil.copyfile(ori_path,den_path)
        else:
            if Save_no_dence:
                ori_path = os.path.join(ori_img_folder, data['imgname'])
                den_path = os.path.join(no_dence_path, data['imgname'])
                shutil.copyfile(ori_path, den_path)
if __name__ == '__main__':
    dence_jsonfile = r"./dence_data/val_dence_data_no_800.json"
    dence_thr = 4.5
    ori_img_folder = r"/home/li/Desktop/disk2/datasets/pubtabnet/pubtabnet_TSRNet_0704/val"
    save_folder = r"/home/li/Desktop/disk2/datasets/pubtabnet/dence_img_4d5_no800"
    f = open(dence_jsonfile, "r")
    dence_data = json.load(f)
    f.close()
    move_dence_img(dence_data=dence_data,dence_thr=dence_thr,
                   ori_img_folder=ori_img_folder,save_folder=save_folder,Save_no_dence=True
                   )
