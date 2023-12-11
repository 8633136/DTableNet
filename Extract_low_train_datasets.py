import os
import shutil
import tqdm
"""
从训练集中提取得分不好的数据，组成小数据集
"""
if __name__ == "__main__":
    little_train_txt = r"/home/li/Desktop/disk2/wy/TableRe/FsrcnnNet_class_2_50/low_score_all_for_train/low_score.txt"
    big_dataset_root_path = r"/home/li/Desktop/disk2/datasets/pubtabnet/voc_pubtabnet_new2"
    big_dataset_train_txt = os.path.join(big_dataset_root_path,'train.txt')
    save_little_dataset_root_path = r"/home/li/Desktop/disk2/datasets/pubtabnet/voc_pubtabnet_low_all"
    ori_img_path = os.path.join(big_dataset_root_path,'JPEGImgages')
    ori_anno_path = os.path.join(big_dataset_root_path,'annotations')
    save_img_path = os.path.join(save_little_dataset_root_path,'JPEGImgages')
    save_anno_path = os.path.join(save_little_dataset_root_path,'annotations')
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    if not os.path.exists(save_anno_path):
        os.makedirs(save_anno_path)
    f = open(little_train_txt,mode='r')
    traintxt = f.readlines()
    f.close()
    f = open(big_dataset_train_txt,mode='r')
    ori_train_list = f.readlines()
    f.close()
    # ori_train_list = ori_train_list.split('\n')
    # ori_train_list = tqdm.tqdm(ori_train_list)
    for id,train_data in enumerate(ori_train_list):
        ori_train_list[id]=ori_train_list[id].replace('\n','')
    # train_name_list = traintxt.split('\n')
    train_name_list = tqdm.tqdm(traintxt)
    for name in train_name_list:
        name = name.replace('\n','')
        file_name = name.replace('.png','')
        file_name = file_name+'.xml'
        ori_anno_file = os.path.join(ori_anno_path,file_name)
        if not os.path.exists(ori_anno_file) or name not in ori_train_list:
            continue
        den_anno_file = os.path.join(save_anno_path,file_name)
        shutil.copyfile(ori_anno_file,den_anno_file)

        file_name = file_name.replace('.xml','.png')
        ori_img_file = os.path.join(ori_img_path,file_name)
        den_img_file = os.path.join(save_img_path,file_name)
        shutil.copyfile(ori_img_file,den_img_file)