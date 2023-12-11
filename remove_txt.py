import os
import shutil
import tqdm
"""
从训练集中提取得分不好的数据，组成小数据集
"""
if __name__ == "__main__":
    little_train_txt = r"/home/li/Desktop/disk2/datasets/pubtabnet/voc_pubtabnet_zpeight/train.txt"
    f = open(little_train_txt,mode='r',encoding='utf-8')
    train_img_data = f.readlines()
    f.close()
    train_img_data = tqdm.tqdm(train_img_data)
    save_data = []
    for data in train_img_data:
        data = data.replace('\n','')
        data = data[:-4]
        save_data.append(data)
    f = open(little_train_txt, mode='w', encoding='utf-8')
    f.write('\n'.join(save_data))
    f.close()