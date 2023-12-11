import os
if __name__ == '__main__':
    txt_file = r"/home/li/Desktop/disk2/datasets/pubtabnet/voc_pubtabnet_dence/val_ori.txt"
    save_txt_file = r"/home/li/Desktop/disk2/datasets/pubtabnet/voc_pubtabnet_dence/val.txt"
    img_folder = r"/home/li/Desktop/disk2/datasets/pubtabnet/voc_pubtabnet_dence/JPEGImgages"
    img_list = os.listdir(img_folder)
    f = open(txt_file,'r')
    val_data = f.read()
    f.close()
    val_data = val_data.split("\n")
    for id,img_name in enumerate(img_list):
        img_list[id] = img_name.split('.')[0]
    new_val_data = []
    for id,img_name in enumerate(img_list):
        if img_name in val_data:
            new_val_data.append(img_name)
    print(len(new_val_data))
    f = open(save_txt_file,'w')
    f.write('\n'.join(new_val_data))
    f.close()