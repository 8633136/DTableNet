import os
import json
import pickle
import numpy as np
def Trans_TableMaster(ori_html,point_row,point_col,merge_data):
    # change my model result into tablemaster:structure.pkl
    bbox = []
    score = 0.99
    text = ""
    merge_data = np.array(merge_data)
    start_row=merge_data[:,0]
    end_row=merge_data[:,1]
    start_col=merge_data[:,2]
    end_col = merge_data[:,3]
    for num_row in range(0,len(point_row)):
        for num_col in range(0,len(point_col)):
            pass


if __name__ == '__main__':
    pkl_file_TM_ori_struct = r"/home/li/Desktop/disk2/wy/TableRe/TableMASTER-mmocr/Results/725_pubtabnet_all/structure_val_result/structure_master_results.pkl"
    pkl_file = r"/home/li/Desktop/disk2/wy/TableRe/TableMASTER-mmocr/Results/709_dence0930/end2end_val_result/end2end_results.pkl"
    pkl_file = r"./low_score_train_convt_all_epoch31_struct_save/struct_TM.pkl"
    pkl_file_struct_epoch36 = r"./epoch36_no_use/struct_TM_epoch36_2.pkl"
    pkl_file_struct_epoch36 = r"./A811_epoch36_save_pkl/struct_TM_epoch36.pkl"
    pkl_file = r"/home/li/Desktop/disk2/wy/TableRe/TableMASTER-mmocr/Results/725_pubtabnet_all/final_results.pkl"
    pkl_file = r"/home/li/Desktop/disk2/wy/TableRe/TableMASTER-mmocr/Results/725_pubtabnet_all/structure_val_result/structure_master_results.pkl"
    pkl_file = r"/home/li/Desktop/disk2/wy/TableRe/TableMASTER-mmocr/Results/811_epoch31/final_results.pkl"
    pkl_file_endt2end = r"//home/li/Desktop/disk2/wy/TableRe/TableMASTER-mmocr/Results/725_pubtabnet_all/end2end_val_result/end2end_results.pkl"
    pkl_file = r"//home/li/Desktop/disk2/wy/TableRe/TableMASTER-mmocr/Results/811_epoch36_bs16_all/final_results.pkl"
    pkl_TM_final=r"/home/li/Desktop/disk2/wy/TableRe/TableMASTER-mmocr/Results/725_pubtabnet_all/final_results.pkl"
    final_pkl = r"/home/li/Desktop/disk2/wy/TableRe/TableMASTER-mmocr/Results/813_epoch36_bs16_2/final_results_4.pkl"
    epoch_28_pkl = r"/home/li/Desktop/disk2/wy/TableRe/TableMASTER-mmocr/Results/815_epoch28/final_results_4.pkl"
    data = pickle.load(open(pkl_file, 'rb'))
    data_text = pickle.load(open(pkl_file_endt2end, 'rb'))
    data_TM_ori_struct = pickle.load(open(pkl_file_TM_ori_struct, 'rb'))
    data_epoch36 = pickle.load(open(pkl_file_struct_epoch36, 'rb'))
    data_TM_final = pickle.load(open(pkl_TM_final, 'rb'))
    data_final = pickle.load(open(final_pkl, 'rb'))
    data_epoch28 = pickle.load(open(epoch_28_pkl, 'rb'))
    for key in data_final.keys():
        if "Country" in data_final[key] and "USA" in data_final[key] and "2005" in data_final[key]:
            print(key)
    # for key in data.keys():
    #     bbox = data[key]['bbox']
    #     bbox = bbox[0]+bbox[1]
    #     bbox = np.array(bbox).reshape(-1,4)
    print(1)
    #PMC2559988_012_00.png
    #PMC1919361_006_00.png
    #PMC1821033_006_01.png
    #PMC1637095_009_00.png
    #PMC2755009_002_00.png