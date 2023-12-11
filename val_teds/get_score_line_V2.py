import json
# from Postprocessor.DB_Postprocessor import DBPostprocessor
import numpy as np
import os
import time
import pickle
from val_teds.src.metric import TEDS
from val_teds.src.mmocr_teds_acc_mp import singleEvaluation
import cv2
import math
import re
import scipy.signal as signal
# from mui_thread import MyThread


class Postprocess():
    def __init__(self, thr_folder, row_thr, col_thr,
                 DBPost_merge, teds, gt_val_file, save_file,
                 min_width_row=8,min_width_col=8, iou=0.6,min_head_width=2,
                 head_thr=0.7):
        self.thr_file = thr_folder
        self.row_thr = row_thr
        self.col_thr = col_thr
        self.DBPost_merge = DBPost_merge
        self.iou = iou
        self.data = []
        self.teds = teds
        self.score = []  # 所有的得分
        self.min_width_row = min_width_row
        self.min_width_col = min_width_col
        self.min_head_width = min_head_width
        self.head_thr = head_thr
        f = open(gt_val_file, 'r', encoding='utf-8')
        self.gt_dict = json.load(f)
        self.save_file = save_file
        self.save_dict = []

    def init(self):
        self.data = []

    def read_file(self):
        self.all_pred_dict = os.listdir(self.thr_file)
        # f = open(self.thr_file,encoding='utf-8',mode='r')
        # thr_data = json.load(f)
        # self.thr_data = thr_data

    def Calcu_html(self, row_line=None, col_line=None, merge_boxs=None, imgshape=None):
        # 得到行列坐标框
        thr_row = np.array(row_line, dtype=np.float32) / 255
        thr_col = np.array(col_line, dtype=np.float32) / 255
        # thr_merge = np.array(thr_merge, dtype=np.float32) / 255
        # self.row_box = self.row_thr(thr_row)
        # self.col_box = self.col_thr(thr_col)
        # self.merge_box = self.DBPost_merge(thr_merge)
        # self.merge_box = []
        for id,mbox in enumerate(merge_boxs):
            mbox = mbox.reshape(2,2).tolist()
            box_8point = [mbox[0][0],mbox[0][1],mbox[1][0],mbox[0][1],
                          mbox[1][0],mbox[1][1],mbox[0][0],mbox[1][1]]
            merge_boxs[id]=box_8point
        self.merge_box = merge_boxs
        for data in self.merge_box:
            # self.data.append({'merge_box':data[0:8]})
            self.data.append({'merge_box': data[0:8]})
        # point_row,point_col = self.get_crpoint(row_box=self.row_box,col_box=self.col_box,imgshape=imgshape)
        # point_row,point_col = self.get_crpoint_by_thr_ave(thr_row=row_line,thr_col=col_line)
        point_row, point_col = self.get_crpoint_line(row_line=row_line, col_line=col_line)
        point_row = (np.array(point_row) / 2).tolist()
        point_col = (np.array(point_col) / 2).tolist()
        # point_row,point_col = self.get_crpoint_line_by_signal(row_line=row_line,col_line=col_line)
        self.point_row = point_row
        self.point_col = point_col
        if len(self.point_col) == 0 or len(self.point_row) == 0:
            return 0
        merge_data = self.get_merge_cr_V2(point_row, point_col, self.merge_box)  # 获得合并信息，未进行iou判断
        # if len(merge_data)!=0:
        #     self.data['merge_data'] = merge_data
        # else:
        #     self.data['merge_data'] = []
        for id, data in enumerate(merge_data):
            self.data[id]['merge_data'] = data
        self.mergedata = merge_data

    def get_crpoint(self, row_box, col_box, imgshape):
        """
           row_box:n*4*2
           """
        point_row = []
        point_col = []
        h, w, _ = imgshape
        row_box = np.clip(row_box, 0, h)
        col_box = np.clip(col_box, 0, w)
        row_box = row_box.astype(np.int32)[:, 0:8]
        col_box = col_box.astype(np.int32)[:, 0:8]
        row_box = row_box[:, 0:8]
        col_box = col_box[:, 0:8]
        row_box = row_box.reshape(-1, 4, 2)
        col_box = col_box.reshape(-1, 4, 2)
        Drop_row = np.zeros([1, h], dtype=np.int32)
        Drop_col = np.zeros([1, w], dtype=np.int32)
        for box in row_box:
            up = max(box[0][1], box[1][1])
            down = min(box[2][1], box[3][1])
            if up > down:
                tem = up
                up = down
                down = tem
            Drop_row[0, up:down] = 1
        for box in col_box:
            up = max(box[0][0], box[1][0])
            down = min(box[2][0], box[3][0])
            if up > down:
                tem = up
                up = down
                down = tem
            Drop_col[0, up:down] = 1
        _, id_row = np.where(Drop_row > 0)
        _, id_col = np.where(Drop_col > 0)
        row_num = 0
        col_num = 0
        start = -1
        for id in range(1, len(id_row)):
            if id_row[id - 1] + 1 == id_row[id] and id_row[id]:
                start = id_row[id - 1]
            if id_row[id - 1] + 1 != id_row[id] and start != -1 and id_row[id] - start >= 8:
                row_num += 1
                point_row.append([start, id_row[id]])
                start = id_row[id]
        start = -1
        for id in range(1, len(id_col)):
            if id_col[id - 1] + 1 == id_col[id] and id_col[id]:
                start = id_col[id - 1]
            if id_col[id - 1] + 1 != id_col[id] and start != -1 and id_col[id] - start >= 8:
                col_num += 1
                point_col.append([start, id_col[id]])
                start = id_col[id]
        # for id in range(1, len(id_row)):
        #     if id_row[id - 1] + 1 != id_row[id] and id_row[id] - id_row[id - 1] >= 3:
        #         row_num += 1
        #         point_row.append([id_row[id-1],id_row[id]])
        # for id in range(1, len(id_col)):
        #     if id_col[id - 1] + 1 != id_col[id] and id_col[id] - id_col[id - 1] >= 3:
        #         col_num += 1
        #         point_col.append([id_col[id - 1],id_col[id]])
        return point_row, point_col

    def get_crpoint_line(self, row_line, col_line):
        point_row = []
        point_col = []
        Drop_row = np.array(row_line, dtype=np.uint16).reshape(1, -1)
        Drop_col = np.array(col_line, dtype=np.uint16).reshape(1, -1)
        _, Drop_row = cv2.threshold(Drop_row, self.row_thr * 255, 255, cv2.THRESH_BINARY)
        _, Drop_col = cv2.threshold(Drop_col, self.col_thr * 255, 255, cv2.THRESH_BINARY)
        _, id_row = np.where(Drop_row > 0)
        _, id_col = np.where(Drop_col > 0)
        row_num = 0
        col_num = 0
        start = -1
        Drop_row = np.clip(Drop_row, 0, 1)
        str_row = Drop_row.tolist()[0]
        str_row = [str(i) for i in str_row]
        str_row = ''.join(str_row)

        Drop_col = np.clip(Drop_col, 0, 1)
        str_col = Drop_col.tolist()[0]
        str_col = [str(i) for i in str_col]
        str_col = ''.join(str_col)
        # list_str = re.split('0*0',str_row)
        list_str = str_row.split('0')
        list_row = []
        begin = 0
        for strs in list_str:
            if len(strs) >= self.min_width_row:
                list_row.append([begin, len(strs) + begin - 1])
                begin += len(strs) + 1
            elif len(strs) > 0 and len(strs) < self.min_width_row:
                begin += len(strs) + 1
            else:
                begin += 1
        list_str = str_col.split('0')
        list_col = []
        begin = 0
        for strs in list_str:
            if len(strs) >= self.min_width_col:
                list_col.append([begin, len(strs) + begin - 1])
                begin += len(strs) + 1
            elif len(strs) > 0 and len(strs) < self.min_width_col:
                begin += len(strs) + 1
            else:
                begin += 1
        print('num_row:', len(list_row), "num_col:", len(list_col))
        return list_row, list_col
        for id in range(1, len(id_row)):
            if id_row[id - 1] + 1 == id_row[id] and id_row[id]:
                start = id_row[id - 1]
            if id_row[id - 1] + 1 != id_row[id] and start != -1 and id_row[id] - start >= 4:
                row_num += 1
                point_row.append([start, id_row[id]])
                start = id_row[id]
        start = -1
        for id in range(1, len(id_col)):
            if id_col[id - 1] + 1 == id_col[id] and id_col[id]:
                start = id_col[id - 1]
            if id_col[id - 1] + 1 != id_col[id] and start != -1 and id_col[id] - start >= 4:
                col_num += 1
                point_col.append([start, id_col[id]])
                start = id_col[id]

        return point_row, point_col

    def get_crpoint_line_by_signal(self, row_line, col_line):
        point_row = []
        point_col = []
        Drop_row = np.array(row_line, dtype=np.uint16).reshape(1, 800)
        Drop_col = np.array(col_line, dtype=np.uint16).reshape(1, 800)
        row_max = signal.argrelextrema(Drop_row, np.greater)
        col_max = signal.argrelextrema(Drop_col, np.greater)

        _, Drop_row = cv2.threshold(Drop_row, self.row_thr * 255, 255, cv2.THRESH_BINARY)
        _, Drop_col = cv2.threshold(Drop_col, self.col_thr * 255, 255, cv2.THRESH_BINARY)
        _, id_row = np.where(Drop_row > 0)
        _, id_col = np.where(Drop_col > 0)
        row_num = 0
        col_num = 0
        start = -1
        Drop_row = np.clip(Drop_row, 0, 1)
        str_row = Drop_row.tolist()[0]
        str_row = [str(i) for i in str_row]
        str_row = ''.join(str_row)

        Drop_col = np.clip(Drop_col, 0, 1)
        str_col = Drop_col.tolist()[0]
        str_col = [str(i) for i in str_col]
        str_col = ''.join(str_col)
        # list_str = re.split('0*0',str_row)
        list_str = str_row.split('0')
        list_row = []
        begin = 0
        for strs in list_str:
            if len(strs) >= self.min_width:
                list_row.append([begin, len(strs) + begin - 1])
                begin += len(strs) + 1
            elif len(strs) > 0 and len(strs) < self.min_width:
                begin += len(strs) + 1
            else:
                begin += 1
        list_str = str_col.split('0')
        list_col = []
        begin = 0
        for strs in list_str:
            if len(strs) >= self.min_width:
                list_col.append([begin, len(strs) + begin - 1])
                begin += len(strs) + 1
            elif len(strs) > 0 and len(strs) < self.min_width:
                begin += len(strs) + 1
            else:
                begin += 1
        print('num_row:', len(list_row), "num_col:", len(list_col))
        return list_row, list_col

    def get_crpoint_by_thr_ave(self, thr_row, thr_col):
        ori_thr_row = np.array(thr_row, dtype=np.uint8)
        ori_thr_col = np.array(thr_col, dtype=np.uint8)
        thr_row = np.array(thr_row, dtype=np.uint8)[:, 32:96]
        thr_col = np.array(thr_col, dtype=np.uint8)[32:96, :]
        ave_row = thr_row.sum(axis=1) / 64
        ave_col = thr_col.sum(axis=0) / 64
        # ave_col = cv2.resize(thr_col, dsize=(1024,1),interpolation=cv2.INTER_LINEAR)
        # ave_row = cv2.resize(thr_row, dsize=(1,1024),interpolation=cv2.INTER_LINEAR)
        Drop_row = np.array(ave_row, dtype=np.uint16).reshape(1, 1024)
        Drop_col = np.array(ave_col, dtype=np.uint16).reshape(1, 1024)
        # _, Drop_row = cv2.threshold(Drop_row, self.row_thr * Drop_row.max(), 255, cv2.THRESH_BINARY)
        # _, Drop_col = cv2.threshold(Drop_col, self.col_thr * Drop_col.max(), 255, cv2.THRESH_BINARY)
        _, Drop_row = cv2.threshold(Drop_row, self.row_thr * Drop_row.max(), 255, cv2.THRESH_BINARY)
        _, Drop_col = cv2.threshold(Drop_col, self.col_thr * Drop_col.max(), 255, cv2.THRESH_BINARY)
        ori_Drop_row = Drop_row
        ori_Drop_col = Drop_col
        _, id_row = np.where(Drop_row > 0)
        _, id_col = np.where(Drop_col > 0)
        row_num = 0
        col_num = 0
        start = -1
        Drop_row = np.clip(Drop_row, 0, 1)
        str_row = Drop_row.tolist()[0]
        str_row = [str(i) for i in str_row]
        str_row = ''.join(str_row)

        Drop_col = np.clip(Drop_col, 0, 1)
        str_col = Drop_col.tolist()[0]
        str_col = [str(i) for i in str_col]
        str_col = ''.join(str_col)
        # list_str = re.split('0*0',str_row)
        list_str = str_row.split('0')
        list_row = []
        begin = 0
        for strs in list_str:
            if len(strs) >= self.min_width:
                list_row.append([begin, len(strs) + begin - 1])
                begin += len(strs) + 1
            elif len(strs) > 0 and len(strs) < self.min_width:
                begin += len(strs) + 1
            else:
                begin += 1
        list_str = str_col.split('0')
        list_col = []
        begin = 0
        for strs in list_str:
            if len(strs) >= self.min_width:
                list_col.append([begin, len(strs) + begin - 1])
                begin += len(strs) + 1
            elif len(strs) > 0 and len(strs) < self.min_width:
                begin += len(strs) + 1
            else:
                begin += 1
        print('num_row:', len(list_row), "num_col:", len(list_col))
        # cv2.imshow("row",ori_Drop_row.reshape(1,1024))
        # cv2.imshow("col",ori_Drop_col.reshape(1,1024))
        # cv2.waitKey(0)
        return list_row, list_col

    def get_crpoint_by_str(self, row_line, col_line):
        point_row = []
        point_col = []
        Drop_row = np.array(row_line, dtype=np.uint16).reshape(1, 800)
        Drop_col = np.array(col_line, dtype=np.uint16).reshape(1, 800)
        _, Drop_row = cv2.threshold(Drop_row, self.row_thr * 255, 255, cv2.THRESH_BINARY)
        _, Drop_col = cv2.threshold(Drop_col, self.col_thr * 255, 255, cv2.THRESH_BINARY)
        _, id_row = np.where(Drop_row > 0)
        _, id_col = np.where(Drop_col > 0)
        row_num = 0
        col_num = 0
        start = -1
        for id in range(1, len(id_row)):
            if id_row[id - 1] + 1 == id_row[id] and id_row[id]:
                start = id_row[id - 1]
            if id_row[id - 1] + 1 != id_row[id] and start != -1 and id_row[id] - start >= 4:
                row_num += 1
                point_row.append([start, id_row[id]])
                start = id_row[id]
        start = -1
        for id in range(1, len(id_col)):
            if id_col[id - 1] + 1 == id_col[id] and id_col[id]:
                start = id_col[id - 1]
            if id_col[id - 1] + 1 != id_col[id] and start != -1 and id_col[id] - start >= 4:
                col_num += 1
                point_col.append([start, id_col[id]])
                start = id_col[id]
        return point_row, point_col

    def get_merge_cr(self, point_row, point_col, merge_box):
        """
        merge_box:n*4*2
        point_row:n*2
        """
        if len(merge_box) == 0:
            return []
        merge_box = np.array(merge_box)
        merge_box = np.array(merge_box[:, 0:8]).reshape(-1, 4, 2)
        point_row = np.array(point_row)
        point_col = np.array(point_col)
        mergedata = []  # start_row,end_row,start_col,end_col
        for merge in merge_box:
            minx = min(merge[:, 0])
            maxx = max(merge[:, 0])
            miny = min(merge[:, 1])
            maxy = max(merge[:, 1])
            index = np.where(point_col[:, 0] < minx)[0]
            if len(index) > 0:
                start_col = index[-1]
            else:
                start_col = 0
            index = np.where(point_col[:, 1] > maxx)[0]
            end_col = index[0] if len(index) > 0 else len(point_col) - 1
            index = np.where(point_row[:, 0] < miny)[0]
            start_row = index[-1] if len(index) > 0 else 0
            index = np.where(point_row[:, 1] > maxy)[0]
            end_row = index[0] if len(index) > 0 else len(point_row) - 1
            mergedata.append([start_row, end_row, start_col, end_col])
        return mergedata

    def get_merge_cr_V2(self, point_row, point_col, merge_box):
        """
        merge_box:n*4*2
        point_row:n*2
        """
        if len(merge_box) == 0:
            return []
        merge_box = np.array(merge_box)
        merge_box = np.array(merge_box[:, 0:8]).reshape(-1, 4, 2)
        ave_row = []
        for id,p_row in enumerate(point_row):
            if id ==0:
                # ave_row.append(p_row[0])
                pass
            # elif id == len(point_row)-1:
            #     # ave_row.append(p_row[1])
            #     pass
            else:
                ave_row.append((p_row[0]+point_row[id-1][1])/2)
        ave_col = []
        for id, p_col in enumerate(point_col):
            if id == 0:
                # ave_col.append(p_col[0])
                pass
            # elif id == len(point_col) - 1:
            #     # ave_col.append(p_col[1])
            #     pass
            else:
                ave_col.append((p_col[0] + point_col[id - 1][1]) / 2)
        point_row = ave_row
        point_col = ave_col
        point_row = np.array(point_row)
        point_col = np.array(point_col)
        # p_box_1 = point_row[:,0]
        mergedata = []  # start_row,end_row,start_col,end_col
        for merge in merge_box:
            minx = min(merge[:, 0])
            maxx = max(merge[:, 0])
            miny = min(merge[:, 1])
            maxy = max(merge[:, 1])

            index = np.where(point_col > minx)[0]
            if len(index) > 0:
                start_col = max(index[0],0)
            else:
                start_col = 0
            index = np.where(point_col < maxx)[0]
            end_col = min(index[-1]+1,len(point_col)) if len(index) > 0 else len(point_col)
            index = np.where(point_row > miny)[0]
            start_row = max(index[0],0) if len(index) > 0 else 0
            index = np.where(point_row < maxy)[0]
            end_row = min(index[-1]+1,len(point_row)) if len(index) > 0 else len(point_row)
            # if end_col-start_col <=1 and end_row - start_row <=1:
            #     pass
            # else:
            mergedata.append([start_row, end_row, start_col, end_col])
        return mergedata
    def get_real_merge(self):
        alldata = self.data
        for id, data in enumerate(alldata):
            index = data['merge_data'][0]
            up = self.point_row[index][0]
            index = data['merge_data'][1]
            down = self.point_row[index][1]
            index = data['merge_data'][2]
            left = self.point_col[index][0]
            index = data['merge_data'][3]
            right = self.point_col[index][1]
            alldata[id]['cr_box'] = [left, up, right, down]
        self.data = alldata
        alldata = self.data
        new_data = []
        for id, data in enumerate(alldata):
            merge = np.array(data['merge_box']).reshape(4, 2)
            minx = min(merge[:, 0])
            maxx = max(merge[:, 0])
            miny = min(merge[:, 1])
            maxy = max(merge[:, 1])
            iou = self.compute_IOU([minx, miny, maxx, maxy], data['cr_box'])
            if iou >= self.iou:
                data['iou'] = iou
                new_data.append(data)
        # self.iou_merge_data = new_data
        return new_data

    def generate_html(self):
        point_row = self.point_row
        point_col = self.point_col
        merge_data = self.iou_merge_data
        row_num = len(point_row)
        col_num = len(point_col)
        html_no_merge = self.Generte_html_no_merge(row_num, col_num, headline=-1)
        html = self.Generate_html_merge(merge_data, html_no_merge)
        self.thtml = html
        return html

    def Generte_html_no_merge(self, row_num, col_num, headline=-1):
        """
        headline:the num line of table head,-1:have no head,also have no body!
        return:
        Thtml(str):html of table (no merge)
        """
        td = [r'<td></td>']
        tr = '<tr>'
        tr_g = '</tr>'
        head = '<thead>'
        head_g = '</thead>'
        body = '<tbody>'
        body_g = '</tbody>'
        # if headline>0:
        thtml = []
        for idr in range(0, row_num - 1):
            if headline - 1 == idr:
                thtml.append(head)
            # tline =[tr]
            # tline.append(td*col_num)
            # tline.append(tr_g)
            tline = td * (col_num - 1)
            # thtml.append(tr)
            # thtml = thtml + td * col_num
            # thtml.append(tr_g)
            thtml.append(tline)
            if headline == idr:
                thtml.append(head_g)
                thtml.append(body)
        if headline > 0:
            thtml.append(body_g)
        if thtml == []:
            return ''
        else:
            return thtml

    def Generate_html_merge(self, merge_data, html_no_merge):
        """
        计算合并问题，并还原真正的表格结构
        html_no_merge(list:num_row*num_col):一行：:td/td,td/td......
        """
        len_col = len(self.point_col)
        len_row = len(self.point_row)
        for id, merge in enumerate(merge_data):
            merge_cr = merge['merge_data']
            for cid in range(merge_cr[2], merge_cr[3]):
                num_col = min(len_col - 1, cid)
                num_row = min(len_row - 1, merge_cr[0])
                html_no_merge[num_row][num_col] = -1
            for cid in range(merge_cr[2], merge_cr[3]):
                for rid in range(merge_cr[0], merge_cr[1]):
                    num_col = min(len_col - 1, cid)
                    num_row = min(len_row - 1, rid)
                    html_no_merge[num_row][num_col] = -1
            colspan = merge_cr[3] - merge_cr[2]
            rowspan = merge_cr[1] - merge_cr[0]
            if colspan > 1 and rowspan > 1:
                crdata = '<td ' + 'colspan="' + str(colspan) + '",rowspan="' + str(rowspan) + '">' + '</td>'
            elif rowspan > 1 and colspan <= 1:
                crdata = '<td ' + ',rowspan="' + str(rowspan) + '">' + '</td>'
            elif rowspan <= 1 and colspan > 1:
                crdata = '<td ' + 'colspan="' + str(colspan) + '">' + '</td>'
            else:
                continue
            html_no_merge[merge_cr[0]][merge_cr[2]] = crdata
        for rid, tline in enumerate(html_no_merge):
            for cid in range(len(tline)):
                if -1 in tline:
                    tline.remove(-1)
                else:
                    html_no_merge[rid] = tline
                    break
            # for cid,td in enumerate(tline):
            #     if td == -1:
            #         html_no_merge.pop([rid][cid])
        html_merge = html_no_merge
        return html_merge

    def compute_IOU(self, rec1, rec2):
        """
        计算两个矩形框的交并比。
        :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
        :param rec2: (x0,y0,x1,y1)
        :return: 交并比IOU.
        """
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])
        # 两矩形无相交区域的情况
        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return 0
        # 两矩形有相交区域的情况
        else:
            S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
            S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
            return S_cross / (S1 + S2 - S_cross)

    def Get_score(self, teds, pred, gt, file_name, delete_head=False, reshape_head=False):
        """
        reshape_head: not used
        """
        context = pred
        gt_context = gt
        # score = pool.apply_async(func=singleEvaluation, args=(teds, file_name, context, gt_context,))
        if delete_head:
            context = context.replace('<thead>', '')
            context = context.replace('</thead>', '')
            context = context.replace('<tbody>', '')
            context = context.replace('</tbody>', '')
            gt_context = gt_context.replace('<thead>', '')
            gt_context = gt_context.replace('</thead>', '')
            gt_context = gt_context.replace('<tbody>', '')
            gt_context = gt_context.replace('</tbody>', '')
        if reshape_head:
            id = gt_context.find('</thead>')
            num = gt_context[:id].count('<tr>')
        gt_context = gt_context.replace('colspan="6"','colspan=\"6\"')
        score = singleEvaluation(teds, file_name, context, gt_context)
        return score

    def __call__(self):
        self.read_file()
        # self.Calcu_html()
        all_pred_dict = self.all_pred_dict
        for id, thr_file in enumerate(all_pred_dict):
            self.init()  # 初始化，将变量置为空
            full_name = os.path.join(self.thr_file, thr_file)
            # print(full_name)
            f = open(full_name, encoding='utf-8', mode='r')
            thr = json.load(f)
            f.close()
            if len(thr) == 0:
                continue
            thr_row = thr['row'][0][0]
            thr_col = thr['col'][0][0]
            thr_merge = np.array(thr['merge'][0][0], dtype=np.uint16)
            thr_merge = cv2.resize(thr_merge, dsize=(0, 0), fy=2, fx=2, interpolation=cv2.INTER_AREA)
            if thr['filename'] == 'PMC1557538_002_01.png':
                # if thr['filename']=='PMC1557538_002_01.png':
                print(thr)
                # show_row = np.array(thr_row,dtype=np.uint8).reshape(1,-1)[0]
                # cv2.imshow(' row',show_row)
                # cv2.waitKey(0)
            self.Calcu_html(thr_row, thr_col, thr_merge, imgshape=thr['imgshape'])
            if len(self.point_col) == 0 or len(self.point_row) == 0:
                score = 0
            else:
                self.get_real_merge()  # 获得真实merge合并的框<做了iou判断>
                html = self.generate_html()  # 生成html代码 (list,row_num*col_num,基本组成:<td></td>),已经做了合并！
                gt = self.gt_dict[thr['filename']]
                """
                self.Add_head_body参数配置：
                    表头打在第一行：num=1 and used_num=True
                    模拟gt的行列：gt = self.gt_dict[thr['filename']]，
                    注意：used_num的优先级高于模拟gt的，当模拟gt行列和used_num同时存在时，不会模拟gt的行列
                    当no_merge为True时，便会根据gt行列 区别在于没有合并！(优先级最高！)
                self.Get_score参数配置：
                    取消表头表体：delete_head=True
                    不取消：delete_head=False
                常见配置：
                    1.正常计算模型的TEDS得分，表头为第1行-计算过程中不取消表头表体
                    num=1,used_num=True,no_merge=False,delete_head=False
                    2.计算和gt一样的行列结构，只是没有合并，表头和gt一样的地方
                    num=1，used_num=False,no_erge=True,gt=self.gt_dict[thr['filename']]
                    3.计算和gt一样的行列结构，只是没有合并，表头为第num行
                     num=1，used_num=True,no_erge=True,gt=self.gt_dict[thr['filename']]

                """
                html_teds = self.Add_head_body(html=html, num=1, gt=None, no_merge=False, use_num=True)

                score = self.Get_score(self.teds, pred=''.join(html_teds),
                                       gt=self.gt_dict[thr['filename']],
                                       file_name=thr['filename'],
                                       delete_head=False)
            self.score.append(score)
            print('id:', id, '  filename:', thr['filename'], '  score:', score)
            self.save_dict.append({thr['filename']: score})
            # self.show_thr_img(thr_col=thr_col,thr_row=thr_row,thr_merge=thr_merge,filename=thr['filename'])
            if id >= 10000:
                break
        ave_score = sum(self.score) / len(self.score)
        print('ave_score:', ave_score)
        f = open(self.save_file, encoding='utf-8', mode='w+')
        f.write(json.dumps(self.save_dict, indent=4, ensure_ascii=False))
        f.close()
        return ave_score, self.save_dict
    def Get_score_out(self,thr_row,thr_col,merge_boxs,filename,imgshape,head_box,head=None):
        # thr_merge = cv2.resize(thr_merge, dsize=(0, 0), fy=2, fx=2, interpolation=cv2.INTER_AREA)
        # if thr['filename'] == 'PMC1557538_002_01.png':
        #     # if thr['filename']=='PMC1557538_002_01.png':
        #     print(thr)
            # show_row = np.array(thr_row,dtype=np.uint8).reshape(1,-1)[0]
            # cv2.imshow(' row',show_row)
            # cv2.waitKey(0)
        self.init()
        self.Calcu_html(thr_row, thr_col, merge_boxs, imgshape=imgshape)
        headnum_out=0
        if len(self.point_col) <=1 or len(self.point_row) <=1:
            score = 0
        else:
            # self.iou_merge_data = self.get_real_merge()  # 获得真实merge合并的框<做了iou判断>
            self.iou_merge_data = self.data
            html = self.generate_html()  # 生成html代码 (list,row_num*col_num,基本组成:<td></td>),已经做了合并！
            gt = self.gt_dict[filename]
            """
            self.Add_head_body参数配置：
                表头打在第一行：num=1 and used_num=True
                模拟gt的行列：gt = self.gt_dict[thr['filename']]，
                注意：used_num的优先级高于模拟gt的，当模拟gt行列和used_num同时存在时，不会模拟gt的行列
                当no_merge为True时，便会根据gt行列 区别在于没有合并！(优先级最高！)
            self.Get_score参数配置：
                取消表头表体：delete_head=True
                不取消：delete_head=False
            常见配置：
                1.正常计算模型的TEDS得分，表头为第1行-计算过程中不取消表头表体
                num=1,used_num=True,no_merge=False,delete_head=False
                2.计算和gt一样的行列结构，只是没有合并，表头和gt一样的地方
                num=1，used_num=False,no_erge=True,gt=self.gt_dict[thr['filename']]
                3.计算和gt一样的行列结构，只是没有合并，表头为第num行
                 num=1，used_num=True,no_erge=True,gt=self.gt_dict[thr['filename']]

            """

            # get head num
            # head_num = self.get_head_num(self.point_row,head)
            head_num = self.get_head_num_from_box(head_box,self.point_row)
            headnum_out = head_num
            # head_num = 1
            # print('--------------------------------------fileanme:',filename,'--------------head_num:',head_num)
            html_teds = self.Add_head_body(html=html, num=head_num, gt=None, no_merge=False, use_num=True)
            # return 0, self.point_row, self.point_col, headnum_out, {}
            score = self.Get_score(self.teds, pred=''.join(html_teds),
                                   gt=self.gt_dict[filename],
                                   file_name=filename,
                                   delete_head=False)
        return_data = {
        'rows': self.point_row,
        'cols': self.point_col}
        return score,self.point_row,self.point_col,headnum_out,return_data

    def get_head_num_from_box(self, head_points, row_points):
        head_num = 1
        if len(head_points) < 1:
            return 1
        head_points = head_points[0].reshape(1, -1).tolist()[0]
        head_points = [head_points[1], head_points[3]]
        # for id,row in enumerate(row_points):
        for id in range(len(row_points) - 1):
            row = row_points[id]
            row_next = row_points[id + 1]
            ave = (row[1] + row_next[0]) / 2
            if head_points[1] < ave:
                head_num = id
                break
        if head_num > 5 or head_num < 1:
            head_num = 1
        return head_num
    # def Add_head_body(self,html,num=1,gt=None,no_merge=False,use_num=True):
    def Add_head_body(self, html, use_num=True, num=1, gt=None, no_merge=False):
        """
        num:表头结束插在num行的开始
        当gt存在时，表示要根据gt的变化更改插入表头的位置
        use_num ：为True时，不再使用gt的行列数目，为False不使用num
        no_merge:为True时，重新计算HTML，并且行列数目和gt的相同，区别在于没有合并！(优先级最高！)
        """
        html_teds = []
        for id, tline in enumerate(html):
            line = []
            if num >= 0 and id == 0 and use_num:
                html_teds.append('<thead>')
            if num == id and use_num:
                html_teds.append('</thead>')
                html_teds.append('<tbody>')
            # line.append('<tr>' + ''.join(tline) + '</tr>')
            html_teds.append('<tr>' + ''.join(tline) + '</tr>')
            if id == len(html) - 1 and num >= 0 and use_num:
                html_teds.append('</tbody>')
            # html_teds.append(line)
        if no_merge:
            tr_ids = []
            id = 0
            gt_tem = gt
            row_num = gt_tem.split('</tr>')
            num_row = len(row_num) - 1
            num_col = 0
            new_html = []
            for line in row_num:
                num_col = max(len(line.split('</td>')), num_col)
            num_col -= 1
            for id in range(num_row):
                new_html.append('<tr>' + ''.join('<td></td>' * num_col) + '</tr>')
            html_teds = new_html
        if gt != None and not use_num:
            html_teds.insert(0, '<thead>')
            id = gt.find('</thead>')
            num = gt[:id].count('<tr>')
            html_teds.insert(num + 1, '</thead>')
            html_teds.insert(num + 2, '<tbody>')
            html_teds.append('</tbody>')
        # elif use_num :
        #     html_teds.insert(0, '<thead>')
        #     html_teds.insert(num + 2, '</thead>')
        #     html_teds.insert(num + 3, '<tbody>')
        #     html_teds.append('</tbody>')
        return html_teds
    def get_head_num(self,row_point,thr_head):
        point_row = []
        Drop_row = np.array(thr_head, dtype=np.uint16).reshape(1, -1)
        _, Drop_row = cv2.threshold(Drop_row, self.head_thr * 255, 255, cv2.THRESH_BINARY)
        _, id_row = np.where(Drop_row > 0)
        row_num = 0
        start = -1
        Drop_row = np.clip(Drop_row, 0, 1)
        str_row = Drop_row.tolist()[0]
        str_row = [str(i) for i in str_row]
        str_row = ''.join(str_row)
        list_str = str_row.split('1')
        list_row = []
        begin = 0
        for strs in list_str:
            if len(strs) >= self.min_head_width:
                list_row.append([begin, len(strs) + begin - 1])
                begin += len(strs) + 1
            elif len(strs) > 0 and len(strs) < self.min_head_width:
                begin += len(strs) + 1
            else:
                begin += 1
        head_num = 1
        true_in_list_row=-1
        if len(list_row)>=1:
            for row in list_row:
                if row[0]*8>=row_point[1][1]:
                    true_in_list_row=row[0]*8
                    break
        else:
            return 1
        if true_in_list_row>0:
            for id,point in enumerate(row_point):
                if point[0]>=true_in_list_row:
                    head_num=id-1
                    break
        if head_num>=1:
            return head_num
        else:
            return 1
    def show_thr_img(self, thr_col, thr_row, thr_merge, filename):
        merge_data = self.data
        path = r"H:\datasets\pubtabnet\pubtabnet\val"
        imgpath = os.path.join(path, filename)
        img = cv2.imread(imgpath)
        h, w, _ = img.shape
        rate = min(800 / h, 800 / w)
        data_img = cv2.resize(img, dsize=(0, 0), fy=rate, fx=rate, interpolation=cv2.INTER_AREA)
        dh, dw, _ = data_img.shape
        thr_merge_cv = np.array(thr_merge, dtype=np.uint8)
        thr_col_cv = np.array(thr_col, dtype=np.uint8)
        # data_img = cv2.resize(img, dsize=(800, 800), fy=rate, fx=rate, interpolation=cv2.INTER_AREA)
        # Grayimg = cv2.cvtColor(imgs2, cv2.COLOR_BGR2GRAY)
        ret = cv2.copyMakeBorder(data_img, math.ceil((800 - dh) / 2), int((800 - dh) / 2), math.ceil((800 - dw) / 2),
                                 int((800 - dw) / 2), cv2.BORDER_CONSTANT, value=(128, 128, 128))
        # ret = data_img
        Grayimg = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
        imgs = ret
        for merge in merge_data:
            mbox = merge['merge_box']
            mbox = np.array(mbox, dtype=np.uint32).reshape(4, 2)
            minx = min(mbox[:, 0])
            maxx = max(mbox[:, 0])
            miny = min(mbox[:, 1])
            maxy = max(mbox[:, 1])
            imgs = cv2.rectangle(ret, (minx, miny), (maxx, maxy), (0, 0, 255), 2)
        for cbox in self.row_box:
            mbox = cbox
            mbox = np.array(mbox[:8], dtype=np.uint16).reshape(4, 2)
            minx = min(mbox[:, 0])
            maxx = max(mbox[:, 0])
            miny = min(mbox[:, 1])
            maxy = max(mbox[:, 1])
            # imgs = cv2.rectangle(ret, (minx, miny), (maxx, maxy), (0, 0, 255), 2)
        print('num_cow:', len(self.row_box))
        cv2.imshow("ori_img", imgs)
        cv2.imshow("thr_merge_cv", thr_merge_cv)
        cv2.waitKey(0)

    def reshape_thtml(self, pred, gt):
        pass


def Find_best_parameter(Save_folder=None):
    if Save_folder == None:
        Save_folder = './Save_Best'
    if not os.path.exists(Save_folder):
        os.makedirs(Save_folder)
    score_dict = []
    best_score = 0
    best_score_output = []
    mask_scope = np.arange(0.6, 0.8, 0.05, dtype=np.float16)
    score_scope = np.arange(0.5, 0.8, 0.05, dtype=np.float16)
    iou_scope = np.arange(0.6, 0.9, 0.05, dtype=np.float16)
    thr_file = r'E:\desktop\TableRe\thr_dict'
    gt_val_file = r'E:\desktop\TableRe\Pubtabnet\gtVal_1212.json'
    for mask in mask_scope:
        for score in score_scope:
            for iou in iou_scope:
                print('begin: ', 'mask:', mask, 'score:', score, 'iou:', iou)
                DBPost_row = DBPostprocessor(text_repr_type='quad', mask_thr=mask, min_text_score=score)
                DBPost_col = DBPostprocessor(text_repr_type='quad', mask_thr=mask, min_text_score=score)
                DBPost_merge = DBPostprocessor(text_repr_type='quad', mask_thr=mask, min_text_score=score)
                post = Postprocess(thr_file, DBPost_row, DBPost_col, DBPost_merge,
                                   teds=TEDS(n_jobs=1),
                                   gt_val_file=gt_val_file,
                                   save_file='./Val_data.json',
                                   iou=iou)
                ave_score, save_dict = post()
                if ave_score > best_score:
                    best_score = ave_score
                    best_score_output = save_dict
                score_dict.append({'mask': mask, 'score:': score, 'iou': iou, 'ave_score': ave_score})
                print('mask:', mask, 'score:', score, 'iou:', iou, 'ave_score:', ave_score)
    score_dict.append({'best_score': best_score})
    save1 = os.path.join(Save_folder, 'Paramter.json')
    save2 = os.path.join(Save_folder, 'Every_img.json')
    f = open(save1, 'w+')
    f.write(json.dumps(score_dict, indent=4, ensure_ascii=False))
    f.close()
    f = open(save2, 'w+')
    f.write(json.dumps(best_score_output, indent=4, ensure_ascii=False))
    f.close()


def Find_best_parameter_mui_thread(thr_file, Save_folder=None):
    if Save_folder == None:
        Save_folder = './Save_Best'
    if not os.path.exists(Save_folder):
        os.makedirs(Save_folder)
    score_dict = []
    best_score = 0
    best_score_output = []
    DBPost_merge = DBPostprocessor(text_repr_type='quad', mask_thr=0.5, min_text_score=0.5)
    mins = 0.2
    maxs = 0.8
    score_scope = np.arange(mins, maxs, 0.1, dtype=np.float16)
    min_width = np.arange(4, 8, 1, dtype=np.uint8)
    # thr_file = r'E:\desktop\TableRe\thr_dict'
    gt_val_file = r'E:\desktop\TableRe\Pubtabnet\gtVal_1212.json'
    mui_list = []
    config = []
    result = []
    best_score = 0
    best_config = ""
    for row_score in score_scope:
        for col_score in score_scope:
            for miw in min_width:
                tem_f = Postprocess(thr_file,
                                    row_thr=row_score, col_thr=col_score,
                                    DBPost_merge=DBPost_merge,
                                    teds=TEDS(n_jobs=1),
                                    gt_val_file=gt_val_file,
                                    min_width=miw,
                                    save_file='./Val_data.json',
                                    iou=0.7)
                tem_thread = MyThread(tem_f)
                mui_list.append(tem_thread)
                config.append("row：" + str(row_score) + " col:" + str(col_score) + " min_width:" + str(miw))

                if len(mui_list) == 15 or (row_score >= maxs and col_score >= maxs):
                    # 启动线程
                    for thread in mui_list:
                        thread.start()
                    # 等待线程执行完毕
                    for thread in mui_list:
                        thread.join()
                    # 获得并输出返回值
                    for id, thread in enumerate(mui_list):
                        ave_score, save_dict = thread.get_result()
                        config[id] = config[id] + " ave_score:" + str(ave_score)
                        if ave_score > best_score:
                            best_score = ave_score
                            best_config = config[id]
                        result.append(config[id])
                    # print(config)
                    print("best_config:", best_config)
                    # 清零
                    mui_list = []
                    config = []
    print("best_config:", best_config)
    save_dicts = {
        "best": best_config,
        "Paramter": result
    }
    save1 = os.path.join(Save_folder, 'save.json')
    f = open(save1, 'w+')
    f.write(json.dumps(save_dicts, indent=4, ensure_ascii=False))
    f.close()


if __name__ == '__main__':
    thr_file = r'E:\desktop\TableRe\thr_dict_58_1x800'
    Find_best_parameter_mui_thread(thr_file=thr_file, Save_folder='./Best_58_1x800')
    print('over!\n')
    exit(0)
    thr_file = r'E:\desktop\TableRe\thr_dict_58_1x800'
    gt_val_file = r'E:\desktop\TableRe\Pubtabnet\gtVal_1212.json'
    DBPost_row = DBPostprocessor(text_repr_type='quad', mask_thr=0.6, min_text_score=0.65)  # (0.6,0.65)
    DBPost_col = DBPostprocessor(text_repr_type='quad', mask_thr=0.6, min_text_score=0.65)
    row_thr = 0.2
    col_thr = 0.2
    DBPost_merge = DBPostprocessor(text_repr_type='quad', mask_thr=0.7, min_text_score=0.3)
    post = Postprocess(thr_file,
                       row_thr=row_thr, col_thr=col_thr,
                       DBPost_merge=DBPost_merge,
                       teds=TEDS(n_jobs=1),
                       gt_val_file=gt_val_file,
                       min_width=4,
                       save_file='./Val_data.json',
                       iou=0.7)
    post()