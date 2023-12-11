import torch
import torch.utils.data
import torch.optim as optim
from SCNN_Network import Network
# import config
from balance_bce_loss import balance_bce_loss,balance_bce_loss_m
import numpy as np
# from Datasets.pubtabnet_dataset import Buliddataset
from torch.utils.data import DataLoader


def train(epoch, device: str, lr):

    # datasetbulider = Buliddataset(cfg=config.config)
    # dataset = datasetbulider()
    # train_DataLoader = DataLoader(dataset=dataset,
    #                               collate_fn=datasetbulider.collate_fn2,
    #                               shuffle=True,
    #                               batch_size=6, num_workers=0)

    device=torch.device(device)
    net = Network()
    net=net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    net.train()
    rloss=[]
    closs=[]
    mloss=[]
    totalloss=[]
    epochlist = []
    for epoch in range(epoch):  # 迭代次数
        epochlist.append(epoch+1)
        loss_r_list = []
        loss_c_list = []
        loss_m_list = []
        losslist = []
        num=0
        num_list=[]
        for traindata in train_DataLoader:
            images, labels, mask = traindata
            images=images.to(device)
            labels_r=labels['map_no_merge_row'].to(device)
            labels_c=labels['map_no_merge_col'].to(device)
            labels_m=labels['map_merge'].to(device)
            mask_r=mask['row'].to(device)
            mask_c=mask['col'].to(device)
            mask_m=mask['merge'].to(device)

            p2_r, p2_c, p2_m = net(images)

            # p2_r = torch.sigmoid(p2_r).to(device)
            # p2_c = torch.sigmoid(p2_c).to(device)

            # p2_m = torch.sigmoid(p2_m).to(device)


            loss_r = balance_bce_loss(p2_r, labels_r, mask_r).to(device)
            loss_c = balance_bce_loss(p2_c, labels_c, mask_c).to(device)
            loss_m = balance_bce_loss_m(p2_m, labels_m, mask_m).to(device)
            loss = loss_r + loss_c + loss_m

            loss_r_list.append(loss_r.item())
            loss_c_list.append(loss_c.item())
            loss_m_list.append(loss_m.item())
            losslist.append(loss.item())

            optimizer.zero_grad()  # 优化器梯度清零

            loss.backward()
            # loss_r2.backward()
            # print('grad',loss.grad())
            optimizer.step()  # 权重更新
            torch.save(net.state_dict(), 'D:\PychramProject\Robus_Tab_Net_1\model/model.pth')
            torch.save(optimizer.state_dict(), 'D:\PychramProject\Robus_Tab_Net_1\model/optimizer.pth')
            print('num:',num,'rloss:',loss_r.item(),'closs:',loss_c.item(),'mloss:',loss_m.item(),'loss:',loss.item())
            num_list.append(num+1)
            num += 1
            np.save('D:\PychramProject\Robus_Tab_Net_1/num_loss_data/rloss.npy', loss_r_list)
            np.save('D:\PychramProject\Robus_Tab_Net_1/num_loss_data/closs.npy', loss_c_list)
            np.save('D:\PychramProject\Robus_Tab_Net_1/num_loss_data/mloss.npy', loss_m_list)
            np.save('D:\PychramProject\Robus_Tab_Net_1/num_loss_data/total_loss.npy',losslist)
            np.save('D:\PychramProject\Robus_Tab_Net_1/num_loss_data/num.npy', num_list)
        rloss.append(sum(loss_r_list))
        closs.append(sum(loss_c_list))
        mloss.append(sum(loss_m_list))
        totalloss.append(sum(losslist))
        print('epoch:', epoch+1, 'rloss:', sum(loss_r_list), 'closs:', sum(loss_c_list), 'mloss:', sum(loss_m_list), 'total_loss:', sum(losslist))

    np.save('D:\PychramProject\Robus_Tab_Net_1\epoch_loss_data/rloss.npy', rloss)
    np.save('D:\PychramProject\Robus_Tab_Net_1\epoch_loss_data/closs.npy', closs)
    np.save('D:\PychramProject\Robus_Tab_Net_1\epoch_loss_data/mloss.npy', mloss)
    np.save('D:\PychramProject\Robus_Tab_Net_1\epoch_loss_data/total_loss.npy', totalloss)
    np.save('D:\PychramProject\Robus_Tab_Net_1\epoch_loss_data/epoch.npy', epochlist)
    return 0



if __name__ == '__main__':
    train(1,'cuda:0',0.0001)
