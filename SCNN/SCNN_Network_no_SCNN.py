import torch
import torch.nn as nn
from .Row_or_col import S_Row,S_Col
from .SCNN2_1x1600_v4_no_SCNN import R_SCNN,C_SCNN

class Network(nn.Module):
    def __init__(self,):
        super(Network,self).__init__()
        self.s_row=S_Row()
        self.s_col=S_Col()
        self.r_scnn=R_SCNN()
        self.c_scnn=C_SCNN()

    def forward(self,p2):

        p2_r=self.s_row(p2)
        p2_r=self.r_scnn(p2_r)

        p2_c = self.s_col(p2)
        p2_c = self.c_scnn(p2_c)

        return p2_r,p2_c