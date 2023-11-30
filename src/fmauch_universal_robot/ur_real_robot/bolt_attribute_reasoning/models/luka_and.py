import torch
import torch.nn as nn
from util.pos_embed import interpolate_pos_embed
from models import models_vit
from timm.models.layers import trunc_normal_
import util.misc as misc

class net_and(nn.Module):

    def __init__(self):
        super(net_and, self).__init__()
        A =torch.full((2,4),0.5,requires_grad=True)
        B =torch.ones((1),requires_grad=True)
        self.A = torch.nn.Parameter(A)
        self.B = torch.nn.Parameter(B)
        self.register_parameter("luka_and",self.A)
        self.register_parameter("b",self.B)

    def forward(self, x,y,i,j):
        # print(self.A)
        # print(self.B)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nums=x.size()
        bb=float(self.B[0])
        aaa=torch.zeros((nums)).to(device, torch.float)
        return torch.max(torch.zeros((nums)).to(device, torch.float),torch.min(torch.ones((nums)).to(device, torch.float),torch.full((nums), bb).to(device, torch.float)-self.A[0][i]*(torch.ones((nums)).to(device, torch.float)-x)-self.A[1][j]*(torch.ones((nums)).to(device, torch.float)-y)))
