import imp
from operator import imod
from re import I
from turtle import forward
import torch
import numpy as np
import os, json, cv2, random
import xml.etree.ElementTree as ET
import torch.nn as nn
import torchvision
import torch.utils.data as Data


class OnClassify_v1(nn.Module):

    def __init__(self):
        super(OnClassify_v1, self).__init__()
        self.F6 = nn.Linear(in_features=8, out_features=50)
        self.R4 = nn.ReLU()
        self.F7 = nn.Linear(in_features=50, out_features=50)
        self.R5 = nn.ReLU()
        self.OUT = nn.Linear(50, 1)

    def forward(self, x):
        x = self.F6(x)
        x = self.R4(x)
        x = self.F7(x)
        x = self.R5(x)
        x = self.OUT(x)
        # return x.softmax(dim=1)
        return x



