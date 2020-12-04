import torch
import torch.nn as nn


class HRNet(nn.Module):
    def __init__(self,out_stride):
        super(HRNet, self).__init__()
