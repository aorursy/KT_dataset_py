import torch 

import torch.nn as nn 

import torch.nn.functional as F

from torch.nn.parameter import Parameter

class PELU(nn.Module):

    def __init__(self,epsilon=1e-9,inplace=False):

        super(PELU,self).__init__()

        self.epsilon=epsilon

        self.inplace=inplace

        self.dummyalpha=1

        self.beta = Parameter(torch.Tensor([0.25],dtype=torch.float64))

        self.alpha = Parameters(torch.Tensor([0.25],dtype=torch.float64))

    def forward(self,input):

        x=1/(self.beta+self.epsilon)

        x=torch.mul(input,x)

        x=F.elu(x,self.dummyalpha,self.inplace)

        return torch.mul(x,self.alpha)