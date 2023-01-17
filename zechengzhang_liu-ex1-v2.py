import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.functional import softmax
from torchvision import datasets, transforms

import os
import pandas as pd
from numpy import linalg as LA
from tqdm import tqdm
import time
from torch.utils import data
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import eigh
from scipy.linalg import eig

from scipy.stats import multivariate_normal as normalpdf
from numpy.random import multivariate_normal as sample_normal
from numpy.random import binomial as binomial
import random

import collections
path = "../input/liu-ex1/ex1S.npy"
test_s = np.load(path)
path = "../input/liu-ex1/ex1T.npy"
test_t = np.load(path)
path = "../input/liu-ex1/ex1V.npy"
test_v = np.load(path)
path = "../input/liu-ex1/ex1Y.npy"
test_y = np.load(path)
print(np.shape(test_s), np.shape(test_t), np.shape(test_v), np.shape(test_y))
# tphyper

#space
space_size = 16
dx = 1/space_size
space_sample = [dx/2+dx*ii for ii in range(space_size)]


# vspace
Nv = 32
vpoints = [-1+2/Nv*(ii+1) for ii in range(Nv)]

# time; deps on space
# time_size will be set according to dx and max_T
max_T = max(test_t)
print("max_T", max_T)
dt = 0.5*dx*dx
time_size = int(max_T/dt)
print("time_size", time_size)
time_sample = [(ii+1)*dt for ii in range(time_size)]

epsilon = 0.001

lambda_ge = 1
lambda_ic = 1000
lambda_bc = 1000

lr1 = 0.0001
beta1 = 0.9999
nb_epochs = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def calLF(tay, tsize, xsize, vsize):
    for tt in range(tsize):
        for xx in range(xsize):
            avg = tay[tt*xsize*vsize+xx*vsize:tt*xsize*vsize+xx*vsize+vsize, ].mean()/2
            for vv in range(vsize):
                tay[tt*xsize*vsize+xx*vsize+vv, ] = avg-tay[tt*xsize*vsize+xx*vsize+vv, ]
    return tay
TT = []
SS = []
VV = []

for tt in range(time_size):
    for xx in range(space_size):
        for vv in range(Nv):
            TT.append(time_sample[tt])
            SS.append(space_sample[xx])
            VV.append(vpoints[vv])
    
TT = np.expand_dims(np.array(TT), axis = 1)
SS = np.expand_dims(np.array(SS), axis = 1)
VV = np.expand_dims(np.array(VV), axis = 1)
print(np.shape(TT), np.shape(SS), np.shape(VV))
# tpbc
# bc data
bc_l = []
bc_y_l = []
for ii in range(time_size):
    for jj in range( int(Nv/2)-1 ):
        tx = time_sample[ii]
        vx = vpoints[jj]
        bc_l.append([tx, 1, vx ])
        bc_y_l.append(0)
    
bc_r = []
bc_y_r = []
for ii in range(time_size):
    for jj in range(int(Nv/2), Nv):
        tx = time_sample[ii]
        vx = vpoints[jj]
        bc_r.append([tx, 0, vx ])
        bc_y_r.append(1)

bc_x = bc_l+bc_r
bc_y = bc_y_l+bc_y_r

bc_x_torch  = torch.tensor(bc_x, device = device, dtype=torch.float32)
bc_y_torch  = torch.tensor(bc_y, device = device, dtype=torch.float32).unsqueeze(dim = 1)
print("bc_x_torch", bc_x_torch.size())
print("bc_y_torch", bc_y_torch.size())

# tpic
# ic data
ic_x = []
for ii in range(space_size):
    for jj in range(Nv):
        xx = space_sample[ii]
        vx = vpoints[jj]
        ic_x.append([ 0, xx, vx ])
ic_y = [0 for ii in range(len(ic_x))]

ic_x_torch  = torch.tensor(ic_x, device = device, dtype=torch.float32)
ic_y_torch  = torch.tensor(ic_y, device = device, dtype=torch.float32).unsqueeze(dim = 1)
print("ic_x_torch", ic_x_torch.size())
print("ic_y_torch", ic_y_torch.size())
# tpactor
activation = nn.ReLU()
class Test(nn.Module):
    def __init__(self):
        super().__init__()
        
        bo_b = True
        self.l1 = nn.Linear(3, 12, bias = bo_b).to(device)
        self.l2 = nn.Linear(12, 1, bias = bo_b).to(device)
#         self.l3 = nn.Linear(6, 6, bias = bo_b).to(device)
#         self.l4 = nn.Linear(6, 3, bias = bo_b).to(device)
#         self.l5 = nn.Linear(3, 1, bias = bo_b).to(device)

        
        
    def forward(self, state):
        v = self.l1(state)  
        v = self.l2(v)  
#         v = self.l3(v)  
#         v = self.l4(v)  
#         v = self.l5(v)  
    
        return v
# tpinitialize
tt_torch = torch.tensor(TT, requires_grad = True, device = device, dtype=torch.float32)
print("tt_torch", tt_torch.size())
ss_torch = torch.tensor(SS, requires_grad = True, device = device, dtype=torch.float32)
print("ss_torch", ss_torch.size())
vv_torch = torch.tensor(VV, requires_grad = True, device = device, dtype=torch.float32)
vv_torch_detach = torch.tensor(VV, requires_grad = True, device = device, dtype=torch.float32)
print("vv_torch", vv_torch.size())
tsv_torch = torch.cat((tt_torch, ss_torch, vv_torch ), dim = -1)
print("tsv_torch", tsv_torch.size())


# testing
test_tsv = np.concatenate((test_t, test_s, test_v), axis = 1)
test_tsv_torch = torch.tensor(test_tsv, device = device, dtype=torch.float32)
print("test_tsv_torch", test_tsv_torch.size()  )

# network
fnn = Test()
criterion = nn.MSELoss()
fnn_opt = optim.Adam(fnn.parameters(), lr = lr1, betas=(beta1, 0.999))

## tprun
nb_epochs = 1000

set_loss_ge = []
set_loss_bc = []
set_loss_ic = []
set_loss_all = []
set_loss_test = []
for ep in range(nb_epochs):
    fnn_opt.zero_grad()
    ge_out = fnn.forward(tsv_torch)
    bc_out = fnn.forward(bc_x_torch)
    ic_out = fnn.forward(ic_x_torch)
    
    loss_bc = criterion(bc_y_torch, bc_out)
    loss_ic = criterion(ic_y_torch, ic_out)
    
    df_dt, df_dx = torch.autograd.grad(ge_out, (tt_torch, ss_torch), 
                                       grad_outputs = torch.ones( ss_torch.size(), device = device ), create_graph = True)
    
    diff_sum = df_dt+df_dx*vv_torch_detach-1/epsilon*calLF(ge_out, time_size, space_size, Nv)
    loss_ge =  (diff_sum*diff_sum ).mean()
            
    loss_all = lambda_ge*loss_ge+lambda_ic*loss_ic+lambda_bc*loss_bc
    
    set_loss_ge.append(loss_ge.item())
    set_loss_bc.append(loss_bc.item())
    set_loss_ic.append(loss_ic.item())
    set_loss_all.append(loss_all.item())
    
    loss_all.backward()
    fnn_opt.step()
    
    test_out = fnn.forward(test_tsv_torch).cpu().detach().numpy()
    relativ_err = LA.norm(test_out-test_y)/LA.norm(test_y)
    set_loss_test.append(relativ_err)
    
    if ep%5== 0:
        print("ep", ep, "ge", '%.3f' %loss_ge.item(), 'bc',
             '%.4f' %loss_bc.item(), 'ic','%.5f' %loss_ic.item(), 'all', '%.3f' %loss_all.item(), "test", '%.4f' %relativ_err)
    
np.save("set_loss_ge", set_loss_ge)
np.save("set_loss_bc", set_loss_bc)
np.save("set_loss_ic", set_loss_ic)
np.save("set_loss_all", set_loss_all)
np.save("set_loss_test", set_loss_test)
plt.plot(set_loss_ge)
plt.show()
plt.plot(set_loss_ic)
plt.show()
plt.plot(set_loss_bc)
plt.show()
plt.plot(set_loss_all)
plt.show()
plt.plot(set_loss_test)
plt.show()
np.save("TT", TT)
np.save("SS", SS)
np.save("VV", VV)
torch.save(fnn.state_dict(), "./fnn.pth")