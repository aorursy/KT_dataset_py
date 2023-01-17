import matplotlib.pyplot as plt

import numpy as np

import torch

from torch import nn, optim

import torch.nn.functional as F

from torch.nn.functional import softmax

from torchvision import datasets, transforms

from torchvision.utils import save_image

import os

import pandas as pd

from numpy import linalg as LA

from tqdm import tqdm

import time

from torch.utils import data
train_samples = 1200

end_samples = 1600 

nb_samples_test = end_samples-train_samples







path = '../input/simon-fourier/coeff2.npy'

coeff1 = np.load(path)/100000



path = '../input/simon-fourier/coeff1.npy'

coeff3 = np.load(path)



path = '../input/simon-fourier/coeff3.npy'

coeff2 = np.load(path)/100000



path = '../input/simon-fourier/loc_avg.npy'

loc_avg = np.load(path)









device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



coeff1_torch = torch.tensor(coeff1[:train_samples, :], dtype=torch.float32, device = device)

coeff2_torch = torch.tensor(coeff2[:train_samples, :], dtype=torch.float32, device = device)

coeff3_torch = torch.tensor(coeff3[:train_samples, :], dtype=torch.float32, device = device)

print("coeff1_torch", np.shape(coeff1_torch))





coeff1_torch_test = torch.tensor(coeff1[train_samples:end_samples, :], dtype=torch.float32, device = device)

coeff2_torch_test = torch.tensor(coeff2[train_samples:end_samples, :], dtype=torch.float32, device = device)

coeff3_torch_test = torch.tensor(coeff3[train_samples:end_samples, :], dtype=torch.float32, device = device)

print("coeff1_torch_test", np.shape(coeff1_torch_test))



u_H = loc_avg[:train_samples, :]

print("u_H", np.shape(u_H))

u_H_torch = torch.tensor(u_H, dtype=torch.float32, device = device)

print("u_H_torch", np.shape(u_H_torch))



u_H_test = loc_avg[train_samples:end_samples, :]

u_H_torch_test = torch.tensor(u_H_test, dtype=torch.float32, device = device)

print("u_H_torch_test", np.shape(u_H_torch_test))





nb_coeff1 = np.shape(u_H)[1]

print("nb_coeff1", nb_coeff1)
# first layer and does not has concat layer

class G1(nn.Module):

    def __init__(self, d_model1, d_model):

        super(G1, self).__init__()

        

        # self.l4 is the dimension reduction layer; if d_model1 = d_model, no reduction;

        self.l4 = nn.Linear(d_model1, d_model).cuda()

        

        self.l5 = nn.Linear(d_model, d_model).cuda()

        self.l6 = nn.Linear(d_model, d_model).cuda()

        

        

    def forward(self, ucoef):

        ucoef = self.l4(ucoef)

        ucoef = self.l5(ucoef)

        ucoef = self.l6(ucoef)

        

        return ucoef

lrG = 0.0001

beta1 = 0.5

epochs = 1000



netG1 = G1(nb_coeff1, nb_coeff1).to(device)



criterion = nn.MSELoss()



optimizerG1 = optim.Adam(netG1.parameters(), lr=lrG, betas=(beta1, 0.999))



sche = optim.lr_scheduler.StepLR( optimizerG1, step_size = int(epochs*1/5), gamma=0.5)
batch_size = train_samples

data_set = data.TensorDataset(coeff1_torch, u_H_torch)

data_loader = data.DataLoader(data_set, batch_size = batch_size, shuffle=True)
loss_G1_set = []

for ep in range(epochs):

    for ix, batch_data in enumerate(data_loader, 0):

        netG1.zero_grad()

        

        output = netG1(batch_data[0].to(device))

        errG1 = criterion(output, batch_data[1].to(device))

        loss_G1_set.append(errG1.item())

        errG1.backward()



        optimizerG1.step()

    sche.step()

        

    if ep % 50 == 0:

        print('epoch [%d] batch [%d] Loss_G1: %.8f ' % (ep, ix, errG1.item()  ) ) 
plt.plot(loss_G1_set)
torch.save(netG1.state_dict(), "./netG1.pth")
outG1 = np.array(netG1(coeff1_torch).cpu().detach())

outG1_torch = torch.tensor(outG1, dtype=torch.float32, device = device)



outG1_test = np.array(netG1(coeff1_torch_test).cpu().detach())



print("L1 norm testing stage 1", np.mean(LA.norm(u_H_test - outG1_test, axis = 1, ord = 1)))

print("L2 norm testing stage 1", np.mean(LA.norm(u_H_test - outG1_test, axis = 1)))



# will be used in later stage

outG1_torch_test = torch.tensor(outG1_test, dtype=torch.float32, device = device)





relative_l2_out1 = []



for ix in range(nb_samples_test):

    err = outG1_test[ix]-u_H_test[ix]

    relative_l2_out1.append(  LA.norm(err)/LA.norm(u_H_test[ix])   )



print("mean relative_l2_out1:", np.mean(relative_l2_out1))

plt.plot(relative_l2_out1)
################################################################### stage 2 #######################################################################################

################################################################### stage 2 #######################################################################################

################################################################### stage 2 #######################################################################################
# used in second and later stages;

# has one more concat layer

class G2(nn.Module):

    def __init__(self, d_model1, d_model):

        super(G2, self).__init__()

        

        # self.l4 is the dimension reduction layer; if d_model1 = d_model, no reduction;

        self.l4 = nn.Linear(d_model1, d_model).cuda()

        

        self.l5 = nn.Linear(d_model, d_model).cuda()

        self.l6 = nn.Linear(d_model, d_model).cuda()

        

        self.l7 = nn.Linear(2*d_model, d_model).cuda()

        

        

    def forward(self, ucoef, uprev):

        ucoef = self.l4(ucoef)

        ucoef = self.l5(ucoef)

        ucoef = self.l6(ucoef)

        

        concat = torch.cat((ucoef, uprev), axis = -1)

        concat = self.l7(concat)

        

        return concat
lrG = 0.0001

beta1 = 0.5

epochs = 1000



netG2 = G2(nb_coeff1, nb_coeff1).to(device)



criterion = nn.MSELoss()



optimizerG2 = optim.Adam(netG2.parameters(), lr=lrG, betas=(beta1, 0.999))



sche = optim.lr_scheduler.StepLR( optimizerG2, step_size = int(epochs*1/5), gamma=0.5)
batch_size = train_samples

data_set = data.TensorDataset(coeff2_torch  , outG1_torch, u_H_torch)

data_loader = data.DataLoader(data_set, batch_size = batch_size, shuffle=True)
loss_G2_set = []

for ep in range(epochs):

    for ix, batch_data in enumerate(data_loader, 0):

        netG2.zero_grad()

        

        output = netG2(batch_data[0].to(device),  batch_data[1].to(device))

        errG2 = criterion(output, batch_data[2].to(device))

        loss_G2_set.append(errG2.item())

        errG2.backward()



        optimizerG2.step()

    sche.step()

        

    if ep % 50 == 0:

        print('epoch [%d] batch [%d] Loss_G2: %.8f ' % (ep, ix, errG2.item()  ) ) 
plt.plot(loss_G2_set)
torch.save(netG2.state_dict(), "./netG2.pth")
outG2 = np.array(netG2(coeff2_torch, outG1_torch).cpu().detach())

outG2_torch = torch.tensor(outG2, dtype=torch.float32, device = device)



outG2_test = np.array(netG2(coeff2_torch_test, outG1_torch_test).cpu().detach())



print("L1 norm testing stage 2", np.mean(LA.norm(u_H_test - outG2_test, axis = 1, ord = 1)))

print("L2 norm testing stage 2", np.mean(LA.norm(u_H_test - outG2_test, axis = 1)))



# will be used in later stage

outG2_torch_test = torch.tensor(outG2_test, dtype=torch.float32, device = device)





relative_l2_out2 = []



for ix in range(nb_samples_test):

    err = outG2_test[ix]-u_H_test[ix]

    relative_l2_out2.append(  LA.norm(err)/LA.norm(u_H_test[ix])   )



print("mean relative_l2_out2:", np.mean(relative_l2_out2))

plt.plot(relative_l2_out2)
################################################################### stage 3 #######################################################################################

################################################################### stage 3 #######################################################################################

################################################################### stage 3 #######################################################################################
lrG = 0.0001

beta1 = 0.5

epochs = 1000



netG3 = G2(nb_coeff1, nb_coeff1).to(device)



criterion = nn.MSELoss()



optimizerG3 = optim.Adam(netG3.parameters(), lr=lrG, betas=(beta1, 0.999))



sche = optim.lr_scheduler.StepLR( optimizerG3, step_size = int(epochs*1/5), gamma=0.5)
batch_size = train_samples

data_set = data.TensorDataset(coeff3_torch  , outG2_torch, u_H_torch)

data_loader = data.DataLoader(data_set, batch_size = batch_size, shuffle=True)
loss_G3_set = []

for ep in range(epochs):

    for ix, batch_data in enumerate(data_loader, 0):

        netG3.zero_grad()

        

        output = netG3(batch_data[0].to(device),  batch_data[1].to(device))

        errG3 = criterion(output, batch_data[2].to(device))

        loss_G3_set.append(errG3.item())

        errG3.backward()



        optimizerG3.step()

    sche.step()

        

    if ep % 50 == 0:

        print('epoch [%d] batch [%d] Loss_G3: %.8f ' % (ep, ix, errG3.item()  ) ) 
plt.plot(loss_G3_set)
torch.save(netG3.state_dict(), "./netG3.pth")
outG3_test = np.array(netG3(coeff3_torch_test, outG2_torch_test).cpu().detach())



print("L1 norm testing stage 2", np.mean(LA.norm(u_H_test - outG3_test, axis = 1, ord = 1)))

print("L2 norm testing stage 2", np.mean(LA.norm(u_H_test - outG3_test, axis = 1)))



relative_l2_out3 = []



for ix in range(nb_samples_test):

    err = outG3_test[ix]-u_H_test[ix]

    relative_l2_out3.append(  LA.norm(err)/LA.norm(u_H_test[ix])   )



print("mean relative_l2_out3:", np.mean(relative_l2_out3))

plt.plot(relative_l2_out3)
print("&",'%.5f' %(np.mean(relative_l2_out1)), "&", '%.5f' %(np.mean(relative_l2_out2)), "&", '%.5f' %(np.mean(relative_l2_out3)), "&")