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
# nb of time steps concatenated to input into the generator

nb_takes = 25

# dim of encoed one time step 

nb_reduced = int(300/nb_takes)

print(nb_takes,  nb_reduced)

# nb of time steps concatenated to input into the generator

nb_takes_phy = nb_takes

# dim of encoded one time step 

nb_reduced_phy = nb_reduced
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# shared data

a = 100

b = 100

dim_mesh = (a-1)*(b-1)



bias_bo = True

bias_bo_g = True

bias_bo_d = True



train_samples = 1200

end_samples = 1600



###################### train data info ######################

# see: 2stages_load_3srs_osc_courses version 2 and 3

# path = '../input/2stage/f_set_osc.npy'

# 2stages_load_3srs_osc_courses_background_time v1 and v2

# path = '../input/2stage/f_set_osc.npy'

# path = '../input/2stage/f_set_4s1.npy'



path = '../input/2stage-simon/f_set_nonl.npy'



ffine_all = 10000*np.load(path)

print("ffine_all", np.shape(ffine_all))

ffine = ffine_all[:train_samples]

print("ffine", np.shape(ffine))



nb_samples = np.shape(ffine)[0]

nb_times = np.shape(ffine)[1]





fcoarse_reshape= np.reshape(ffine, (nb_samples, nb_times, a-1, b-1))

print("fcoarse_reshape", np.shape(fcoarse_reshape))

fcoarse_reshape_torch_before = torch.tensor(fcoarse_reshape, dtype=torch.float32, device = device)

print("fcoarse_reshape_torch_before", np.shape(fcoarse_reshape_torch_before))





###################### test data info ######################



ffine_test = ffine_all[train_samples:end_samples]

print("ffine_test", np.shape(ffine_test))



nb_samples_test = np.shape(ffine_test)[0]

nb_times_test = np.shape(ffine_test)[1]



fcoarse_reshape_test = np.reshape(ffine_test, (nb_samples_test, nb_times_test, a-1, b-1))

fcoarse_reshape_test_torch_before = torch.tensor(fcoarse_reshape_test, dtype=torch.float32, device = device)

print("fcoarse_reshape_test_torch_before", np.shape(fcoarse_reshape_test_torch_before))

path = '../input/cem100/cembasis_10.npy'

_R_ = np.load(path)



Rt = np.transpose(_R_)

print("Rt shape", np.shape(Rt))



# R^t*F_h

RtFh = []

for jx in tqdm(   range(nb_samples)   ):

    RtFh_t = []

    for ix in range(nb_times):

        RtFh_t.append(  np.matmul(  Rt  ,ffine[jx, ix, :])  )

    RtFh.append(RtFh_t)

print("np.shape(RtFh):  R^t*F_h", np.shape(RtFh))



RtFh = np.array(RtFh)

####################################### RtFh training #####################################



## 1 basis

RtFh0 = RtFh[:,:, 0::3]



Rtfh0_torch = torch.tensor(RtFh0, dtype=torch.float32, device = device)

print("Rtfh0_torch", Rtfh0_torch.size())



nb_model_phy0 = Rtfh0_torch.size()[-1]

print("nb_model_phy0", nb_model_phy0)





## 2 basis

RtFh1 = RtFh[:,:, 1::3]



Rtfh1_torch = torch.tensor(RtFh1, dtype=torch.float32, device = device)

print("Rtf1torch", Rtfh1_torch.size())



nb_model_phy1 = Rtfh1_torch.size()[-1]

print("nb_model_phy1", nb_model_phy1)





## 3 basis

RtFh2 = RtFh[:,:, 2::3]



Rtfh2_torch = torch.tensor(RtFh2, dtype=torch.float32, device = device)

print("Rtfh2_torch", Rtfh2_torch.size())



nb_model_phy2 = Rtfh2_torch.size()[-1]

print("nb_model_phy2", nb_model_phy2)





## 3 basis althogether

Rtfh_torch = torch.tensor(RtFh, dtype=torch.float32, device = device)

print("Rtfh_torch", Rtfh_torch.size())



nb_model_phy = Rtfh_torch.size()[-1]

print("nb_model_phy", nb_model_phy)



##################################### RtFh testing #####################################



# R^t*F_h test data

RtFh_test = []

for jx in tqdm(   range(nb_samples_test)   ):

    RtFh_t = []

    for ix in range(nb_times_test):

        RtFh_t.append(  np.matmul(  Rt  ,ffine_test[jx, ix, :])  )

    RtFh_test.append(RtFh_t)

print("np.shape(RtFh_test):  R^t*F_h", np.shape(RtFh_test))



RtFh_test = np.array(RtFh_test)



## 1 basis

RtFh_test0 = RtFh_test[:,:, 0::3]

Rtfh0_torch_test = torch.tensor(RtFh_test0, dtype=torch.float32, device = device)

print("Rtfh0_torch_test", Rtfh0_torch_test.size())



nb_model_phy0_test = Rtfh0_torch_test.size()[-1]

print("nb_model_phy0_test", nb_model_phy0_test)





## 2 basis

RtFh_test1 = RtFh_test[:,:, 1::3]

Rtfh1_torch_test = torch.tensor(RtFh_test1, dtype=torch.float32, device = device)

print("Rtfh1_torch_test", Rtfh1_torch_test.size())



nb_model_phy1_test = Rtfh1_torch_test.size()[-1]

print("nb_model_phy1_test", nb_model_phy1_test)





# ## 3 basis

# RtFh_test2 = RtFh_test[:,:, 2::3]

# Rtfh2_torch_test = torch.tensor(RtFh_test2, dtype=torch.float32, device = device)

# print("Rtfh2_torch_test", Rtfh2_torch_test.size())



# nb_model_phy2_test = Rtfh2_torch_test.size()[-1]

# print("nb_model_phy2_test", nb_model_phy2_test)





# ## 3 basis all together

# Rtfh_torch_test = torch.tensor(RtFh_test, dtype=torch.float32, device = device)

# print("Rtfh_torch_test", Rtfh_torch_test.size())



# nb_model_phy_test = Rtfh_torch_test.size()[-1]

# print("nb_model_phy_test", nb_model_phy_test)





####################################### u_H and u_H_test #####################################



invRtR = LA.inv( np.matmul(Rt, _R_) )



# path = '../input/2stage/sol_set_osc.txt'

# path = '../input/2stage/sol_set_4s1.txt'

path = '../input/2stage-simon/sol_set_nonl.txt'

u_f_all = 10000*pd.read_csv(path, sep=" ", header=None).values

print("u_f_all", np.shape(u_f_all))

u_f = u_f_all[:train_samples]

print("u_f", np.shape(u_f))



u_H = []

for ix in range(nb_samples):

    u_H.append( np.matmul( invRtR,  np.matmul( Rt,  u_f[ix]     )    )  ) 

u_H = np.array(u_H)

print("u_H", np.shape(u_H))

# reduecd model dim; nb of basis in cem setting

nb_rdm = np.shape(u_H)[1]

print("nb_rdm", nb_rdm)





u_f_test = u_f_all[train_samples:end_samples]

print("u_f_test", np.shape(u_f_test))



u_H_test = []

for ix in range(nb_samples_test):

    u_H_test.append( np.matmul( invRtR,  np.matmul( Rt,  u_f_test[ix]     )    ) )

u_H_test = np.array(u_H_test)



print("u_H_test", np.shape(u_H_test))



# recued model dim; nb of basis in cem setting

nb_rdm_test = np.shape(u_H_test)[1]

print("nb_rdm_test", nb_rdm_test)
# # input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1

# generate one head of the transformer (one layer)

# output dim: nb_samples, nb of time steps, reduced dim encode for each time step

class OneHeadAttention(nn.Module):

    def __init__(self, d_model, d_reduced, bo ):

        super().__init__()

        

        self.v_linear = nn.Linear( d_model, d_reduced, bias = bias_bo).cuda()

        self.q_linear = nn.Linear( d_model, d_reduced, bias = bias_bo).cuda()

        self.k_linear = nn.Linear( d_model, d_reduced, bias = bias_bo).cuda()

        

    # self.vv: nb samples, nb time steps, 81 (reduced due to the max pooling, no training)

    def forward(self, vv):

        v = self.v_linear( vv )

        k = self.k_linear( vv )

        q = self.q_linear( vv )

        qkt = torch.matmul(q, torch.transpose(k, 1, 2))



        sm_qkt = softmax(qkt, dim = -1)



        out = torch.matmul(sm_qkt, v)

        return out



model = OneHeadAttention(nb_model_phy, nb_reduced_phy, True )

def init_weights(m):

    if type(m) == nn.Linear:

        m.bias.data.uniform_(-1/100000, 1/100000)





model.apply(init_weights)

out = model(Rtfh_torch)  

np.shape(out)
# V1 OF HEAD 6 heads attention

# need class OneHeadAttention

# generate multi-head of one layer using OneHeadAttention; the nb of heads is fixed and is equal to 3 in this code

# # input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1

# output size: nb batches, time steps, reduced dim (note, equal to the encode dim of one head)



nb_heads = 6

class MultiHeads(nn.Module):

    def __init__(self, d_model, d_reduced, d_head, bo):

        super().__init__()



        self.head1 = OneHeadAttention(d_model, d_reduced, bo)

        self.head2 = OneHeadAttention(d_model, d_reduced, bo)

        self.head3 = OneHeadAttention(d_model, d_reduced, bo)

        self.head4 = OneHeadAttention(d_model, d_reduced, bo)

        self.head5 = OneHeadAttention(d_model, d_reduced, bo)

        self.head6 = OneHeadAttention(d_model, d_reduced, bo)

        

        self.linear = nn.Linear(d_reduced*d_head, d_reduced, bias = bias_bo).cuda()

    def forward(self, v):

        out1 = self.head1(v)

        out2 = self.head2(v)

        out3 = self.head3(v)

        out4 = self.head4(v)

        out5 = self.head5(v)

        out6 = self.head6(v)

        

        concat_out = torch.cat((out1, out2, out3, out4, out5, out6), dim = -1)

        

        out = self.linear(  concat_out )

        

        return out

    

def init_weights(m):

    if type(m) == nn.Linear:

        m.bias.data.uniform_(-1/100000, 1/100000)

        

model = MultiHeads(nb_model_phy, nb_reduced_phy, nb_heads, True )

model.apply(init_weights)

out = model(Rtfh_torch)  

print(out.size())
# version 2 of LAYERS OF TRANSFORMERS: 1 layers transformer

# output size: nb of samples, dim of encoded vector for one samnples



class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_reduced, d_head, d_take):

        super().__init__()



        self.layer1 = MultiHeads(d_model, d_reduced, d_head, True)

        

        self.d_take = d_take

    

    # output dim: (nb_batches, nb_time_steps, encode_dims (d_model) )

    def forward(self, v, d_samples):

        out1 = self.layer1(v)

        

        # out1 size: nb_samples, nb_time_steps, dim

        out = out1[:,-self.d_take:,:].view(d_samples, -1)

        return out

        

model = EncoderLayer(nb_model_phy, nb_reduced_phy, nb_heads, nb_takes_phy)

def init_weights(m):

    if type(m) == nn.Linear:

        m.bias.data.uniform_(-1/100000, 1/100000)

model.apply(init_weights)

out = model(Rtfh_torch, nb_samples)  

print("out", np.shape(out))
####### compare #################################################################################################### compare #

####################### ####################### compare ####################### ####################### ######################

######### compare ################################################### compare ################################################
# Data stage 1 generator

# output dim: nb samples, dim of reduced encoded

class Generator(nn.Module):

    def __init__(self, d_model, d_reduced, d_head, d_take):

        super(Generator, self).__init__()

        

        self.l4 = nn.Linear(d_take*d_reduced,  nb_rdm, bias_bo_g).cuda()

        self.l5 = nn.Linear(nb_rdm,  nb_rdm, bias_bo_g).cuda()

        

        self.encode = EncoderLayer(d_model, d_reduced, d_head, d_take)

        

    def forward(self, v, d_samples):

        encode_out0 = self.encode(v, d_samples)

        encode_out = self.l4(encode_out0)

        G1out = self.l5(encode_out)

                                

        return G1out





# modelG = Generator(nb_model_phy, nb_reduced_phy, nb_heads, nb_takes_phy)

# def init_weights(m):

#     if type(m) == nn.Linear:

#         m.bias.data.uniform_(-1/100000, 1/100000)

# modelG.apply(init_weights)

# Gout = modelG(Rtfh_torch, nb_samples)  

# print(Gout.size())
torch.cuda.is_available()
# prepare the training

u_H_torch = torch.tensor(u_H, dtype=torch.float32, device = device)

####### 1basis #################################################################################################### 1basis #

####################### ####################### 1basis ####################### ####################### ######################

######### 1basis ################################################### 1basis ################################################
netG1 = Generator(nb_model_phy0, nb_reduced, nb_heads, nb_takes).to(device)

path = '../input/2stage-simon/netG1_25.pth'

netG1.load_state_dict(torch.load(path))

# netG1.eval()              
# lrG1 = 0.0003

# beta1 = 0.5



# netG1 = Generator(nb_model_phy0, nb_reduced, nb_heads, nb_takes).to(device)



# criterion = nn.L1Loss()



# def init_weights(m):

#     if type(m) == nn.Linear:

#         m.bias.data.uniform_(-1/100000, 1/100000)





# netG1.apply(init_weights)



# optimizerG1 = optim.Adam(netG1.parameters(), lr = lrG1, betas=(beta1, 0.999))
# lrG1 = 0.0003

# beta1 = 0.5

# optimizerG1 = optim.Adam(netG1.parameters(), lr = lrG1, betas=(beta1, 0.999))
# # training

# epochs = 2000



# loss_G1_set = []

# for ep in range(epochs):

   

#     netG1.zero_grad()

#     output = netG1(Rtfh0_torch, nb_samples)

#     errG1 = criterion(output, u_H_torch)

#     loss_G1_set.append(errG1.item())

#     errG1.backward()



#     optimizerG1.step()



#     if ep % 100 == 0:

#         print('[%d] Loss_G1: %.8f ' % (ep, errG1.item()  ) ) 
# torch.save(netG1.state_dict(), "./netG1.pth")
# plt.plot(loss_G1_set)
outG1 = np.array(netG1(Rtfh0_torch, nb_samples).cpu().detach())

outG1_phy_torch = torch.tensor(outG1, dtype=torch.float32, device = device)



start_time1 = time.time()



time1 = time.time() - start_time1

outG1_test = np.array(netG1(Rtfh0_torch_test, nb_samples_test).cpu().detach())

time1 = time.time() - start_time1

outG1_phy_torch_test = torch.tensor(outG1_test, dtype=torch.float32, device = device)



print("L1 norm testing stage 1", np.mean(LA.norm(u_H_test-outG1_test, axis = 1, ord = 1)))

print("L2 norm testing stage 1", np.mean(LA.norm(u_H_test-outG1_test, axis = 1)))
relative_l2_phy1 = []



for ix in range(nb_samples_test):

    err = outG1_test[ix]-u_H_test[ix]

    relative_l2_phy1.append(  LA.norm(err)/LA.norm(u_H_test[ix])   )



print("mean relative_l2_phy1:", np.mean(relative_l2_phy1))

plt.plot(relative_l2_phy1)
####### 2basis #################################################################################################### 2basis  #

####################### ####################### 2basis ####################### ####################### ######################

######### 2basis ################################################### 2basis ################################################
# Generator of physics; combined with 



# output dim: nb samples, dim of reduced encoded

# input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1

class G2(nn.Module):

    def __init__(self, d_model, d_reduced, d_head, d_take):

        super(G2, self).__init__()

        

        self.l4 = nn.Linear(d_take*d_reduced,  nb_rdm, bias_bo_g).cuda()

        self.l5 = nn.Linear(nb_rdm,  nb_rdm, bias_bo_g).cuda()

        self.l6 = nn.Linear( 2*nb_rdm, nb_rdm   ).cuda()

        self.encode = EncoderLayer(d_model, d_reduced, d_head, d_take)

        

    def forward(self, v, dt, d_samples):

        encode_out0 = self.encode(v, d_samples)

        encode_out = self.l4(encode_out0)

        

        

        phy = self.l5(encode_out)

        

        concat = torch.cat((phy, dt), axis = -1)

        G2out =  self.l6(concat)

        return G2out, phy





# modelG2 = G2(nb_model, nb_reduced, nb_heads, nb_takes)



# def init_weights(m):

#     if type(m) == nn.Linear:

#         m.bias.data.uniform_(-1/100000, 1/100000)

# modelG2.apply(init_weights)



# G2out, corrector = modelG2(Rtfh1_torch, outG1_phy_torch, nb_samples)  

# print(G2out.size())
netG2 = G2(nb_model_phy1, nb_reduced, nb_heads, nb_takes).to(device)

path = '../input/2stage-simon/netG2_25.pth'

netG2.load_state_dict(torch.load(path))

# netG2.eval()           
# # prepare the training



# lrG2 = 0.0003

# beta1 = 0.5



# netG2 = G2(nb_model_phy1, nb_reduced, nb_heads, nb_takes).to(device)



# criterion = nn.L1Loss()



# def init_weights(m):

#     if type(m) == nn.Linear:

#         m.bias.data.uniform_(-1/100000, 1/100000)





# netG2.apply(init_weights)



# optimizerG2 = optim.Adam(netG2.parameters(), lr = lrG2, betas=(beta1, 0.999))
# # training

# epochs = 2000



# loss_G2_set = []

# for ep in range(epochs):

#     netG2.zero_grad()

#     output, corrector = netG2(Rtfh1_torch, outG1_phy_torch, nb_samples)

#     errG2 = criterion(output, u_H_torch)

#     loss_G2_set.append(errG2.item())

#     errG2.backward()



#     optimizerG2.step()



#     if ep % 100 == 0:

#         print('[%d] Loss_G2: %.8f ' % (ep, errG2.item()  ) ) 
# plt.plot(loss_G2_set)

# plt.show()
# study the role of the corrector

test_id = 10

outG2 = np.array(netG2(Rtfh1_torch, outG1_phy_torch, nb_samples)[0].cpu().detach())



outG2_phy_torch = torch.tensor(outG2, dtype=torch.float32, device = device)



start_time2 = time.time()







outG2_test = np.array(netG2(Rtfh1_torch_test, outG1_phy_torch_test, nb_samples_test)[0].cpu().detach())



time2 = time.time() - start_time2

corrector2_test = np.array(netG2(Rtfh1_torch_test, outG1_phy_torch_test, nb_samples_test)[1].cpu().detach())



outG2_phy_torch_test = torch.tensor(outG2_test, dtype=torch.float32, device = device)



print("L1 norm testing stage 2", np.mean(LA.norm(u_H_test-outG2_test, axis = 1, ord = 1)))

print("L2 norm testing stage 2", np.mean(LA.norm(u_H_test-outG2_test, axis = 1)))
# torch.save(netG2.state_dict(), "./netG2.pth")
relative_l2_phy2 = []



for ix in range(nb_samples_test):

    err = outG2_test[ix]-u_H_test[ix]

    relative_l2_phy2.append(  LA.norm(err)/LA.norm(u_H_test[ix])   )



print("relative_l2_phy2:", np.mean(relative_l2_phy2))

plt.plot(relative_l2_phy2)
####### 3basis #################################################################################################### 3basis  #

####################### ####################### 3basis ####################### ####################### ######################

######### 3basis ################################################### 3basis ################################################
# reduce the spatial dim of f (f projected on the fine mesh ) by max pooling ## CUDA

# reshape the pooling output to a vector



# batch_size: nb of samples

# d_channels: nb of time steps

# dt: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1; can als be nb of samples, nb of time steps, d_reduced

# output shape: nb of samples, nb of time steps, 81 (reduced dim due to the pooling)

# no training included



rand_factor = 0



class PreProcessingData(nn.Module):

    def __init__(self, bo):

        super().__init__()

        self.pool = nn.MaxPool2d(10, stride = 10)

        self.bo = bo

    # q = [q1; q2;...,qn]; dim: # model dim (d_model), # time steps

    # note the values for Q K are all V which is model reduction coeff; 

    # there is no need to generate the random values for Q and K

    # each row is: the query key and vals of a word

    def forward(self, dt, batch_size, d_channels):

        rd = torch.randn(batch_size, d_channels,  dt.size()[-1], device = device)

        

        if self.bo:

            dt = self.pool(dt)

            dt = dt.view(batch_size, d_channels, -1)

            rd = torch.randn(batch_size, d_channels,  dt.size()[-1], device = device)

            return (dt+rd*rand_factor)



        else:

            rd = torch.randn(batch_size, d_channels,  dt.size()[-1], device = device)

            return (dt+rd*rand_factor)



model = PreProcessingData(True)

Rtfh2_torch = model(fcoarse_reshape_torch_before, nb_samples, nb_times)  

print( "Rtfh2_torch",  np.shape(Rtfh2_torch))

nb_model_phy2 = Rtfh2_torch.size()[-1]

print("nb_model_phy2", nb_model_phy2)



model = PreProcessingData(True)

Rtfh2_torch_test = model(fcoarse_reshape_test_torch_before, nb_samples_test, nb_times_test)  

print( "Rtfh2_torch_test",  np.shape(Rtfh2_torch_test))

nb_model_phy2_test = Rtfh2_torch_test.size()[-1]

print("nb_model_phy2_test", nb_model_phy2_test)
# prepare the training



lrG3 = 0.0003

beta1 = 0.5



netG3 = G2(nb_model_phy2, nb_reduced, nb_heads, nb_takes).to(device)



criterion = nn.L1Loss()



def init_weights(m):

    if type(m) == nn.Linear:

        m.bias.data.uniform_(-1/100000, 1/100000)





netG3.apply(init_weights)



optimizerG3 = optim.Adam(netG3.parameters(), lr = lrG3, betas=(beta1, 0.999))
# training

epochs = 2000



loss_G3_set = []

for ep in range(epochs):

    netG3.zero_grad()

    output, corrector = netG3(Rtfh2_torch, outG2_phy_torch, nb_samples)

    errG3 = criterion(output, u_H_torch)

    loss_G3_set.append(errG3.item())

    errG3.backward()



    optimizerG3.step()



    if ep % 100 == 0:

        print('[%d] Loss_G3: %.8f ' % (ep, errG3.item()  ) ) 
plt.plot(loss_G3_set)

plt.show()
torch.save(netG3.state_dict(), "./netG3.pth")
# study the role of the corrector

test_id = 10

outG3 = np.array(netG3(Rtfh2_torch, outG2_phy_torch, nb_samples)[0].cpu().detach())



start_time3 = time.time()

outG3_test = np.array(netG3(Rtfh2_torch_test, outG2_phy_torch_test, nb_samples_test)[0].cpu().detach())

time3 = time.time() - start_time3

corrector3_test = np.array(netG3(Rtfh2_torch_test, outG2_phy_torch_test, nb_samples_test)[1].cpu().detach())





print("L1 norm testing stage 2", np.mean(LA.norm(u_H_test-outG3_test, axis = 1, ord = 1)))

print("L2 norm testing stage 2", np.mean(LA.norm(u_H_test-outG3_test, axis = 1)))
# relative_l2_corrector3 = []



# for ix in range(nb_samples_test):

#     err = corrector3_test[ix]-u_H_test[ix]

#     relative_l2_corrector3.append(  LA.norm(err)/LA.norm(u_H_test[ix])   )



# print("mean relative_l2_corrector3:", np.mean(relative_l2_corrector3))

# plt.plot(relative_l2_corrector3)
relative_l2_phy3 = []



for ix in range(nb_samples_test):

    err = outG3_test[ix]-u_H_test[ix]

    relative_l2_phy3.append(  LA.norm(err)/LA.norm(u_H_test[ix])   )



print("relative_l2_phy3:", np.mean(relative_l2_phy3))

plt.plot(relative_l2_phy3)
####################### wrap up. 
relative_l2_mean = []



mean_ = np.mean(u_H, axis = 0)



for ix in range(nb_samples_test):

    err = mean_-u_H_test[ix]

    relative_l2_mean.append(  LA.norm(err)/LA.norm(u_H_test[ix])   )



print("mean relative_l2_mean:", np.mean(relative_l2_mean))

plt.plot(relative_l2_mean)
print( np.mean(relative_l2_phy1) ,np.mean(relative_l2_phy2), np.mean(relative_l2_phy3)  )
print('%.5f %.5f %.5f' % (  np.mean(relative_l2_phy1) ,np.mean(relative_l2_phy2), np.mean(relative_l2_phy3) ) ) 
####### error #################################################################################################### error  #

####################### ####################### error ####################### ####################### ######################

######### error ################################################### error ################################################
# load data

path = '../input/full-mesh/ele.txt'

ele = pd.read_csv(path, sep=" ", header=None)

ele = ele.values

ele = ele[:,range(2,6)]



path2 = '../input/full-mesh/coord.txt'

coord = pd.read_csv(path2, sep=" ", header=None)

coord = coord.values





path3 = '../input/full-mesh/ms-data.txt'

ms_data = pd.read_csv(path3, sep = r'\s{2,}', header=None, engine='python')

ms_data = ms_data.values



# nb of elements in x dir

a = 100

# nb of elements in y dir

b = 100



# total dofs

dof = np.unique(ele)

nb_dof = np.size(dof)



# nb of element

nb_ele,nb_col = ele.shape





int_dof_domain = []

for i in range(1, a):

    for j in range(1, b):

        int_dof_domain.append(i*(a+1)+j)
ms_data1 = ms_data[0:-1:6, 0:-1:6]

plt.imshow(ms_data1)

plt.show()

kappa = np.reshape(ms_data1,(10000,))

for i in range(len(kappa)):

    if kappa[i]>10:

        kappa[i] = 1000

        

print(np.shape(kappa))

print(kappa.min())

print(kappa.max())
loc_stiff = [[  2/3, -1/6, -1/3, -1/6],

            [ -1/6,  2/3, -1/6, -1/3],

             [ -1/3, -1/6,  2/3, -1/6],

            [ -1/6, -1/3, -1/6,  2/3]

            ]



loc_stiff = np.matrix(loc_stiff)



p1p1 = 1/90000

p1p2 = 1/180000

p1p3 = 1/360000

p1p4 = 1/180000



p2p2 = 1/90000

p2p3 = 1/180000

p2p4 = 1/360000



p3p3 = 1/90000

p3p4 = 1/180000



p4p4 = 1/90000



loc_mass = [[p1p1,p1p2,p1p3,p1p4],

        [p1p2,p2p2,p2p3,p2p4],

        [p1p3,p2p3,p3p3,p3p4],

        [p1p4,p2p4,p3p4,p4p4]]



loc_mass =np.matrix(loc_mass)
stiff_fine = np.zeros(( len(dof), len(dof) ))

mass_norm = np.zeros((  len(dof), len(dof) ))



for jx in range(len(ele)):

    loc_dof = ele[jx]

    coeff = kappa[jx]

    for p in range(0,4):

        for q in range(0,4):

            ai = loc_dof[p]

            bi = loc_dof[q]

            stiff_fine[ai,bi] = stiff_fine[ai,bi]+coeff*loc_stiff[p,q]

            mass_norm[ai,bi] = mass_norm[ai,bi]+loc_mass[p,q]*coeff





stiff_fine = stiff_fine[int_dof_domain ,:]

stiff_fine = stiff_fine[:, int_dof_domain]



mass_norm = mass_norm[int_dof_domain ,:]

mass_norm = mass_norm[:, int_dof_domain]
# # linear version;

# path = '../input/2stage/mass_f.txt'

# mass_f = pd.read_csv(path, sep=" ", header=None).values[train_samples:end_samples]

# path = '../input/2stage/stiff_f.txt'

# stiff_f = pd.read_csv(path, sep=" ", header=None).values[train_samples:end_samples]

# path = '../input/2stage/s3h_mass.txt'

# s3h_mass = pd.read_csv(path, sep=" ", header=None).values[train_samples:end_samples]

# path = '../input/2stage/s3h_stiff.txt'

# s3h_stiff = pd.read_csv(path, sep=" ", header=None).values[train_samples:end_samples]
# non-linear version;

path = '../input/2stage-simon/mass_f.txt'

mass_f = pd.read_csv(path, sep=" ", header=None).values[train_samples:end_samples]

path = '../input/2stage-simon/stiff_f.txt'

stiff_f = pd.read_csv(path, sep=" ", header=None).values[train_samples:end_samples]

path = '../input/2stage-simon/s3h_mass.txt'

s3h_mass = pd.read_csv(path, sep=" ", header=None).values[train_samples:end_samples]

path = '../input/2stage-simon/s3h_stiff.txt'

s3h_stiff = pd.read_csv(path, sep=" ", header=None).values[train_samples:end_samples]
# mass_f = []

# stiff_f = []

# total_samples = np.shape(u_f_all)[0]

# for ix in tqdm(range(total_samples)):

#     mass_f.append(    np.matmul(  np.transpose(u_f_all[ix]),       np.matmul(   mass_norm, u_f_all[ix]   )      )         )

#     stiff_f.append(    np.matmul(  np.transpose(u_f_all[ix]),       np.matmul(   stiff_fine, u_f_all[ix]   )      )         )
# np.savetxt("mass_f.txt",mass_f)

# np.savetxt("stiff_f.txt",stiff_f)
def err_( coeff, fine, spls):

    

    mass_relative = []

    stiff_relative = []

    for ix in tqdm(range(spls)):

        err = np.matmul(_R_, coeff[ix] )- fine[ix]

        mass_relative.append(    np.matmul(  np.transpose(err),       np.matmul(   mass_norm, err  )      )      /mass_f[ix]           )

        stiff_relative.append(    np.matmul(  np.transpose(err),       np.matmul(   stiff_fine, err  )      )     /stiff_f[ix]            )

    

    return mass_relative, stiff_relative
# u_H_all = []

# for ix in range(total_samples):

#     u_H_all.append( np.matmul( invRtR,  np.matmul( Rt,  u_f_all[ix]     )    )  ) 

# u_H_all = np.array(u_H_all)



# s3h_mass, s3h_stiff = err_( u_H_all, u_f_all, total_samples)

# print(np.mean(s3h_mass), np.mean(s3h_stiff)  )
# np.savetxt("s3h_mass.txt",s3h_mass)

# np.savetxt("s3h_stiff.txt",s3h_stiff)
s3_mass, s3_stiff = err_( outG3_test, u_f_test, nb_samples_test)

print(np.mean(s3_mass), np.mean(s3_stiff)  )
print("&",'%.5f' %(np.mean(relative_l2_phy1)), "&", '%.5f' %(np.mean(relative_l2_phy2)), "&", '%.5f' %(np.mean(relative_l2_phy3)))
print("&",'%.5f' %(np.mean(s3_mass)), "&", '%.5f' %(np.mean(s3h_mass)), "&", '%.5f' %(np.mean(s3_stiff)), "&", '%.5f' %(np.mean(s3h_stiff)), "&")
print(nb_takes,  nb_reduced)
print(time1+time2+time3)