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
# tphyper

Nv = 32

Nt = 64

Nx = 40



l_points, l_weights = np.polynomial.legendre.leggauss(Nv)

dx = 0.025

dt = 0.5*dx*dx



nb_bc = 300

nb_ic = 300



epsilon = 0.0001





lambda_ge = 1

lambda_ic = 1

lambda_bc = 1



lr1 = 0.0001

beta1 = 0.9999

nb_epochs = 1000



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load and pre-processing data

# of size: nb_space_points, nb_v

ex1t8 = []

ex1t8_ix = [2, 3, 4, 6, 7, 8, 20, 30]

for ii in ex1t8_ix:

    path = "../input/liu-ex1/EX1Space%dTime8.txt"%ii

    ex1t8.append(np.loadtxt(path))



ex1t16 = []

ex1t16_ix = [2, 4, 6, 8, 10, 20, 30, 40]

for ii in ex1t16_ix :

    path = "../input/liu-ex1/EX1Space%dTime16.txt"%ii

    ex1t16.append(np.loadtxt(path))



ex1t32 = []

ex1t32_ix = [1,2,3,5,6,8,10,20,30,40]

for ii in ex1t32_ix :

    path = "../input/liu-ex1/EX1Space%dTime32.txt"%ii

    ex1t32.append(np.loadtxt(path))



ex1t64 = []

ex1t64_ix = [2,3,4,6,10,25,30,35,40]

for ii in ex1t64_ix :

    path = "../input/liu-ex1/EX1Space%dTime64.txt"%ii

    ex1t64.append(np.loadtxt(path))



ex1_dic = [ex1t8_ix, ex1t16_ix, ex1t32_ix, ex1t64_ix]

nb_x_1 = len( ex1_dic[0] )

nb_x_2 = len( ex1_dic[1] )

nb_x_3 = len( ex1_dic[2] )

nb_x_4 = len( ex1_dic[3] )



# make feature x and target y

# loop over time

for nn in [8, 16, 32, 64]:

    globals()["t"+str(nn)+"x"] = []

    globals()["t"+str(nn)+"y"] = []

    globals()["t"+str(nn)+"L"] = []

    

    # of size: nb_space points* nb_V

    ex1t = globals()["ex1t"+str(nn)]

    # ix is used to calculate the exact x value in space

    ex1t_ix = globals()["ex1t"+str(nn)+"_ix"]

    nb_x = np.shape(ex1t)[0]

    

    # loop over all space points

    for ii in range(nb_x):

        ex1t_x_mean = np.mean(ex1t[ii])

        

        # loop over v at t = nn and x = ii

        for vv in range(Nv):

            globals()["t"+str(nn)+"x"].append([nn*dt, dx/2+(ex1t_ix[ii]-1)*dx, l_points[vv]] )

            globals()["t"+str(nn)+"y"].append(ex1t[ii][vv])

            globals()["t"+str(nn)+"L"].append(ex1t[ii][vv]-ex1t_x_mean)



# files created (DO NOT REMOVE)

# t8x, t8y, t16x, t16y, t32x, t32y, t64x, t64y 

# t8L, t16L, t32L, t64L
ex1x = np.array( t8x+t16x+t32x+t64x )



ex1Y = np.array(t8y+t16y+t32y+t64y)

ex1L = np.array(t8L+t16L+t32L+t64L)



ex1T = np.expand_dims(ex1x[:, 0], axis = 1)

ex1S = np.expand_dims(ex1x[:, 1], axis = 1)

ex1V = np.expand_dims(ex1x[:, 2], axis = 1)

ex1Y = np.expand_dims(ex1Y, axis = 1)

np.save("ex1Y", ex1Y)

np.save("ex1T", ex1T)

np.save("ex1S", ex1S)

np.save("ex1V", ex1V)
# tplf

def calLF(tay):

    

    seg1 = tay[ :nb_x_1*Nv   , ]

    seg2 = tay[ nb_x_1*Nv:nb_x_1*Nv + nb_x_2*Nv , ]

    seg3 = tay[ nb_x_1*Nv + nb_x_2*Nv:nb_x_1*Nv + nb_x_2*Nv+nb_x_3*Nv   , ]

    seg4 = tay[ -nb_x_4*Nv:   , ]

    

    # loop over x

    for xx in range(nb_x_1):

        avg = seg1[xx*Nv:(xx+1)*Nv ].mean(dim = 0)/2



        for vv in range(Nv):

            seg1[xx*Nv+vv, ] = avg-seg1[xx*Nv+vv, ]

    # loop over x

    for xx in range(nb_x_2):

        avg = seg2[xx*Nv:(xx+1)*Nv ].mean(dim = 0)/2

        for vv in range(Nv):

            seg2[xx*Nv+vv, ] = avg-seg2[xx*Nv+vv, ]

    # loop over x

    for xx in range(nb_x_3):

        avg = seg3[xx*Nv:(xx+1)*Nv ].mean(dim = 0)/2

        for vv in range(Nv):

            seg3[xx*Nv+vv, ] = avg-seg3[xx*Nv+vv, ]

    # loop over x

    for xx in range(nb_x_4):

        avg = seg4[xx*Nv:(xx+1)*Nv ].mean(dim = 0)/2

        for vv in range(Nv):

            seg4[xx*Nv+vv, ] = avg-seg4[xx*Nv+vv, ]

    

    return torch.cat((seg1,seg2,seg3,seg4), dim = 0 )


