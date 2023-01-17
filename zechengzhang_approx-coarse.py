import numpy as np

import math

from scipy import sparse

import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve

from numpy import linalg as LA

import pandas as pd

import time

from scipy.linalg import eigh

from scipy.linalg import eig

from tqdm import tqdm

import random



import time
# load data

path = '../input/full-mesh/ele.txt'

ele = pd.read_csv(path, sep=" ", header=None)

ele = ele.values

ele = ele[:,range(2,6)]



path2 = '../input/full-mesh/coord.txt'

coord = pd.read_csv(path2, sep=" ", header=None)

coord = coord.values



# total dofs

dof = np.unique(ele)

nb_dof = np.size(dof)



# nb of element

nb_ele,nb_col = ele.shape





path = '../input/kappa-1000/RtMR0.npy'

RtMR0 = np.load(path)

path = '../input/kappa-1000/RtMR1.npy'

RtMR1 = np.load(path)

path = '../input/kappa-1000/RtMR2.npy'

RtMR2 = np.load(path)





path = '../input/kappa-1000/RtSR0.npy'

RtSR0 = np.load(path)

path = '../input/kappa-1000/RtSR1.npy'

RtSR1 = np.load(path)

path = '../input/kappa-1000/RtSR2.npy'

RtSR2 = np.load(path)



path = '../input/kappa-1000/invMS0.npy'

invMS0 = np.load(path)

path = '../input/kappa-1000/invMS1.npy'

invMS1 = np.load(path)

path = '../input/kappa-1000/invMS2.npy'

invMS2 = np.load(path)



path = '../input/kappa-1000/cembasis_5_100.npy'

R0 = np.load(path)
x1 = 1.0

x2 = 1.0

x3 = 1.0



nx1 = 100

nx3 = 100



nx2 = 100

dx2 = x2/nx2



dt = 0.0001

T = 1.0

nt = int(T/dt)

print("nt", nt)





# coeff of the 

coeff_l = (1/(dt*dt)+1/( 2*dt*dx2) )





int_dof_domain = []

for i in range(1, nx1):

    for j in range(1, nx3):

        int_dof_domain.append(i*(nx1+1)+j)
# f = sin(pi x)sin(pi y)

x = np.zeros(nx1+1)

for i in range(0,nx1+1):

    x[i] = 0+i*1/nx1

y = np.zeros(nx3+1)

for i in range(0,nx3+1):

    y[i] = 0+i*1/nx3



xv, yv = np.meshgrid(x,y)

f_notime_x2 = np.multiply(np.sin(np.pi*xv), np.sin(np.pi*yv))



f_notime_x2 = f_notime_x2.reshape((nx1+1)*(nx3+1))

f_notime_x2 = f_notime_x2[int_dof_domain]

plt.imshow( np.reshape(f_notime_x2,  (99, 99)  )      )

plt.colorbar()
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
# part 1/3

# define global fine stiff and mass matrix

# and hence define _RtMR_ and _RtSR_

# these two matrices are indep of coarse mesh size

# make global mass and stiff matrix

# domain bnd dofs are included

mass_fine = np.zeros((  len(dof), len(dof) ))



for jx in range(len(ele)):

    loc_dof = ele[jx]

    for p in range(0,4):

        for q in range(0,4):

            ai = loc_dof[p]

            bi = loc_dof[q]

            mass_fine[ai,bi] = mass_fine[ai,bi]+loc_mass[p,q]





mass_fine = mass_fine[int_dof_domain ,:]

mass_fine = mass_fine[:, int_dof_domain]
# 2d matrix; v_{prev x2 direction}; it is the fine solution at x2 = 0

# pre-defined for x2 = bnd and solution for all times

# nb times, dim for x1*x3 directions

# this is used for the first iteration from 0 to first time step; then will be set to 0

v_xknown = []



Massf_notime = np.matmul(mass_fine, f_notime_x2)



# load R0 before; load RtMR0

fcoarse = np.matmul( LA.inv(RtMR0)    , np.matmul( np.transpose(R0) , Massf_notime)  )



for tx in range(nt):

    v_xknown.append(fcoarse*np.sin(tx*dt))

print("v_xknown", np.shape(v_xknown))
v_xknown[0] = 0

v_xknown[1] = 0
# 2d matrix; v_{prev x2 direction}; it is the coarse solution

# repeatedly define (x2, for all times); initial is 0 and is not contributing to the computing

# nb times, coarse scale dimension for x1 x3 directions



seg1 = 0

print("seg1", seg1)

seg2 = int(1/3*nx2)

print("seg2", seg2)

seg3 = int(1/3*nx2)*2

print("seg3", seg3)



nb_coarse_dofs = np.shape(  np.array(fcoarse) )[0]

# first iteration; fine solution is given;

u_tmnl = []

# v_{x2 direction, n+1}^{t, k}

v_xnp1_tk = np.zeros(  nb_coarse_dofs  )

# v_{x2 direction, n+1}^{t, k-1}

v_xnp1_tkm1 = np.zeros( nb_coarse_dofs  )



# v at n+1 in x2 direction; save for all times; will be updated at every tiem iteration; coarse solution

# each row is the solution at x2_{n+1} level at a certain time

v_xnp1 = []

    

for tx in tqdm( range(1, nt-1)  ):

    f_ = 1/(dt*dx2*coeff_l)*(v_xknown[tx+1]-v_xknown[tx] )

    uM = 2/(dt*dt*coeff_l)*v_xnp1_tk+   ( 1/(2*dt*dx2)-1/(dt*dt) )*v_xnp1_tkm1/coeff_l



    invMSu = -1/(2*coeff_l)*np.matmul( invMS0, v_xnp1_tk)



    v_xnp1_tkm1 = v_xnp1_tk

    v_xnp1_tk = f_ + uM + invMSu



    v_xnp1.append(v_xnp1_tk)

v_xknown_H = v_xnp1

u_tmnl.append(v_xnp1_tk)
print(np.shape(v_xknown_H))
# coarse solutions in x1 x3 directions at all terminal time

# dim x2 directions, dim coarse grid solution



for ix2 in tqdm(range(1, nx2)):

    

    if ix2 == 1:        

        invMS = invMS0

    elif ix2 == seg2:

        invMS = invMS1

    elif ix2 == seg3:

        invMS = invMS2

                

    # v_{x2 direction, n+1}^{t, k}

    v_xnp1_tk = np.zeros(  nb_coarse_dofs  )

    # v_{x2 direction, n+1}^{t, k-1}

    v_xnp1_tkm1 = np.zeros( nb_coarse_dofs  )

    

    

    

    # solution at n+1 in x2 direction; save for all times; will be updated at every tiem iteration; coarse solution

    # each row is the solution at x2_{n+1} level at a certain time

    v_xnp1 = []

    

    f_ = 1/(dt*dx2*coeff_l)*(v_xknown_H[0] )

    uM = 2/(dt*dt*coeff_l)*v_xnp1_tk+   ( 1/(2*dt*dx2)-1/(dt*dt) )*v_xnp1_tkm1/coeff_l



    invMSu = -1/(2*coeff_l)*np.matmul( invMS, v_xnp1_tk)



    v_xnp1_tkm1 = v_xnp1_tk

    v_xnp1_tk =  uM + invMSu

    v_xnp1.append(v_xnp1_tk)

    

    f_ = 1/(dt*dx2*coeff_l)*(v_xknown_H[1]-v_xknown_H[0] )

    uM = 2/(dt*dt*coeff_l)*v_xnp1_tk+   ( 1/(2*dt*dx2)-1/(dt*dt) )*v_xnp1_tkm1/coeff_l



    invMSu = -1/(2*coeff_l)*np.matmul( invMS, v_xnp1_tk)



    v_xnp1_tkm1 = v_xnp1_tk

    v_xnp1_tk = f_ + uM + invMSu



    v_xnp1.append(v_xnp1_tk)



    

    for tx in  range(2, nt-2):



        f_ = 1/(dt*dx2*coeff_l)*(v_xknown_H[tx]-v_xknown_H[tx-1] )

        uM = 2/(dt*dt*coeff_l)*v_xnp1_tk+   ( 1/(2*dt*dx2)-1/(dt*dt) )*v_xnp1_tkm1/coeff_l



        invMSu = -1/(2*coeff_l)*np.matmul( invMS, v_xnp1_tk)



        v_xnp1_tkm1 = v_xnp1_tk

        v_xnp1_tk = f_ + uM + invMSu



        v_xnp1.append(v_xnp1_tk)

    

    u_tmnl.append(v_xnp1_tk)

    v_xknown_H = v_xnp1

print(np.shape(u_tmnl))
np.save("u_tmnl", u_tmnl)