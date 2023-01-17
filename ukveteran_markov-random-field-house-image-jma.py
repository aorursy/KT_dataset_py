import pylab as pl
import numpy as np

def MRF(I,J,eta=2.0,zeta=1.5):
    ind =np.arange(np.shape(I)[0])
    np.random.shuffle(ind)
    orderx = ind.copy()
    np.random.shuffle(ind)

    for i in orderx:
        for j in ind:
            oldJ = J[i,j]
            J[i,j]=1
            patch = 0
            for k in range(-1,1):
                for l in range(-1,1):
                    patch += J[i,j] * J[i+k,j+l]
            energya = -eta*np.sum(I*J) - zeta*patch
            J[i,j]=-1
            patch = 0
            for k in range(-1,1):
                for l in range(-1,1):
                    patch += J[i,j] * J[i+k,j+l]
            energyb = -eta*np.sum(I*J) - zeta*patch
            if energya<energyb:
                J[i,j] = 1
            else:
                J[i,j] = -1
    return J
            
I = pl.imread('../input/house-image/House.png')
N = np.shape(I)[0]
I = I[:,:,0]
I = np.where(I<0.1,-1,1)
pl.imshow(I)
pl.title('Original Image')

noise = np.random.rand(N,N)
J = I.copy()
ind = np.where(noise<0.1)
J[ind] = -J[ind]
pl.figure()
pl.imshow(J)
pl.title('Noisy image')
newJ = J.copy()
newJ = MRF(I,newJ)
pl.figure()
pl.imshow(newJ)
pl.title('Denoised version')
print (np.sum(I-J), np.sum(I-newJ))
pl.show()