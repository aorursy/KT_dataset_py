import pylab as pl
import numpy as np

def pxgiveny(y,mx,my,s1,s2):
    return np.random.normal(mx + (y-my)/s2,s1)
    #return random.binomial(16,y,1)

def pygivenx(x,mx,my,s1,s2):
    return np.random.normal(my + (x-mx)/s1,s2)
    #return random.beta(x+2,16-x+4,1)

def gibbs(N=500):
    k=10
    x0 = np.zeros(N,dtype=float)
    m1 = 10
    m2 = 20
    s1 = 2
    s2 = 3
    for i in range(N):
        y = np.random.rand(1)
        for j in range(k):
            x = pxgiveny(y,m1,m2,s1,s2)
            y = pygivenx(x,m1,m2,s1,s2)
        x0[i] = x
    
    return x0


def f(x):
    return np.exp(-(x-10)**2/10)

N=500
s=gibbs(N)
x1 = np.arange(0,17,1)
pl.hist(s,bins=x1,fc='k')
x1 = np.arange(0,17,0.1)
px1 = np.zeros(len(x1))
for i in range(len(x1)):
    px1[i] = f(x1[i])
pl.plot(x1, px1*N*10/np.sum(px1), color='k',linewidth=3)

pl.show()