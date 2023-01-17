import numpy as np
import math
from scipy.stats import bernoulli
from scipy.stats import norm
def berntest(n,p,p0,alpha):
    cnt=0
    for i in range(1000):
        Xsample=bernoulli.rvs(p, size=n)
        phat=np.mean(Xsample)
        Z=math.sqrt(n)*(phat-p0)/math.sqrt(phat*(1-phat))
        cnt+=Z>norm.ppf(1-alpha/2) or Z<norm.ppf(alpha/2)
    return cnt/1000
berntest(100,0.5,0.5,0.05)
print(berntest(100,0.3,0.3,0.05))
print(berntest(100,0.8,0.8,0.05))
print(berntest(100,0.1,0.1,0.05))
print(berntest(100,0.92,0.92,0.05))
for i in range(10,0,-1):
    print(berntest(10*i,0.5,0.5,0.05))
for i in range(9,0,-1):
    print(berntest(i,0.5,0.5,0.05))
print(berntest(100,0.6,0.5,0.05))
print(berntest(100,0.7,0.5,0.05))
print(berntest(100,0.8,0.5,0.05))
print(berntest(100,0.9,0.5,0.05))
for i in range(10000,80000,10000):
    print(berntest(i,0.51,0.5,0.05))
from scipy.stats import t
def normtest(n,mu,sigma,mu0):
    pvalues=np.zeros(1000)
    for i in range(1000):
        Xsample=norm.rvs(mu,sigma, size=n)
        muhat=np.mean(Xsample)
        T=math.sqrt(n)*(muhat-mu0)/np.std(Xsample,ddof=1)
        pvalues[i]=2*(1-t.cdf(abs(T),n-1))        
    return pvalues
import matplotlib.pyplot as plt
plt.hist(normtest(50,1,2,0),100)
plt.show()
def berntest2(n,p,p0):
    pvalues=np.zeros(1000)
    for i in range(1000):
        Xsample=bernoulli.rvs(p, size=n)
        phat=np.mean(Xsample)
        Z=math.sqrt(n)*(phat-p0)/math.sqrt(phat*(1-phat))
        pvalues[i]=1-norm.cdf(Z)        
    return pvalues
plt.hist(berntest2(100,0.5,0.5))
plt.show()