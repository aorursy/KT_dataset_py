import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f
Nsim=10000
n=10
m=20
Xsample=np.random.normal(2,1,(Nsim,n))
Ysample=np.random.normal(4,1,(Nsim,m))
Fstat=np.zeros(Nsim)
for i in range(Nsim):
    Fstat[i]=np.var(Xsample[i,:],ddof=1)/np.var(Ysample[i,:],ddof=1)
rv = f(n, m)
x = np.linspace(f.ppf(0.001, n, m), f.ppf(0.999, n, m), 1000)
plt.hist(Fstat,bins=64,density=True)
plt.plot(x, rv.pdf(x),lw=2)
plt.show()
from scipy.stats import norm
import math
def intreturn(n,sd,mean,alpha):
    Xsample=np.random.normal(mean,sd,n)
    interval=[np.mean(Xsample)-norm.ppf(1-alpha/2)*sd/math.sqrt(n),np.mean(Xsample)+norm.ppf(1-alpha/2)*sd/math.sqrt(n)]
    return interval
intreturn(100,3,2,0.05)
cnt=0
for i in range(1000):
    interval=intreturn(100,3,2,0.05)
    if interval[0]<2 and 2<interval[1]:
        cnt+=1
cnt/1000
from scipy.stats import t
def intreturn2(n,sd,mean,alpha):
    Xsample=np.random.normal(mean,sd,n)
    S=math.sqrt(np.var(Xsample,ddof=1))
    interval=[np.mean(Xsample)-t.ppf(1-alpha/2,n-1)*S/math.sqrt(n),np.mean(Xsample)+t.ppf(1-alpha/2,n-1)*S/math.sqrt(n)]
    return interval
intreturn2(100,3,2,0.05)
cnt=0
for i in range(1000):
    interval=intreturn2(100,3,2,0.05)
    if interval[0]<2 and 2<interval[1]:
        cnt+=1
cnt/1000
from scipy.stats import bernoulli
def intreturn3(n,p,alpha):
    Xsample=bernoulli.rvs(p, size=n)
    phat=np.mean(Xsample)
    sd=math.sqrt(phat*(1-phat))
    interval=[phat-norm.ppf(1-alpha/2)*sd/math.sqrt(n),phat+norm.ppf(1-alpha/2)*sd/math.sqrt(n)]
    return interval
intreturn3(100,0.2,0.05)
cnt=0
for i in range(1000):
    interval=intreturn3(100,0.2,0.05)
    if interval[0]<0.2 and 0.2<interval[1]:
        cnt+=1
cnt/1000
from scipy.stats import chi2
def intreturn4(n,sd,mean,alpha):
    Xsample=np.random.normal(mean,sd,n)
    S2=np.var(Xsample,ddof=1)
    interval=[(n-1)*S2/chi2.ppf(1-alpha/2,n-1),(n-1)*S2/chi2.ppf(alpha/2,n-1)]
    return interval
intreturn4(100,3,2,0.05)
cnt=0
for i in range(1000):
    interval=intreturn4(100,3,2,0.05)
    if interval[0]<9 and 9<interval[1]:
        cnt+=1
cnt/1000
cnt=0
for i in range(1000):
    Xsample=np.random.normal(2,3,100)
    cnt+=np.mean(Xsample)>2+norm.ppf(1-0.05)*3/10
cnt/1000
cnt=0
for i in range(1000):
    Xsample=np.random.normal(2,3,100)
    cnt+=10*(np.mean(Xsample)-2)/3>norm.ppf(1-0.05)
cnt/1000
cnt=0
for i in range(1000):
    Xsample=np.random.normal(2,3,100)
    cnt+=10*(np.mean(Xsample)-2)/3>norm.ppf(1-0.05/2) or 10*(np.mean(Xsample)-2)/3<norm.ppf(0.05/2)
cnt/1000
cnt=0
for i in range(1000):
    Xsample=np.random.normal(2,3,100)
    cnt+=10*(np.mean(Xsample)-2)/3<t.ppf(0.05,99)
cnt/1000
cnt=0
for i in range(1000):
    Xsample=np.random.normal(2,3,100)
    cnt+=10*(np.mean(Xsample)-2)/3>t.ppf(1-0.05/2,99) or 10*(np.mean(Xsample)-2)/3<t.ppf(0.05/2,99)
cnt/1000