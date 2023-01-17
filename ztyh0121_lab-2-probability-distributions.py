from subprocess import check_output

print(check_output(["ls", ".."]).decode("utf8"))
import numpy as np

"""

np.random.uniform(size=10)

np.random.uniform(3,4,5)

np.random.normal(size=4)

np.random.normal(-2,2,20)

np.random.binomial(10,0.1,size=2)

np.random.hypergeometric(10,20,5,10)

np.random.geometric(0.2,10)

np.random.negative_binomial(5,0.1,size=10)

np.random.poisson(5,10)

np.random.exponential(2,10)

np.random.chisquare(10,10)

np.random.f(3,4,20)

np.random.standard_t(3,10)

"""
from scipy.stats import norm

nrm=norm(4,3)

sample=nrm.rvs(10)

sample
sample.mean()
sample.std()
nrm.pdf(4)
nrm.pdf(4)
nrm.logpdf(7)
nrm.cdf(4)
nrm.logcdf(0)
nrm.ppf(0.5)
nrm.pdf(np.arange(2,5,0.2))
nrm.cdf([1,7])
from scipy.stats import binom

bnm=binom(10,0.3)

sample=bnm.rvs(10)

sample
sample.mean()
sample.std()
bnm.pmf(3)
bnm.cdf(2)
bnm.ppf(0.3)
bnm.pmf(range(11))
bnm.cdf(range(11))
from sklearn.datasets import load_boston

boston=load_boston()

import pandas as pd

data=pd.DataFrame(boston.data, columns=boston.feature_names)

boston.target
data.head(2)
data['CRIM'].describe()
import matplotlib.pyplot as plt

plt.hist(data['CRIM'],bins=50,density=True)

plt.show()
plt.hist(data['CRIM'],bins=20,density=False)

plt.show()
import scipy.stats as stats

import pylab

nrm=norm(11,5)

sample=nrm.rvs(50)

stats.probplot(sample, dist="norm", plot=pylab)

pylab.show()
norm(0,1).ppf(0.5**(1/50))
norm(0,1).ppf(1 - 0.5**(1/50))
"""

Filliben's estimate

0.5**(1/n) for i = n

(i - 0.3175) / (n + 0.365) for i = 2, ..., n-1

1 - 0.5**(1/n) for i = 1

"""
help(stats.probplot)
#E[X+Y]=EX+EY

nrm1=norm(-2,5)

nrm2=norm(4,2)

sample1=nrm1.rvs(100)

sample2=nrm2.rvs(100)

sample1.shape
sample2.shape
(sample1+sample2).shape
(sample1+sample2).mean()
sample1.mean()+sample2.mean()
#V(X+Y)=V(X)+V(Y)

(sample1+sample2).var()
sample1.var()+sample2.var()
nrm1.rvs(10).mean()
nrm1.rvs(100).mean()
nrm1.rvs(1000).mean()
nrm1.rvs(10000).mean()
nrm1.rvs(int(1e5)).mean()
nrm1.rvs(int(1e6)).mean()
nrm1.rvs(int(1e7)).mean()
nrm1.rvs(int(1e8)).mean()
1e1
1e2
from scipy.stats import expon

import math

result=np.zeros(5000)

for i in range(5000):

    x=expon(loc=0,scale=1).rvs(1000)

    result[i]=(sum(x)-1000)/math.sqrt(1000)
plt.hist(result,bins=100,density=True)

plt.show()
stats.probplot(result, dist="norm", plot=pylab)

pylab.show()
np.random.randint(0,4,10)
np.random.choice([3,8,1],10)
np.random.choice([3,8,1],3,replace=False)
np.random.choice([3,8,1],3,replace=True,p=[0.05,0.05,0.9])
# other useful things for the lab assignment

sum(bnm.pmf(range(1,5)))
bnm.cdf(4)-bnm.cdf(0)
from scipy.stats import poisson

poisson(2).rvs(100).mean()
poisson(2).pmf(range(10))
poisson(2).cdf(range(10))
alphas=np.arange(0.1,1,0.1)

nalphas=-alphas[::-1]

alphas
nalphas
combined=np.concatenate([alphas,nalphas])

combined
stats.probplot(combined, dist="norm", plot=pylab)

pylab.show()
from scipy.stats import nbinom

nbnm=nbinom(5,0.1)

# success probability 0.1 

# stop when you've achieved 5 success

# counts number of failures
nbnm.rvs(20)
nbinom.stats(5, 0.1, moments='mvsk')
from scipy.stats import uniform

unfm=uniform(0,1)

unfm.rvs(20)