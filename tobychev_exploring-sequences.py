import numpy  as np
import pandas as pd
import scipy.stats as sst
import sklearn.neighbors as skn

import matplotlib.pyplot as pl
from subprocess import check_output

def stoarray(data = [], sep = ','):
    return data.map(lambda x: np.array(x.split(sep), dtype=float))

# load the data
colna = ['id', 'seq']
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
test.columns = colna
train.columns = colna
train['seq'], test['seq'] = stoarray(train['seq']), stoarray(test['seq']) 

train.head()
train["last"] = train['seq'].apply(lambda x: x[-1])
train["length"] = train['seq'].apply(len)
print(train["last"].describe())
print("95%       {:.6e}".format(train["last"].quantile(0.95)))
print("% < 0     {:.6f}".format((train["last"] < 0).sum()/len(train["last"])) )
print("""Ends between 0,16: 
          {:.6f}""".format( ((train["last"] < 17  ) & (train["last"] >= 0 )).sum() /len(train["last"])) )

kde = skn.KernelDensity(kernel='gaussian', bandwidth=0.75).fit(train["length"].reshape(-1,1))
x = np.linspace(0,train["length"].max(),1000).reshape(-1,1)
y_kde = np.exp( kde.score_samples(x) ) 
pl.figure(figsize=(16,8))
pl.title("Distribution of sequence lengths")
pl.plot(x,y_kde)
kde = skn.KernelDensity(kernel='gaussian', bandwidth=0.75).fit(train["last"].reshape(-1,1))
x = np.linspace(0,100,1000).reshape(-1,1)
y_kde = np.exp( kde.score_samples(x) ) 
pl.figure(figsize=(16,8))
pl.title("Distribution of sequence endings below 100")
pl.plot(x,y_kde)
#x = np.linspace(0,1e5,1e7).reshape(-1,1)
y_kde = np.exp( kde.score_samples(x) ) 
pl.figure(figsize=(16,8))
pl.title("Distribution of sequence endings, log scale")
pl.loglog(x,y_kde)
x = np.logspace(0,27,1e3).reshape(-1,1)
y_kde = np.exp( kde.score_samples(x) ) 
pl.figure(figsize=(16,8))
pl.title("Distribution of sequence endings, log scale")
pl.loglog(x,y_kde)