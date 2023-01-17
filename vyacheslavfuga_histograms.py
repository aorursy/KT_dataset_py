import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import random

import matplotlib.patches as patches

%matplotlib inline

df = pd.read_csv('../input/cs448b_ipasn.csv')
grouped = df[['date','l_ipn','f']].groupby(['date','l_ipn'])

res = grouped[['f']].agg(['sum','count'])['f'][['sum','count']];
res=res.rename(columns={"sum": "connections_count", "count": "isp_count"})

print (res.groupby(level=1).sum().reset_index())

res = res.reset_index()     ## reset grouping

res.head(15)                ## show sample
df0=res

df0=df0.rename(columns={"connections_count":"f", "isp_count":"cnt"})

plt.rcParams['figure.figsize']=15,50

f,axes = plt.subplots(10, 2, sharey=False)



for i in range(10):

    b=20

    mu=df0[df0.l_ipn==i]["f"].values.mean()

    s=df0[df0.l_ipn==i]["f"].values.std()

    axes[i][0].set_title('Connections count for host: '+str(i)+', $ \mu=' + str(mu) + ', \sigma=' + str(s) +'$')

    axes[i][0].hist(df0[df0.l_ipn==i]["f"].values,bins=b,label='ISP count for host:')

    axes[i][0].axvline(mu,color='g',linewidth=2)

    #axes[i][0].axvline(mu+s+s+s,color='r',linewidth=1)

    rect = patches.Rectangle((mu,0), s, axes[i][0].get_ylim()[1], linewidth=0,facecolor='g',alpha=0.1)

    axes[i][0].add_patch(rect)

    rect = patches.Rectangle((mu+s,0), s, axes[i][0].get_ylim()[1], linewidth=0,facecolor='g',alpha=0.06)

    axes[i][0].add_patch(rect)

    rect = patches.Rectangle((mu+s+s,0), s, axes[i][0].get_ylim()[1], linewidth=0,facecolor='g',alpha=0.03)

    axes[i][0].add_patch(rect)

    

    mu=df0[df0.l_ipn==i]["cnt"].values.mean()

    s=df0[df0.l_ipn==i]["cnt"].values.std()   

    axes[i][1].set_title('ISP count for host: '+str(i)+', $ \mu=' + str(mu) + ', \sigma=' + str(s) +'$')

    axes[i][1].hist(df0[df0.l_ipn==i]["cnt"].values,bins=b)

    axes[i][1].axvline(mu,color='g',linewidth=2)

    #axes[i][1].axvline(mu+s+s+s,color='r',linewidth=1)    

    rect = patches.Rectangle((mu,0), s, axes[i][1].get_ylim()[1], linewidth=0,facecolor='g',alpha=0.1)

    axes[i][1].add_patch(rect)

    rect = patches.Rectangle((mu+s,0), s, axes[i][1].get_ylim()[1], linewidth=0,facecolor='g',alpha=0.06)

    axes[i][1].add_patch(rect)

    rect = patches.Rectangle((mu+s+s,0), s, axes[i][1].get_ylim()[1], linewidth=0,facecolor='g',alpha=0.03)

    axes[i][1].add_patch(rect)    
