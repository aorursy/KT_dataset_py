from copy import deepcopy
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pclust={}
plt.rcParams['figure.figsize']=(16,9)
data=pd.read_csv('../input/xclara.csv')
print(data.shape)
data.head(10)
f1=data['V1'].values
f2=data['V2'].values
X=np.array(list(zip(f1,f2)))
#plt.scatter(f1,f2)
clust={}
k=0
def sets(t):
    global clust
    global k
    k=t
    for i in range(k):
        clust[i]=[]
        clust[i].append(np.random.choice(f1))
        clust[i].append(np.random.choice(f2))
    plt.scatter(f1,f2,c='r')
    plt.title('SCATTER PLOT BEFORE CLUSTERING')
    plt.rcParams['figure.figsize']=(16,9)
    for i in range(0,k):
        plt.scatter(clust[i][0],clust[i][1],c='b',marker='s',s=200)
def dist(x,y):
    li=[]
    for i in range(0,k):
        v=np.power((clust[i][0]-x),2)+np.power((clust[i][1]-y),2)
        li.append(round(np.sqrt(v),4))
    return li
clusp={}
def call():
    global clusp
    for i in range(0,k):
        clusp[i]=[]
var=1
def comp():
    global var
    global clust
    global pclust
    global clusp
    call()
    for i in range(0, len(f1)):
        li=dist(f1[i],f2[i])
        mi=min(li)
        ix=li.index(mi)
        clusp[ix].append(i)
    clust=[avg(i) for i in range(0,k)]
    if pclust==clust:
        var=0
    pclust=clust
    return var,clusp
def avg(ix):
    x=[]
    y=[]
    for i in clusp[ix]:
        x.append(f1[i])
        y.append(f2[i])
    t1,t2=sum(x)/len(x),sum(y)/len(y)
    return [round(t1,4),round(t2,4)]
def cluster(kk):
    global var
    sets(kk)
    call()
    while var!=0:
        var,clp=comp()
def printcluster():
    global k
    global clust
    global clusp
    c=['y','r','b','c','m','k','w','y','r','m','c']
    kk=[[] for i in range(k)]
    jj=[[] for i in range(k)]
    for i in range(0,k):
        for j in clusp[i]:
            kk[i].append(f1[j])
            jj[i].append(f2[j])
    for i in range(k):
        plt.scatter(kk[i],jj[i],c=c[i])
        plt.scatter(clust[i][0],clust[i][1],marker='^',c='g',s=200)
    _=plt.title('SCATTER PLOT SHOWING {0} CLUSTERS'.format(k))
cluster(6)
printcluster()
