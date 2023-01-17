# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/ml-in-graphs-hw0/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!python -m pip install snap-stanford
import snap
G_small = snap.TNGraph.New()
G_small.AddNode(1)
G_small.AddNode(2)
G_small.AddNode(3)

G_small.AddEdge(1,2)
G_small.AddEdge(2,1)
G_small.AddEdge(1,3)
G_small.AddEdge(1,1)
path = '../input/ml-in-graphs-hw0/wiki-Vote.txt'
G = snap.LoadEdgeList(snap.PNGraph, path, 0, 1)
# The number of nodes in the network. (G small has 3 nodes.)
G.GetNodes()
# The number of nodes with a self-edge (self-loop), i.e., the number of nodes a ∈ V where (a, a) ∈ E. (G small has 1 self-edge.)
snap.CntSelfEdges(G)
# The number of directed edges in the network, i.e., the number of ordered pairs (a, b) ∈ E for which a 6 = b. (G small has 3 directed edges.)
snap.CntUniqDirEdges(G)
# The number of undirected edges in the network, i.e., the number of unique unordered pairs (a, b), a 6 = b, for which (a, b) ∈ E or (b, a) ∈ E (or both). If both (a, b) and (b, a) are edges, this counts a single undirected edge. (G small has 2 undirected edges.)
snap.CntUniqUndirEdges(G)
# The number of reciprocated edges in the network, i.e., the number of unique unordered pairs of nodes (a, b), a 6 = b, for which (a, b) ∈ E and (b, a) ∈ E. (G small has 1 reciprocated edge.)
snap.CntUniqBiDirEdges(G)
# The number of nodes of zero out-degree. (G small has 1 node with zero out-degree.)
DegToCntV = snap.TIntPrV()
snap.GetOutDegCnt(G, DegToCntV)
cnt=0
for item in DegToCntV:
    if item.GetVal1()==0:
        cnt=item.GetVal2()
print(cnt)    
# The number of nodes of zero in-degree. (G small has 0 nodes with zero in-degree.)
DegToCntV = snap.TIntPrV()
snap.GetInDegCnt(G, DegToCntV)
cnt=0
for item in DegToCntV:
    if item.GetVal1()==0:
        cnt=item.GetVal2()
print(cnt)
# The number of nodes with more than 10 outgoing edges (out-degree > 10). 
DegToCntV = snap.TIntPrV()
snap.GetOutDegCnt(G, DegToCntV)
cnt=0
for item in DegToCntV:
    if item.GetVal1()>10:
        cnt+=item.GetVal2()
print(cnt)
# The number of nodes with fewer than 10 incoming edges (in-degree < 10).
DegToCntV = snap.TIntPrV()
snap.GetInDegCnt(G, DegToCntV)
cnt=0
for item in DegToCntV:
    if item.GetVal1()<10:
        cnt+=item.GetVal2()
print(cnt)
from matplotlib import pyplot
DegToCntV = snap.TIntPrV()
snap.GetOutDegCnt(G, DegToCntV)
x=[]
y=[]
for item in DegToCntV:
    if item.GetVal1()>0:
        x.append(np.log10(item.GetVal1()))
        y.append(np.log10(item.GetVal2()))
pyplot.plot(x,y)
pyplot.xlabel('Out-deegree')
pyplot.ylabel('Number of Nodes')
pyplot.title('Distribution of out-degrees')
a,b = np.polyfit(x, y, 1.3)
print(a,b)
pyplot.plot(x,y)
pyplot.xlabel('Out-deegree')
pyplot.ylabel('Number of Nodes')
pyplot.title('Distribution of out-degrees')

x2=np.linspace(min(x),max(y),100)
pyplot.plot(x2,a*x2+b)
pyplot.legend(['Data line','Regression line'])

path = '../input/ml-in-graphs-hw0/stackoverflow-Java.txt'
G2 = snap.LoadEdgeList(snap.PNGraph, path, 0, 1)
G2.GetNodes()
G2.GetEdges()
# The number of weakly connected components in the network.
Components = snap.TCnComV()
snap.GetWccs(G2, Components)
Components.Len()
# The number of edges and the number of nodes in the largest weakly connected component. 
MxWcc = snap.GetMxWcc(G2)
print(MxWcc.GetEdges(), MxWcc.GetNodes())
# IDs of the top 3 most central nodes in the network by PagePank scores.
PRankH = snap.TIntFltH()
snap.GetPageRank(G2, PRankH)
pr_list=[]
for item in PRankH:
    pr_list.append([item, PRankH[item]])
pr=pd.DataFrame(pr_list,columns=['Node','PageRank']).sort_values(by='PageRank',ascending=0)
pr[0:3]
# IDs of the top 3 hubs and top 3 authorities in the network by HITS scores.
NIdHubH = snap.TIntFltH()
NIdAuthH = snap.TIntFltH()
snap.GetHits(G2, NIdHubH, NIdAuthH)
hubs_list=[]
aths_list=[]
for item in NIdHubH:
    hubs_list.append([item, NIdHubH[item]])
for item in NIdAuthH:
    aths_list.append([item, NIdAuthH[item]])
hubs=pd.DataFrame(hubs_list,columns=['Node','HITS']).sort_values(by='HITS',ascending=0)
aths=pd.DataFrame(aths_list,columns=['Node','HITS']).sort_values(by='HITS',ascending=0)
hubs[:3]
aths[:3]