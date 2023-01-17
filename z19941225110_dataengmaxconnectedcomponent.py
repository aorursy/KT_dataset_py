# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import networkx as nx
nodes=pd.read_csv('../input/nodes.csv')

edges=pd.read_csv('../input/edges.csv')
nodes=[x[0] for x in nodes.values]

edges=[tuple(x) for x in edges.values]
'''

undirected graph, because

https://networkx.github.io/documentation/stable/reference/classes/index.html

'''

G=nx.Graph()

G.add_nodes_from(nodes)

G.add_edges_from(edges)
len(G.nodes())
nx.number_connected_components(G)
#sort subgraphs

components=[c for c in sorted(nx.connected_components(G), key=len, reverse=True)]

#####print the line below to understand the line above

[len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)][0:10]
Gmax=G.subgraph(components[0])

# write edgelist to grid.edgelist

nx.write_edgelist(Gmax, path="MaxConn.edgelist", delimiter=":")

nx.write_graphml(Gmax, 'MaxConn.graphml')

nx.write_adjlist(Gmax, 'MaxConn.adjlist')

nx.write_multiline_adjlist(G,"MaxConn_multiline.adjlist")

nx.write_gexf(Gmax, 'MaxConn.gexf')

# nx.write_gml(Gmax, 'MaxConn.gml')

nx.write_gpickle(Gmax, 'MaxConn.gpickle')

nx.write_yaml(Gmax, 'MaxConn.yaml')

nx.write_pajek(Gmax, 'MaxConn.net')