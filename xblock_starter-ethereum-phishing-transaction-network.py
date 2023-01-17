# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import platform

import pickle

import networkx as nx
print("python version:\t\t", platform.python_version())

print("networkx version:\t", nx.__version__)
def load_pickle(fname):

    with open(fname, 'rb') as f:

        return pickle.load(f)

    

G = load_pickle('../input/ethereum-phishing-transaction-network/Ethereum Phishing Transaction Network/MulDiGraph.pkl')

print(nx.info(G))
# Traversal nodes:

for idx, nd in enumerate(G.nodes):

    # print the current node.

    print("node:", nd)

    # print node label ∈{0,1}

    # 1 represents phishing node, 0 represents the node of unknown tag.

    print("label:", G.nodes[nd]['isp'])

    break
# Travelsal edges:

for ind, edge in enumerate(nx.edges(G)):

    # gets the nodes on both sides of the edge.

    (u, v) = edge

    # gets the first edge from node u to node v.

    eg = G[u][v][0]

    # gets the properties of the directed edge: the amount and timestamp of the transaction.

    amo, tim = eg['amount'], eg['timestamp']

    print("amount:", amo)

    print("timestamp:", tim)

    break
# get the graph adjacency matrix(where a_ij means the number of edges of i to j) as a SciPy sparse matrix.

# more operations can refer to the networkx help documentation.

sparse_adj_matrix = nx.to_scipy_sparse_matrix(G)
print(sparse_adj_matrix)