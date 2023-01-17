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
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
train.head()
n=100
from collections import Counter

top_n = Counter([item for sublist in train.ingredients for item in sublist]).most_common(n)
top_n
import networkx as nx
import random
from itertools import combinations

G=nx.Graph()

G.clear()

random_pick_from_top_n = random.sample(top_n, 10)

for list_of_nodes in train.ingredients:
    filtered_nodes = set(list_of_nodes).intersection(set([x[0] for x in random_pick_from_top_n]))  
    for node1,node2 in list(combinations(filtered_nodes,2)): 
        G.add_node(node1)
        G.add_node(node2)
        G.add_edge(node1, node2)

nx.draw_networkx(G)

