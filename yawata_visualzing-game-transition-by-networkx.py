import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import networkx as nx

from tqdm import tqdm

import gc
title = ['Cart Balancer (Assessment)','Chest Sorter (Assessment)','Cauldron Filler (Assessment)','Bird Measurer (Assessment)','Mushroom Sorter (Assessment)',\

         'Chicken Balancer (Activity)','Egg Dropper (Activity)','Sandcastle Builder (Activity)','Bottle Filler (Activity)','Watering Hole (Activity)','Bug Measurer (Activity)',\

         'Fireworks (Activity)','Flower Waterer (Activity)','Crystal Caves - Level 3','Honey Cake','Lifting Heavy Things','Crystal Caves - Level 2','Heavy, Heavier, Heaviest',\

         'Balancing Act','Crystal Caves - Level 1','Magma Peak - Level 1','Slop Problem','Magma Peak - Level 2','Welcome to Lost Lagoon!','Costume Box','Pirate\'s Tale',\

         'Tree Top City - Level 2','Tree Top City - Level 3','Treasure Map','12 Monkeys','Tree Top City - Level 1','Ordering Spheres','Rulers','Happy Camel','Leaf Leader',\

         'Chow Time','Pan Balance','Scrub-A-Dub','Bubble Bath','Dino Dive','Dino Drink','Air Show','All Star Sorting','Crystals Rule']

title_dic = dict(zip(title,np.arange(len(title))))



train_all = pd.read_csv('../input/data-science-bowl-2019/train.csv')

# I want to see only game trainsition. Therefore, I remove events whose event_count is not 1.

train = train_all[train_all['event_count']==1].reset_index(drop=True)

train = train.drop(['game_session','event_data','event_count','event_code','game_time','type','world'],axis=1)

del train_all

gc.collect()

train.head()

for c in ['installation_id','title','timestamp']:

    train['previous_'+c] = ''

    train.loc[train.index[1:],'previous_'+c]=np.array(train.loc[train.index[:len(train.index)-1],c])

train['title'] = train['title'].apply(lambda x:title_dic[x] if x in title_dic.keys() else -1)

train['previous_title'] = train['previous_title'].apply(lambda x:title_dic[x] if x in title_dic.keys() else -1)

print('datasize: ',len(train))

train = train[train['title']>=0]

train = train[train['previous_title']>=0]

print('datasize: ',len(train))

train = train[train['installation_id']==train['previous_installation_id']]

print('datasize: ',len(train))

train.head()
link_count = np.zeros([len(title_dic),len(title_dic)],dtype=np.int)

node_count = np.zeros([len(title_dic)])

for i in tqdm(train.index):

    link_count[train.loc[i,'previous_title']][train.loc[i,'title']]+=1

    node_count[train.loc[i,'title']] += 1
plt.figure(figsize=(10,8))

sns.heatmap(link_count)

plt.show()
Graph = nx.DiGraph()

Graph.add_nodes_from(title)

weight = []

for i in range(len(title)):    

    for j in range(i+1,len(title)):

        if link_count[title_dic[title[i]]][title_dic[title[j]]]>100:

            Graph.add_edge(title[i],title[j])

            weight.append(np.log(link_count[title_dic[title[i]]][title_dic[title[j]]])/2)

pos = dict(zip(title,[[np.cos(2*np.pi*i/(len(title))),np.sin(2*np.pi*i/(len(title)))] for i in range(len(title))]))

plt.figure(figsize=(15,15))

node_color = ['cyan' for i in range(5)]

node_color += ['greenyellow' for i in range(8)]

node_color += ['orange' for i in range(20)]

node_color += ['magenta' for i in range(11)]



nx.draw_networkx(Graph,pos=pos,node_color=node_color,font_size=8,edge_color=weight,edge_cmap=plt.cm.autumn_r,node_size=np.array(node_count)/5)

plt.title('Game Trainsition')

plt.show()
Graph = nx.DiGraph()

Graph.add_nodes_from(title)

weight = []

for i in range(len(title)):    

    for j in range(0,5):

        if link_count[title_dic[title[i]]][title_dic[title[j]]]>10:

            Graph.add_edge(title[i],title[j])

            weight.append(np.log(link_count[title_dic[title[i]]][title_dic[title[j]]])/2)

pos = dict(zip(title,[[np.cos(2*np.pi*i/(len(title))),np.sin(2*np.pi*i/(len(title)))] for i in range(len(title))]))

plt.figure(figsize=(15,15))

node_color = ['cyan' for i in range(5)]

node_color += ['greenyellow' for i in range(8)]

node_color += ['orange' for i in range(20)]

node_color += ['magenta' for i in range(11)]



nx.draw_networkx(Graph,pos=pos,node_color=node_color,font_size=8,edge_color=weight,edge_cmap=plt.cm.autumn_r,node_size=np.array(node_count)/5)

plt.title('Game Transition to Assessment')

plt.show()