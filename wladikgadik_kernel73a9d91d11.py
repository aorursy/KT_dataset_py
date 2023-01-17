#!pip install ipywidgets pyvis jsonpickle
import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from pyvis.network import Network
import networkx as nx
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

data = pd.read_csv('../input/_vs___.xlsx - Page1.csv', index_col=[0,1,2,3,4,5], skipinitialspace=True,na_values=0)
data = data.fillna(0)
c = data.index.names
data = data.reset_index().fillna('').set_index(c)
data.index = [' '.join(col).strip() for col in data.index.values]

@interact
def main(INIT_ROLE = data.columns):
    init_vect = np.array(data[INIT_ROLE].fillna(0)).reshape(-1, 1)
    simlist = []
    for role in data.columns:
        trg_vect = np.array((data[role].fillna(0))).reshape(-1, 1)
        sim = cosine(trg_vect, init_vect)
        simlist.append(sim)
    res = pd.DataFrame(data={'Роли': data.columns,'Похожесть':simlist}).sort_values('Похожесть')
    
    return res[1:11]
@interact
def main2(INIT_ROLE = data.columns, TARGET_ROLE=data.columns):
    df = data[[INIT_ROLE,TARGET_ROLE]].copy()
    df['Разница'] = data[TARGET_ROLE] - data[INIT_ROLE]
    res = df.sort_values('Разница', ascending=False)
    return res
def roles_pairs(set_of_roles):
    pairs = []
    for i in range(len(set_of_roles)):
        for j in range(i+1, len(set_of_roles)):
            pairs.append((set_of_roles[i], set_of_roles[j]))
    return pairs
def init_pairs(set_of_roles, INIT_ROLE):
    pairs = []
    for i in range(len([INIT_ROLE])):
        for j in range(i+1, len(set_of_roles)):
            pairs.append(([INIT_ROLE][i], set_of_roles[j]))
    return pairs
df_dict = {}
for INIT_ROLE in data.columns:
    init_vect = np.array(data[INIT_ROLE].fillna(0)).reshape(-1, 1)
    simlist = []
    for role in data.columns:
        trg_vect = np.array((data[role].fillna(0))).reshape(-1, 1)
        sim = cosine(trg_vect, init_vect)
        simlist.append(sim)
    res = pd.DataFrame(data={'Роли': data.columns,'Похожесть':simlist}).sort_values('Похожесть')
    res['node_title'] = res['Роли'] +': ' + res['Похожесть'].apply(np.around, args=({3})).astype(str).values
    df_dict[INIT_ROLE] = res
@interact
def graph(INIT_ROLE=data.columns):
    got_net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    got_net.hrepulsion()
    #got_net.barnes_hut()
    got_net.show_buttons(filter_=['physics'])
    trg_df = df_dict[INIT_ROLE][:11]
    pairs = init_pairs(trg_df['Роли'].values, INIT_ROLE)
    for e in pairs:
        src = e[0]
        dst = e[1]
        #init_vect = np.array(data[src].fillna(0)).reshape(-1, 1)
        #trg_vect = np.array((data[dst].fillna(0))).reshape(-1, 1)
        w = trg_df[trg_df['Роли'] == dst]['Похожесть'].values[0]
        got_net.add_node(src, src, title=src)
        got_net.add_node(dst, dst, title=dst)
        got_net.add_edge(src, dst, width=w*w*100, title='')

    neighbor_map = got_net.get_adj_list()
    
    for node in got_net.nodes:
        node["title"] += " Похожие роли:<br>" + "<br>".join(trg_df['node_title'][1:11].values)
        node["value"] = len(neighbor_map[node["id"]])
        
    for edge in got_net.edges:
        df = data[[edge['from'],edge['to']]].copy()
        df['Разница'] = data[edge['to']] - data[edge['from']]
        res = df.sort_values('Разница', ascending=False)
        res['edge_title'] = res.index +': ' + res['Разница'].astype(str).values
        edge["title"] += " Инструменты:<br>" + "<br>".join(res['edge_title'][:10].values)
        #edge["value"] = len(neighbor_map[node["id"]])
        
    return got_net.show("roles.html")

@interact
def graph():
    got_net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    got_net.hrepulsion()
    #got_net.barnes_hut()
    got_net.show_buttons(filter_=['physics'])
    pairs = roles_pairs(data.columns)
    for e in pairs:
        src = e[0]
        dst = e[1]
        #init_vect = np.array(data[src].fillna(0)).reshape(-1, 1)
        #trg_vect = np.array((data[dst].fillna(0))).reshape(-1, 1)
        w = 1 - df_dict[src][df_dict[src]['Роли'] == dst]['Похожесть'].values[0]
        got_net.add_node(src, src, title=src)
        got_net.add_node(dst, dst, title=dst)
        if w>0.8:
            got_net.add_edge(src, dst, width=w*w, title='')

    neighbor_map = got_net.get_adj_list()
    
    for node in got_net.nodes:
        node["title"] += " Похожие роли:<br>" + "<br>".join(neighbor_map[node["id"]])
        node["value"] = len(neighbor_map[node["id"]])
        
    for edge in got_net.edges:
        df = data[[edge['from'],edge['to']]].copy()
        df['Разница'] = data[edge['to']] - data[edge['from']]
        res = df.sort_values('Разница', ascending=False)
        res['edge_title'] = res.index +': ' + res['Разница'].astype(str).values
        edge["title"] += " Инструменты:<br>" + "<br>".join(res['edge_title'].values)
        #edge["value"] = len(neighbor_map[node["id"]])
        
    return got_net.show("roles.html")











