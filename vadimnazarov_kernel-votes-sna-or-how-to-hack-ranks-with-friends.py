# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import networkx as nx

import seaborn as sns
scripts = pd.read_csv("../input/Scripts.csv")

script_versions = pd.read_csv("../input/ScriptVersions.csv")

script_votes = pd.read_csv("../input/ScriptVotes.csv")
vers_votes = pd.merge(script_versions, script_votes, left_on = "Id", right_on = "ScriptVersionId")

vers_votes.head()
scripts_full = pd.merge(scripts, vers_votes, left_on = "Id", right_on = "ScriptId")

scripts_full.head()
edges = scripts_full[["AuthorUserId", "UserId"]]

edges
edges_group = edges.groupby("UserId").apply(len)

sns.distplot(edges_group)
edges_group
edges_group.index[edges_group >= 5]
edges_filter = edges.loc[edges.AuthorUserId.isin(edges_group.index[edges_group >= 5]), :]

edges_filter
edges_dup = edges_filter.groupby(['AuthorUserId', 'UserId'], as_index = False).size().reset_index()

edges_dup.columns = ['AuthorUserId', 'UserId', 'Size']

edges_dup
edges_dup = edges_dup.loc[edges_dup.Size >= 5, :]

edges_dup
G = nx.Graph()

G.clear()

for i in range(len(edges_dup)):

    G.add_edge(edges_dup.iloc[i, 0], edges_dup.iloc[i, 1])

# Remove self-loops, i.e. user which voted for themselves

G.remove_edges_from(G.selfloop_edges())

pos = nx.spring_layout(G)

nx.draw_networkx_labels(G, pos, font_size = 7)

nx.draw(G, pos=pos)
G = nx.Graph()

G.clear()

for i in range(len(edges_dup)):

    G.add_edge(edges_dup.iloc[i, 0], edges_dup.iloc[i, 1])

# Remove self-loops, i.e. user which voted for themselves

G.remove_edges_from(G.selfloop_edges())

G_5cores = nx.k_core(G, 4)

pos = nx.spring_layout(G_5cores)

nx.draw_networkx_labels(G_5cores, pos, font_size = 7)

nx.draw(G_5cores, pos=pos)
pairs = {}

for i in range(len(edges_dup)):

    src, tgt = edges_dup.iloc[i, 0], edges_dup.iloc[i, 1]

    if (tgt, src) in pairs:

        pairs[src, tgt] = True

    else:

        pairs[src, tgt] = False



interesting = list(filter(lambda x: x[1], pairs.items()))

interesting