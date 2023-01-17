%matplotlib inline

import networkx as nx

import pandas as pd

from itertools import combinations



G = nx.Graph()

df = pd.read_csv('../input/train.csv').fillna("")



edges = [tuple(x) for x in df[['question1', 'question2']].values]

G.add_edges_from(edges)



map_label = dict(((x[0], x[1]), x[2]) for x in df[['question1', 'question2', 'is_duplicate']].values)

map_clique_size = {}

cliques = sorted(list(nx.find_cliques(G)), key=lambda x: len(x))

for cli in cliques:

    for q1, q2 in combinations(cli, 2):

        if (q1, q2) in map_label:

            map_clique_size[q1, q2] = len(cli)

        elif (q2, q1) in map_label:

            map_clique_size[q2, q1] = len(cli)

 

df['clique_size'] = df.apply(lambda row: map_clique_size.get((row['question1'], row['question2']), -1), axis=1)
# Average true rate of clique. Large cliques have many of true data. 

df.groupby('clique_size')['is_duplicate'].mean().plot(kind='bar')
# clique size count

df.groupby('clique_size')[['is_duplicate']].count().plot(kind='bar')
df['is_greater2'] = df['clique_size'] > 2

df.groupby('is_greater2')[['is_duplicate']].sum() / df['is_duplicate'].sum()