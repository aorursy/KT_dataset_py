import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# for basic visualizations

import matplotlib.pyplot as plt

import seaborn as sns



# for network visualizations

import networkx as nx

import community
# create graph from the dataset

fb_net = nx.read_edgelist('/kaggle/input/facebook_combined.txt', create_using = nx.DiGraph(), nodetype = int)
# print the graph info

print(nx.info(fb_net))
n = 4039

fb_in = fb_net.in_degree()

fb_out = fb_net.out_degree()

in_degrees = []

out_degrees = []

for i in range(n):

    in_degrees.append(fb_in[i])

    out_degrees.append(fb_out[i])

ids = [i for i in range(n)]



plt.title("in degree")

plt.xlabel("in degree")

plt.ylabel("count")

plt.hist(in_degrees)

plt.show()



out_degrees = [d for n, d in fb_net.out_degree()]

plt.title("out degree")

plt.xlabel("out degree")

plt.ylabel("count")

plt.hist(out_degrees)

plt.show()
in_d = np.array(in_degrees)

out_d = np.array(out_degrees)



# nodes with min or max in/out degrees

print(ids[np.argmin(in_d)], np.min(in_d))

print(ids[np.argmax(in_d)], np.max(in_d))

print(ids[np.argmin(out_d)], np.min(out_d))

print(ids[np.argmax(out_d)], np.max(out_d))
# show the clustering of the nodes with min/max degree values

clusters = nx.clustering(fb_net)

print(clusters[0])

print(clusters[1888])

print(clusters[11])

print(clusters[107])
avg_cluster_coef = nx.average_clustering(fb_net)

print(avg_cluster_coef)
density = nx.density(fb_net)

print(density)
pagerank = nx.pagerank(fb_net)



import operator

sorted_pagerank = sorted(pagerank.items(), key=operator.itemgetter(1),reverse=True)

print(sorted_pagerank[:10])
# since some methods in nx have not implemented for directed type, so here use non-directed graph to visualize the network

fb_net = nx.read_edgelist('/kaggle/input/facebook_combined.txt', create_using = nx.Graph(), nodetype = int)
pos = nx.spring_layout(fb_net)
plt.figure(figsize=(10, 10))

plt.axis('off')

plt.margins(tight=True)

nodes = nx.draw_networkx_nodes(fb_net, pos, node_size=30, node_color='blue')

nodes.set_edgecolor('black')

nodes.set_linewidth(1.0)

edges = nx.draw_networkx_edges(fb_net, pos, edge_color='black')

edges.set_linewidth(0.5)

plt.show()
coms = community.best_partition(fb_net)

size = float(len(set(coms.values())))

print("community count:", size)

mode = community.modularity(coms, fb_net)

print("modularity:", mode)



count = 0.

plt.figure(figsize=(10, 10))

plt.axis('off')

plt.margins(tight=True)

for com in set(coms.values()) :

    count = count + 1.

    list_nodes = [nodes for nodes in coms.keys() if coms[nodes] == com]

    values = [ (count / size) for nodes in list_nodes]

    nodes = nx.draw_networkx_nodes(fb_net, 

                                   pos,

                                   list_nodes,

                                   cmap=plt.get_cmap('rainbow'),

                                   with_labels=False,

                                   node_size = 30,

                                   node_color = values,

                                   vmin=0.0, vmax=1.0)

    nodes.set_edgecolor('black')

    nodes.set_linewidth(1.0)



edges = nx.draw_networkx_edges(fb_net, pos)

edges.set_linewidth(0.5)

plt.show()
def drawCentralityGraph(G, pos, cent, fgSize, nodeSize=30, weight=400):

    count = 0.

    plt.figure(figsize=(fgSize, fgSize))

    plt.axis('off')

    plt.margins(tight=True)

    for v in set(cent.values()) :

        count = count + 1.

        list_nodes = [nodes for nodes in cent.keys() if cent[nodes] == v]

        values = [(weight * v) for nodes in list_nodes]

        nodes = nx.draw_networkx_nodes(G, 

                                       pos,

                                       list_nodes,

                                       cmap=plt.get_cmap('rainbow'),

                                       with_labels=False,

                                       node_size = values,

                                       node_color = values,

                                       vmin=0.0, vmax=1.0 ) 

        nodes.set_edgecolor('black')

        nodes.set_linewidth(1.0)

    

    

    edges = nx.draw_networkx_edges(G, pos, alpha=0.3)

    edges.set_linewidth(0.5)

    plt.show()
def centralityPlot(cent, fgSize):

    plt.figure(figsize=(fgSize, fgSize))

    plt.margins(tight=True)

    cent = sorted(cent.items())

    values = [c for (node, c) in cent]

    nodes = [node for (node, c) in cent]

    plt.plot(nodes, values)

    plt.show()
cent_bet = nx.centrality.betweenness_centrality(fb_net)

centralityPlot(cent_bet, 8)

drawCentralityGraph(fb_net, pos, cent_bet, 10)
cent_de = nx.centrality.degree_centrality(fb_net)

centralityPlot(cent_de, 8)

drawCentralityGraph(fb_net, pos, cent_de, 10, weight=500)
cent_clo = nx.centrality.closeness_centrality(fb_net)

centralityPlot(cent_clo, 8)

drawCentralityGraph(fb_net, pos, cent_clo, 10, weight=200)
cent_eigen = nx.centrality.eigenvector_centrality(fb_net)

centralityPlot(cent_eigen, 8)

drawCentralityGraph(fb_net, pos, cent_eigen, 10, weight=1000)
lap_spec = nx.laplacian_spectrum(fb_net)

plt.plot(lap_spec)

plt.title('Eigenvalues of the Laplacian')

plt.show()



adj_spec = nx.adjacency_spectrum(fb_net)

plt.plot(adj_spec)

plt.title('Eigenvalues of the Adjaceny')

plt.show()



spec_ordering = nx.spectral_ordering(fb_net)

plt.plot(spec_ordering)

plt.title('Spectral Ordering')

plt.show()