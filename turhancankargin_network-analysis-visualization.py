import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Introduction to Network Analysis

    #Basics of NetworkX

import networkx as nx

G = nx.Graph()

G.add_node(1)

G.add_nodes_from([2,3])

G.add_nodes_from(["u","v"])

print(G.nodes())

G.add_edge(1,2)

G.add_edge("u","v")

G.add_edges_from([(1,3),(1,4),(1,5),(1,6)])

print(G.edges())

G.remove_node(2)

print(G.nodes())

G.remove_nodes_from([4,5])

print(G.nodes())
#Graph Visualization

G = nx.karate_club_graph()

import matplotlib.pyplot as plt

nx.draw(G,with_labels=True,node_color="lightblue",edge_colr="gray")

print(G.degree())

print("---->",G.degree()[16])
#Random Graphs (Erdős-Rényi (ER) graph)

from scipy.stats import bernoulli

bernoulli.rvs(p=0.2)

N = 20

p = 0.2

def er_graph(N,p):

    """ Generate an ER Graph """

    # Create empty graph

    G = nx.Graph()

    G.add_nodes_from(range(N))

    # Add all N nodes in the graph

    # Loop over all pairs of nodes

    for node1 in G.nodes():

        for node2 in G.nodes():

            if node1 < node2 and bernoulli.rvs(p=p):

                G.add_edge(node1,node2)

                # Add an edge with prob p

    return G

            

nx.draw(er_graph(50,0.08),node_size=40, node_color ="gray")

nx.erdos_renyi_graph(n=10,p=1)
#Plotting the Degree Distribution

def plot_degree_distribution(G):

    degree_sequence = [d for n, d in G.degree()]

    plt.hist(degree_sequence, histtype="step")

    plt.xlabel("Degree $k$")

    plt.ylabel("$P(k)$")

    plt.title("Degree Distribution")

    

G1 = er_graph(500,0.08)

plot_degree_distribution(G1)

G2 = er_graph(500,0.08)

plot_degree_distribution(G2)

G3 = er_graph(500,0.08)

plot_degree_distribution(G3)
#Descriptive Statistics of Empirical Social Networks

# Look at the basic properties of social networks 

# in two villages in rural India

import numpy as np

A1 = np.loadtxt("/kaggle/input/village-relationships/VillageRelationships1.csv", delimiter=",")

A2 = np.loadtxt("/kaggle/input/village-relationships/VillageRelationships2.csv", delimiter=",")



G1 = nx.to_networkx_graph(A1)

G2 = nx.to_networkx_graph(A2)



def basic_net_stats(G):

    print("Number of nodes: %d" %G.number_of_nodes())

    print("Number of edges: %d" %G.number_of_edges())

    degree_sequence = [d for n, d in G.degree()]

    print("Average degree: %.2f" % np.mean(degree_sequence))



basic_net_stats(G1)

basic_net_stats(G2)

plot_degree_distribution(G1)

plot_degree_distribution(G2)
 # Findingt he Largest Connected Component

def connected_component_subgraphs(G):

    for c in nx.connected_components(G):

        yield G.subgraph(c)

gen = connected_component_subgraphs(G1)

g = gen.__next__()

type(g)

g.number_of_nodes()

len(g)

len(G1)

G1.number_of_nodes()

G1_LCC = max(connected_component_subgraphs(G1),key = len)

G2_LCC = max(connected_component_subgraphs(G2),key = len)

len(G1_LCC)

len(G2_LCC)

G1_LCC.number_of_nodes()

G1_LCC.number_of_nodes() / G1.number_of_nodes()

G2_LCC.number_of_nodes() / G2.number_of_nodes()



# Visualize the largest connected component



plt.figure()

nx.draw(G1_LCC,node_color="red",edge_color="gray",node_size=20)
nx.draw(G2_LCC,node_color="green",edge_color="gray",node_size=20)



# =============================================================================

# The visualization algorithm that we have used

# is stochastic, meaning that if you run it several times,

# you will always get a somewhat different graph layout.

# However, in most visualizations, you should

# find that the largest connected component of G2

# appears to consist of two separate groups.

# These groups are called network communities.

# And the idea is that a community is a group

# of nodes that are densely connected to other nodes in the group,

# but only sparsely connected nodes outside of that group.

# Finding network communities is a very interesting and timely problem.

# It's also one of those problems that is fairly easy to state in words,

# but a more mathematically rigorous formulation of the problem

# reveals that a problem is not so easy after all.

# 

# =============================================================================