import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import math

import networkx as nx

#from nxviz import CircosPlot #nxviz unavailable on kaggle

import random



from collections import defaultdict

from scipy.stats import ttest_ind



print(os.listdir("../input"))
df = pd.read_csv("../input/CityAunrecMW.csv", low_memory=False)
print(len(df))

list(df)
df.sender_gender.value_counts()
df.receiver_gender.value_counts()
df = df[df.receiver_gender == 1]
# redo this with column as datetime object

df['timestamp'] = pd.to_datetime(df.timestamp, infer_datetime_format=True)



end_date = df.timestamp.max()

start_date = df.timestamp.min()

print("Messaging data was collected from {0} until {1}.".format(start_date.date(), end_date.date()))
df.nway.value_counts(1)
senders = set(df.senderid)

receivers = set(df.receiverid)

num_users = len(senders.union(receivers))

print("We have message data for {} distinct users.".format(str(num_users)))

print("{} users sent messages on the platform.".format(str(len(senders))))

print("{} users received messages on the platform.".format(str(len(receivers))))
df['age_difference'] = df.sender_age - df.receiver_age

plt.hist(df.age_difference)

plt.title("Histogram of Difference Between Sender Age and Receiver Age")

df.age_difference.describe()
df['attractive_difference'] = df.sender_attractive - df.receiver_attractive

plt.hist(df.attractive_difference)

plt.title("Histogram of Difference in Attractiveness Scores of Sender and Receiver")

df.attractive_difference.describe()
plt.scatter(df.sender_age, df.receiver_age, alpha = 0.01)

plt.xlabel("Sender Age")

plt.ylabel("Receiver Age")

plt.title("Receiver Age (y) vs. Sender Age (x)")

plt.show()
plt.scatter(df.sender_attractive, df.receiver_attractive, alpha = 0.01)

plt.xlabel("Sender Attractiveness Score")

plt.ylabel("Receiver Attractiveness Score")

plt.title("Receiver Attractiveness Score (y) vs. Sender Attractiveness Score (x)")

plt.show()
df.sender_attractive.describe()
df.receiver_attractive.describe()
fig, ax = plt.subplots(1,2)



ax[0].hist(df.sender_attractive, alpha=0.5, label='M', color='b')

ax[1].hist(df.receiver_attractive, alpha=0.5, label='F', color='r')

plt.show()
t = ttest_ind(df.sender_attractive, df.receiver_attractive, equal_var=False)

print(t)
df['sender_attractive_normalized'] = (df.sender_attractive - df.sender_attractive.mean()) / df.sender_attractive.std()

df['receiver_attractive_normalized'] = (df.receiver_attractive - df.receiver_attractive.mean()) / df.receiver_attractive.std() 



plt.scatter(df.sender_attractive_normalized, df.receiver_attractive_normalized, alpha = 0.01)

plt.show()
df['normalized_attractive_difference'] = df.sender_attractive_normalized - df.receiver_attractive_normalized

plt.hist(df.normalized_attractive_difference)

df.normalized_attractive_difference.describe()
G = nx.Graph()

G.add_nodes_from(df.senderid, bipartite='sender')

G.add_nodes_from(df.receiverid, bipartite='receiver')

G.add_edges_from(zip(df.senderid, df.receiverid))

nx.is_bipartite(G) # verifies that created graph is bipartite
def get_nodes_from_partition(G, partition):

    """

    Returns a list of all nodes from G belong to the specified partition.

    

    Args:

        G - a networkx Graph object, assumed to be a bipartite graph

            NB: we assume the metadata 'bipartite' is correctly assigned

            to nodes in this graph

        partition - a value corresponding to one of the two partitions of

            nodes in the bipartite graph.

            

    Output:

        a list of nodes belonging to the specified partition

    """

    return [n for n in G.nodes() if G.node[n]['bipartite'] == partition]

sender_nodes = get_nodes_from_partition(G, 'sender')

receiver_nodes = get_nodes_from_partition(G, 'receiver')



print("Number of sender nodes: " + str(len(sender_nodes)))

print("Number of receiver nodes: " + str(len(receiver_nodes)))



deg_centrality = nx.bipartite.degree_centrality(G, sender_nodes)



deg_centrality_series = pd.Series(list(deg_centrality.values()))



print(deg_centrality_series.describe())



plt.yscale('log')

plt.hist(deg_centrality.values(), bins=20)

plt.title("Histogram of Degree Centrality (log scale)")

plt.show()
receiver_dcs = [deg_centrality[n] for n in receiver_nodes]

sender_dcs = [deg_centrality[n] for n in sender_nodes]



print("Summary Statistics for Sender Degree Centralities:")

print(pd.Series(sender_dcs).describe())

print()

print("Summary Statistics for Receiver Degree Centralities")

print(pd.Series(receiver_dcs).describe())



fig, ax = plt.subplots(1,2)



plt.yscale('log')

ax[0].hist(sender_dcs, label='Sender', color='b')

ax[1].hist(receiver_dcs, label='Receiver', color='r')

plt.show()
temp = df.groupby(by=['senderid']).sender_attractive.mean()

sender_attractive_series = pd.Series([temp[n] for n in sender_nodes])



temp = df.groupby(by=['receiverid']).receiver_attractive.mean()

receiver_attractive_series = pd.Series([temp[n] for n in receiver_nodes])



print("Summary Statistics for Sender Attractiveness Score: ")

print(sender_attractive_series.describe())

print()

print("Summary Statistics for Receiver Attractiveness Score:")

print(receiver_attractive_series.describe())



fig, ax = plt.subplots(1,2)



ax[0].hist(sender_attractive_series, alpha=0.5, label='M', color='b')

ax[1].hist(receiver_attractive_series, alpha=0.5, label='F', color='r')

plt.show()
plt.title("Sender Attractiveness Score vs. Sender Degree Centrality")

plt.scatter([math.log(s) for s in sender_dcs], sender_attractive_series, alpha = 0.01)

plt.show()
plt.title("Receiver Attractiveness Score vs. Receiver Degree Centrality")

plt.scatter([math.log(s) for s in receiver_dcs], receiver_attractive_series, alpha = 0.01)

plt.show()
def shared_partition_nodes(G, node1, node2):

    """

    Returns the nodes which are neighbors of both node1 and node2

    

    Args:

        G - a networkx graph object

        node1 - a networkx node belonging to G

        node2 - a networkx node belonging to G

        

    Output:

        a set of nodes belonging to the other partition of the bipartite

        graph which are neighbors of both node1 and node2.

    """

    assert G.node[node1]['bipartite'] == G.node[node2]['bipartite']



    nbrs1 = G.neighbors(node1)

    nbrs2 = G.neighbors(node2)



    return set(nbrs1).intersection(nbrs2)



def node_similarity(G, sender1, sender2, receiver_nodes):

    """

    Returns a measure of the similarity between the nodes sender1 and sender2.

    

    Args:

        G - a networkx Graph object representing a bipartite graph

        sender1 - a node in G belonging to the same partition as sender2

        sender2 - a node in G belonging to the same partition as sender1

        receiver_nodes - a list of nodes in the other partition of the bipartite graph G

        

    Output:

        a number between 0 and 1 representing the fraction of the total possible neighbors

        that the nodes sender1 and sender 2 share.

    """

    assert G.node[sender1]['bipartite'] == 'sender'

    assert G.node[sender2]['bipartite'] == 'sender'



    shared_nodes = shared_partition_nodes(G, sender1, sender2)



    return len(shared_nodes) / len(receiver_nodes)



def most_similar_users(G, user, user_nodes, receiver_nodes):

    """

    Returns a list of users with the highest similarity score to user.

    

    Args:

        G - a networkx Graph object representing a bipartite graph

        user - the node for which we want to find the most similar users

        user_nodes - a list containing the other nodes in the same partition as the user node

        receiver_nodes - a list of nodes in the other partition

        

    Output:

        a list of users with the highest similarity score to user

    """

    assert G.node[user]['bipartite'] == 'sender'



    user_nodes = set(user_nodes)

    user_nodes.remove(user)



    similarities = defaultdict(list)

    for n in user_nodes:

        similarity = node_similarity(G, user, n, receiver_nodes)

        similarities[similarity].append(n)



    max_similarity = max(similarities.keys())



    return similarities[max_similarity]



def suggest_receiver(G, from_user, to_user):

    """

    Returns the set of all neighbors of from_user which are not neighbors of to_user.

    

    Args:

        G - a networkx Graph object representing a bipartite graph

        from_user - the user we are recommending from

        to_user - the user we are recommending to

        

    Outputs:

        a set of nodes representing the neighbors of from_user which are not neighbors

        of to_user. If this set is empty, returns the set of all non-neighbors of the

        to_user. If to_user is neighbors with all possible nodes, it returns the set of

        all nodes.

    """

    from_receivers =  set(G.neighbors(from_user))

    to_receivers = set(G.neighbors(to_user))

    suggestions = from_receivers.difference(to_receivers)

    backup = set(receiver_nodes).difference(to_receivers)

    if suggestions:

        return suggestions

    elif backup: # if intersected set is empty, return receivers the user hasn't messaged

        return backup

    else: #only if user has messaged every possible receiver

        return set(receiver_nodes)



def recommend(G, user, sender_nodes, receiver_nodes):

    """

    Recommends a profile for user to try messaging.

    """

    most_similar = most_similar_users(G, user, sender_nodes, receiver_nodes)

    if most_similar:

        node2 = random.choice(most_similar)

    else:

        node2 = rando.choice(sender_nodes)

    return random.choice(list(suggest_receiver(G, user, node2)))
user = random.choice(sender_nodes)

print(user)

most_sim = most_similar_users(G, user, sender_nodes, receiver_nodes)

print(most_sim)

suggest_receiver(G, user, random.choice(most_sim))



print(recommend(G, user, sender_nodes, receiver_nodes))