! pip uninstall -y networkx-neo4j #remove the old installation
! pip install git+https://github.com/ybaktir/networkx-neo4j
import datetime, time
print ('Last run on: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + repr(time.tzname))
user = 'neo4j'
password = 'span-moneys-mail' #YOUR OWN SANDBOX PASSWORD
uri = 'bolt://34.232.72.6:32988' #YOUR OWN SANDBOX URL
from neo4j import GraphDatabase
driver = GraphDatabase.driver(uri=uri,auth=(user,password))
                              #OR "bolt://localhost:7673" for Neo4j Desktop
                              #OR the cloud url
import nxneo4j as nx
G = nx.Graph(driver)
G.delete_all()  #This will delete all the data, be careful
                #Just making sure that the results are reprodusible.
#Add a node
G.add_node("Yusuf")
#Add node with features
G.add_node("Nurgul",gender='F')
#Add multiple properties at once
G.add_node("Betul",age=4,gender='F')
#Check nodes
for node in G.nodes():   #Unlike networkX, nxneo4j returns a generator
    print(node)
#Or simply
list(G.nodes())
#Get the data associated with each node
list(G.nodes(data=True))
#number of nodes
len(G)
#Display
nx.draw(G) #It is interactive, drag the nodes!
#Check a particular node feature
G.nodes['Betul']
#You can be more specific
G.nodes['Betul']['age']
G.add_nodes_from([1,2,3,4])
list(G.nodes())
#Add one edge
G.add_edge('Yusuf','Betul')
nx.draw(G) #default relationship label is "CONNECTED"
#You can change the default connection label like the following
G.relationship_type = 'LOVES'
G.add_edge('Yusuf','Nurgul')
G.add_edge('Nurgul','Yusuf')
nx.draw(G)
#You can add properties as well
G.add_edge('Betul','Nurgul',how_much='More than Dad')
#display the values
list(G.edges(data=True))
G.relationship_type = 'CONNECTED'
G.add_edges_from([(1,2),(3,4)])
nx.draw(G)
G.remove_node('Yusuf')
list(G.nodes())
G.delete_all()
G.load_got()
#You can change the default parameters like the following:
G.identifier_property = 'name'
G.relationship_type = '*'
G.node_label = 'Character'
nx.draw(G) #Zoom in to see the names :)
len(G) #796 nodes
nx.pagerank(G) #RAW OUTPUT
# the most influential characters
response = nx.pagerank(G)
sorted_pagerank = sorted(response.items(), key=lambda x: x[1], reverse=True)
for character, score in sorted_pagerank[:10]:
    print(character, score)
# Betweenness centrality
nx.betweenness_centrality(G) #RAW OUTPUT
# RANKED OUTPUT
response = nx.betweenness_centrality(G)

sorted_bw = sorted(response.items(), key=lambda x: x[1], reverse=True)
for character, score in sorted_bw[:10]:
    print(character, score)
# Closeness centrality
nx.closeness_centrality(G) #RAW OUTPUT
# RANKED
response = nx.closeness_centrality(G)

sorted_cc = sorted(response.items(), key=lambda x: x[1], reverse=True)
for character, score in sorted_cc[:10]:
    print(character, score)
# Label propagation
nx.label_propagation_communities(G) #RAW OUPUT is a generator
communities = nx.label_propagation_communities(G)
sorted_communities = sorted(communities, key=lambda x: len(x), reverse=True)
for community in sorted_communities[:10]:
    print(list(community)[:10])
# Clustering
nx.clustering(G) #RAW OUTPUT
response = nx.clustering(G)

biggest_coefficient = sorted(response.items(), key=lambda x: x[1], reverse=True)
for character in biggest_coefficient[:10]:
    print(list(character)[:10])
list(nx.connected_components(G))
nx.number_connected_components(G)
nx.triangles(G) #RAW OUTPUT
# Shortest path
nx.shortest_path(G, source="Tyrion-Lannister", target="Hodor")
# Shortest weighted path
nx.shortest_weighted_path(G, source="Tyrion-Lannister", target="Hodor",weight='weight')