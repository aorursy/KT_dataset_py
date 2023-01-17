import pandas as pd
import networkx as nx
dat = pd.read_csv("../input/graph_lyon_remy.csv")
graph = nx.DiGraph()
for index,e in dat.iterrows():
    source = e["source"]
    target = e["target"]
    pos = e["WKT"]
    posList = pos[12:-1].split(",")
    pos1 = posList[0].split(" ")
    pos2 = posList[-1].split(" ")
    graph.add_node(source,long=float(pos1[0]),lat=float(pos1[1]))
    graph.add_node(target,long=float(pos2[0]),lat=float(pos2[1]))
for index,e in dat.iterrows():
    graph.add_edge(e["source"],e["target"],length=e["length"],max_speed=e["max_speed"],max_flow=e["max_flow"],width=e["width"])
#nx.write_gexf(graph,"LyonRoads/road_cleaned.gexf")
btw = nx.edge_betweenness_centrality(graph,normalized=False,k=100)
EV = nx.eigenvector_centrality(graph)
Katz = nx.katz_centrality(graph)
#com = nx.communicability(graph.to_undirected())
PR = nx.pagerank(graph)
#closeness = nx.closeness_centrality(graph,distance="length")
nx.set_node_attributes(graph,"EV",EV)
nx.set_node_attributes(graph,"Katz",Katz)
nx.set_node_attributes(graph,"PageRank",PR)
nx.set_edge_attributes(graph,"btw",btw)
nx.set_node_attributes(graph,"closeness",closeness)
nx.write_gexf(graph,"LyonRoads/road_cleaned.gexf")
