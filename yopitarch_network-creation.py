import networkx as nx

import pandas as pd
pathFile = "../input/reddit-ut3-ut1/comments_students.csv"
df = pd.read_csv(pathFile)
df.head()
g= nx.DiGraph()
g.add_nodes_from(df.link_id, type="link")
g.add_nodes_from(df.name, type="comment")
g.add_edges_from(df[["name","parent_id"]].values, link_type="parent")
nx.write_gml(g, "graph.gml")