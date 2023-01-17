import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap as Basemap

import pandas as pd

import numpy as np

import seaborn as sns

airport_df = pd.read_csv("../input/AirportInfo.csv")

route_df = pd.read_csv("../input/prediction.csv")
airport_df['country'] = airport_df['OAGName'].str.split(' ').apply(lambda x:x[-1])

air_df = airport_df.groupby('country').filter(lambda x: len(x)>=28)

merge_df = pd.merge(air_df,route_df,left_on='NodeName',right_on='Origin',how='left')
air_vol = merge_df.groupby('country')['PredMu'].agg([sum])

air_vol.sort_values("sum")
route_sum = pd.DataFrame(route_df.groupby(["Origin","Destination"]).size().reset_index(name = "counts"))

route_sum = pd.merge(route_df,airport_df,left_on ="Origin", right_on = "NodeName" )

route_sum = pd.merge(route_sum, airport_df, left_on = "Destination", right_on = "NodeName")



route_small = route_sum[route_sum["PredMu"]>=10000]
graph = nx.from_pandas_edgelist(route_small,'Origin','Destination','PredMu')

graph

graph.nodes(data = True)

graph.edges(data = True)

print(nx.info(graph))
graph_sum = nx.from_pandas_edgelist(route_sum,'Origin','Destination','PredMu')

graph_sum

graph_sum.nodes(data = True)

graph_sum.edges(data = True)

print(nx.info(graph_sum))
plt.figure(figsize=(10,10))

pos = nx.spring_layout(graph_sum)

nx.draw_networkx_nodes(graph_sum, pos,alpha=0.2 ,node_size=100)

nx.draw_networkx_edges(graph_sum, pos, alpha=0.2)

plt.axis('off')

plt.tight_layout()

plt.show()
routes_lats = airport_df[["NodeName","Lon","Lat"]]

xy = routes_lats.drop_duplicates("NodeName")

xy["pos"]= list(zip(xy.Lon, xy.Lat))

routes_lats = xy[["NodeName","pos"]]

routes_lats = routes_lats.set_index("NodeName")

routes_lats = routes_lats.to_dict("index")    



pos_dict = {}    

for key,value in routes_lats.items():

    for key2,value2 in value.items():

        pos_dict[key] = np.asarray(value2)
plt.figure(figsize = (14,10))

m = Basemap()



nx.draw_networkx_nodes(graph_sum,pos_dict,alpha=0.5,node_size = 80)

m.drawcountries(linewidth = 1)

m.drawcoastlines(linewidth=1)



plt.axis('off')

plt.tight_layout()

plt.title("The spatial pattern of airports in the world",size = 30)

plt.show()



degree_dict = dict(graph.degree(graph.nodes()))

nx.set_node_attributes(graph, degree_dict, 'degree')

flight_weight = nx.get_edge_attributes(graph,'PredMu')





plt.figure(figsize = (15,10))

m = Basemap(llcrnrlon=-160,llcrnrlat=-75,urcrnrlon=170,urcrnrlat=80)





nx.draw_networkx_edges(graph,pos_dict,style = "solid", alpha=0.05, width=[np.log(v)*0.1 for v in flight_weight.values()],edge_color = "r")

nx.draw_networkx_nodes(graph,pos_dict, alpha = 0.5, node_size = 50, node_shape = "o")



m.drawcoastlines(color = "white",linewidth = 0.4)

m.drawlsmask(land_color='black',ocean_color='black')



plt.tight_layout()

plt.title("Locations and routes of airports around the world",size = 30)

plt.show()
airport = list(nx.eigenvector_centrality(graph).keys())

degree  = list(nx.degree_centrality(graph).values())

clo = list(nx.closeness_centrality(graph).values())

bet = list(nx.betweenness_centrality(graph).values())

eig = list(nx.eigenvector_centrality(graph).values())


l = [degree,clo,bet,eig]

name = ['degree','closeness','betweenness','eigenvector']

plt.figure(figsize=(12,10))

sns.set(style='whitegrid')

for num,(i,n) in enumerate(zip(l,name)):

    ax = plt.subplot(2,2,num+1)

    x = np.linspace(0,len(i),len(i))

    y = sorted(i,reverse=True)

    ax.scatter(x, y,s = 8,facecolors='none', edgecolors='black')

    ax.set_title("Rank-size distribution of {} centrality".format(n))

    
all_centrality = pd.DataFrame({'airport':airport,'degree':degree,'clo':clo,'bet':bet,'eig':eig})

degree_top20 = all_centrality.sort_values("degree",ascending= False).head(20)

clo_top20 = all_centrality.sort_values("clo",ascending= False).head(20)

betw_top20 = all_centrality.sort_values("bet",ascending= False).head(20)

eig_top20 = all_centrality.sort_values("eig",ascending= False).head(20)
top_sum = pd.DataFrame({'degree centrality':degree_top20['airport'].head(5).values,'closeness centrality':clo_top20['airport'].head(5).values,

                    'betweenness centrality':betw_top20['airport'].head(5).values, 'eigenvector centrality':eig_top20['airport'].head(5).values })

airport_dict={}

for air_name,name in airport_df[['NodeName','OAGName']].values:

    airport_dict[air_name] = name
top_sum.apply(lambda x:x.map(airport_dict))
top_airports = set(np.concatenate([degree_top20.airport.values,clo_top20.airport.values,

                                   betw_top20.airport.values,eig_top20.airport.values]))

top_airports
labels = {}    

for node in graph.nodes():

    if node in top_airports:

        labels[node] = node

        

        

plt.figure(figsize = (12,12))

m = Basemap(llcrnrlon=-160,llcrnrlat=-75,urcrnrlon=170,urcrnrlat=80)





#nx.draw_networkx_labels(graph,pos_dict,labels,font_size=10,font_color='black')

nx.draw_networkx_nodes(graph,pos_dict, alpha = 0.3, node_size = 50, node_shape = "o",node_color = "orange")

nx.draw_networkx_nodes(graph,pos_dict,nodelist = labels.values(), alpha = 0.7, node_color = "orange", node_shape = "d", node_size=[v*10  for v in degree_dict.values()])



#m.drawmapboundary()

#m.drawcoastlines()

#m.shadedrelief()

m.bluemarble(scale=0.5,alpha = 0.7)



plt.tight_layout()

plt.title("Locations of airports around the world", size = 25)

plt.show()
graph.add_node("SYW")



e = [('SYW',i) for i in top_airports]

graph.add_edges_from(e)



pos_dict["SYW"] = (150.45, -33.53) 

top_airports.add("SYW")


plt.figure(figsize = (16,12))

m = Basemap(llcrnrlon=-160,llcrnrlat=-75,urcrnrlon=170,urcrnrlat=80)



nx.draw_networkx_edges(graph,pos_dict,edgelist = e, style = "solid", alpha=0.4, width=[np.log(v)*0.03 for v in flight_weight.values()])

nx.draw_networkx_nodes(graph,pos_dict, alpha = 0.5, nodelist = top_airports,  node_shape = "o",node_size=[v*15  for v in degree_dict.values()])

nx.draw_networkx_labels(graph,pos_dict,labels, font_size =10 , font_color='black')



m.drawmapboundary()

m.drawcoastlines(linewidth = 0.8)







plt.tight_layout()

plt.title("Sydney Western Airport and its connection to other important airports", size = 30)

plt.show()
degree_syd = nx.degree_centrality(graph)["SYD"]

clo_syd = nx.closeness_centrality(graph)["SYD"]

bet_syd = nx.betweenness_centrality(graph)["SYD"]

eig_syd = nx.eigenvector_centrality(graph)["SYD"]

syd_list = [degree_syd,clo_syd,bet_syd,eig_syd]

syd_list
l = [degree,clo,bet,eig]

l_syd = [degree_syd,clo_syd,bet_syd,eig_syd]

name = ['degree','closeness','betweenness','eigenvector']

plt.figure(figsize=(16,12))

plt.suptitle("Four Types of Centrality",size = 20, y = 0.93)

sns.set()

for num,(i,i_syd,n) in enumerate(zip(l,l_syd,name)):

    ax = plt.subplot(2,2,num+1)

    sns.violinplot( y = i)

    sns.lineplot(ax = ax, x = [-0.3,0.3],y=[i_syd,i_syd],c='r',markers='__')

    ax.annotate('SYW {} centrality: {:0.4f}'.format(n,i_syd), xy=(-0.15, i_syd), xytext=(-0.1, i_syd+0.03),

            arrowprops=dict(facecolor='red', shrink=0.05))