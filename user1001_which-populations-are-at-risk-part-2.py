%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd 
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import re
from sklearn.preprocessing import OneHotEncoder
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
import community
import networkx as nx

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# functions: 

# check duplicated cols:
def GetDuplicatedColumns(df):
    '''
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    '''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
 
    return list(duplicateColumnNames)

# select only numerical cols:
def get_num_cols(df):
    myCols = ['Country']
    for item_ in df.columns.tolist():
        if df[item_].dtype == 'float64':
            myCols.append(item_) 
    return myCols
# read files:
myDir = '../input/' 
df = pd.DataFrame()
for dirname, _, filenames in os.walk(myDir):
    for filename in filenames:
        myFile = os.path.join(dirname, filename)
        tmp_ = pd.read_csv(myFile)
        df = df.append(tmp_)
        #print(myFile, tmp_.shape, df.shape)
        
df.shape
# count nans:
#print ('1:\n',df.isna().sum())

# replacements for Nan values:
obj_replacement = 'Not Existing'
float_replacement = -1.
int_replacement = -10
for item_ in df.columns.tolist():
    type_ = df[item_].dtype
    #print (item_, type_)
    if type_ == 'object':
        df[item_].fillna(value=obj_replacement, inplace=True)
    if type_ == 'float64':
        df[item_].fillna(value=float_replacement, inplace=True)
    if type_ == 'int64':
        df[item_].fillna(value=int_replacement, inplace=True)
        
## count nans:
#print ('Check NaN values after replacement:\n',df.isna().sum())
# select cols to delete:
cols_to_remove = []
for item_ in df.columns.tolist():
    #print ('col = ',item_)
    #print (df[item_].unique())
    if len(df[item_].unique()) < 3:
        cols_to_remove.append(item_)
    #print ('==========')
    
print ('Nr of columns to remove: ',len(cols_to_remove))
df_ = df.copy()
df_.drop(cols_to_remove, axis=1, inplace=True)
print ('Shape of resulting Data Frame: ',df_.shape)
cols_to_remove_ = GetDuplicatedColumns(df_)
# drop duplicated cols:
print ('Nr of columns to be removed: ',len(cols_to_remove_))
df__ = df_.drop(columns=cols_to_remove_).copy()
print ('Shape of the Data Frame after removal of the duplicated coilumns: ',df__.shape)
# take into account only numerical cols:
cols = df__.columns.tolist()
cols.insert(0, cols.pop(cols.index('Country')))
df__ = df__.reindex(columns= cols)
myCols = get_num_cols(df__)
df_num = df__[myCols].copy()
print ('The shape of resulting Data Frame (with numercial columns)', df_num.shape)
myCountries = df_num.Country.values
print ('Available Countries: ',myCountries)
additional_to_delete = ['hq_zip_cod','lat','latitude_d','long','longitude']
df_num.drop(additional_to_delete, axis=1, inplace=True)
print ('Shape of the current Data Frame: ',df_num.shape)
myCols_num = df_num.columns.tolist()
dic_part2 = {'AUC__cases':4068, 'AUC__deaths':4069, 'AUC__ratio_death_to_cases':4070, 
 'active':4071, 'active_cases':4072, 'bed_utiliz':4073, 'cnty_fips':4074, 'data_value':4075, 'elevation':4076,
 'fips':4077, 'high_confidence_limit':4078, 'id':4079, 
 'inform_epidemic_lack_of_coping_capacity':4080, 'inform_epidemic_vulnerability':4081,
 'inform_lack_of_coping_capacity':4082, 'inform_p2p_hazard_and_exposure_dimension':4083, 
 'inform_risk':4084, 'inform_vulnerability':4085, 'locationid':4086, 'low_confidence_limit':4087, 
 'new_deaths':4088, 'num_licens':4089, 'num_staffe':4090, 'objectid':4091, 
 'people_using_at_least_basic_sanitation_services':4092,'percent_yoy_change':4093,'popdata2018':4094, 
 'population_density':4095,'population_living_in_urban_areas':4096,'potential':4097,
 'prevalence_of_undernourishment':4098, 'recovered':4099,'sample_size':4100, 
 'serious_critical_cases':4101, 'state_fips':4102, 'total_cases':4103, 'total_cases_per_1m_pop':4104, 
 'total_confirmed_cases_of_covid_19_cases':4105, 
 'total_confirmed_cases_of_covid_19_per_million_people_cases_per_million':4106, 
 'total_covid_19_tests':4107, 'total_covid_19_tests_per_million_people':4108, 
 'total_deaths':4109, 'total_deaths_per_1m_pop':4110, 'total_recovered':4111}
print (list(dic_part2.keys()))
# save 
file_ = './df_num__data_analyzer_v4_global.csv'
df_num.to_csv(file_,index=False)
# Define product records
Selected_countries = myCountries 
products = df_num.iloc[0:,0:].values
print ('Shape of products: ',products[:,:].shape)
keys = list(dic_part2.keys())

def GetDictionary(keys, attrs):
    d = dict((key, value) for (key, value) in zip(keys, attrs))
    return d

#
def get_cos_dist(a,b):
    # cos dist:
    offset = 1. # 1. is added in order to avoid negative values !!
    myDist = offset + ((1. - spatial.distance.cosine(a, b)))
    return (myDist)

# Returns a Product Similarity Matrix for specified products &
# similarity score, sim_score.
# Part 1 - related to the statistical description of the countries
def product_similarity_matrix_part1(products, sim_score=get_cos_dist):
    prod_sim_mat = {}
    # n(n-1)/2 scores
    for i, product_i in enumerate(products[:]):
        for product_j in products[:i]:
            idi, attr_i =  (product_i[0]), [float(i) for i in product_i[1:4068]] 
            idj, attr_j =  (product_j[0]), [float(i) for i in product_j[1:4068]] 
            prod_sim_mat[(idi, idj)] = sim_score(attr_i, attr_j)
            
    return prod_sim_mat

# Returns a Product Similarity Matrix for specified products 
# similarity score, sim_score.
# Part 2 - related to the virus measures per countries
def product_similarity_matrix_part2(products, sim_score=get_cos_dist): 
    prod_sim_mat = {}
    # n(n-1)/2 scores
    for i, product_i in enumerate(products[:]):
        for product_j in products[:i]:
            idi, attr_i =  (product_i[0]), [float(i) for i in product_i[4068:]] 
            idj, attr_j =  (product_j[0]), [float(i) for i in product_j[4068:]] 
            prod_sim_mat[(idi, idj)] = sim_score(attr_i, attr_j)
            
    return prod_sim_mat

# Combine Product Similarity Matrix and products to construct a 
# Product Similarity Graph.
def product_similarity_graph(prod_sim_mat_1, prod_sim_mat_2, products):
    # Create networkx graph
    PSG = nx.Graph() #nx.DiGraph()

    # Add nodes and attrs to graph
    for product in products:
        id_, attrs1, attrs2 = (product[0]), product[1:4068], product[4068:]
        attrs_dict = GetDictionary(keys, attrs2)
        PSG.add_node(id_, attrs1=attrs1, attrs2=attrs2, attr3=attrs_dict) 
        
    # Add edges and scores to nodes
    for ind, score1 in prod_sim_mat_1.items():
        if score1 > 0:
            start, end = ind
            PSG.add_edge(start, end, score1=score1)
            PSG.add_edge(end, start, score1=score1)
            
    # Add edges and scores to nodes
    for ind, score2 in prod_sim_mat_2.items():
        if score2 > 0:
            start, end = ind
            PSG.add_edge(start, end, score2=score2)
            PSG.add_edge(end, start, score2=score2)
            
    return PSG

# Generate a set of attribute differences between each pair of connected 
# nodes in prod_sim_graph and add to edges.
def generate_diff_attrs(prod_sim_graph):
    for anchor, neighbour in prod_sim_graph.edges():
        
        anchor_attrs1, anchor_attrs2 = prod_sim_graph.nodes[anchor]['attrs1'],prod_sim_graph.nodes[anchor]['attrs2']
        neighbour_attrs1, neighbour_attrs2 = prod_sim_graph.nodes[neighbour]['attrs1'],prod_sim_graph.nodes[neighbour]['attrs2']
        
        # cos dist:
        offset = 1.
        cos_dist_1 = offset + (1. - spatial.distance.cosine([float(i) for i in anchor_attrs1], 
                                                   [float(i) for i in neighbour_attrs1]))
        cos_dist_2 = offset + (1. - spatial.distance.cosine([float(i) for i in anchor_attrs2], 
                                                   [float(i) for i in neighbour_attrs2]))
        #cos_dist_euclidean = np.sqrt(cos_dist_1**(2.) + cos_dist_2**(2.))
        delta = cos_dist_2 - cos_dist_1
        eta = np.sqrt(1. + (delta/cos_dist_1) + (delta/cos_dist_1)**(2.))
        cos_dist_euclidean = np.sqrt(2.)*cos_dist_1*eta
        # we will calculate provide cos_dist_euclidean & eta 
        dist_ratio = (offset + cos_dist_1)/(offset + cos_dist_2)
        dist_anchor_1 = np.linalg.norm([float(i) for i in anchor_attrs1])
        dist_anchor_2 = np.linalg.norm([float(i) for i in anchor_attrs2])
        dist_neighbour_1 = np.linalg.norm(np.array([float(i) for i in neighbour_attrs1]))
        dist_neighbour_2 = np.linalg.norm(np.array([float(i) for i in neighbour_attrs2]))
        dist_env_1 = dist_anchor_1/dist_neighbour_1 # ratio
        dist_action_2 = dist_anchor_2/dist_neighbour_2 # ratio
        
        prod_sim_graph.edges[anchor,neighbour]['diff_tags_1'] = cos_dist_1
        prod_sim_graph.edges[anchor,neighbour]['diff_tags_2'] = cos_dist_2
        prod_sim_graph.edges[anchor,neighbour]['diff_dist_ratio'] = dist_ratio
        prod_sim_graph.edges[anchor,neighbour]['diff_ratio_env_action_part1'] = dist_env_1
        prod_sim_graph.edges[anchor,neighbour]['diff_ratio_env_action_part2'] = dist_action_2
        prod_sim_graph.edges[anchor,neighbour]['cos_dist_eucl'] = cos_dist_euclidean
        prod_sim_graph.edges[anchor,neighbour]['cos_dist_eta'] = eta
            
    return prod_sim_graph

# Generate a navigation tag map for each node of the prod_sim_graph.
def generate_nav_tags(prod_sim_graph):
    
    for anchor in prod_sim_graph.nodes():
        
        tag_map_1 = {}
        tag_map_2 = {}
        tag_map_3 = {}
        tag_map_4 = {}
        tag_map_5 = {}
        tag_map_6 = {}
        tag_map_7 = {}
        for neighbour in prod_sim_graph.neighbors(anchor):
        
            tag1 = prod_sim_graph[anchor][neighbour]['diff_tags_1']
            tag2 = prod_sim_graph[anchor][neighbour]['diff_tags_2']
            tag3 = prod_sim_graph[anchor][neighbour]['diff_dist_ratio']
            tag4 = prod_sim_graph[anchor][neighbour]['diff_ratio_env_action_part1']
            tag5 = prod_sim_graph[anchor][neighbour]['diff_ratio_env_action_part2']
            tag6 = prod_sim_graph[anchor][neighbour]['cos_dist_eucl']
            tag7 = prod_sim_graph[anchor][neighbour]['cos_dist_eta']
            
            tag_map_1[tag1] = tag_map_1.get(tag1, []) + [neighbour]
            tag_map_2[tag2] = tag_map_2.get(tag2, []) + [neighbour]
            tag_map_3[tag3] = tag_map_3.get(tag3, []) + [neighbour]
            tag_map_4[tag4] = tag_map_4.get(tag4, []) + [neighbour]
            tag_map_5[tag5] = tag_map_5.get(tag5, []) + [neighbour]
            tag_map_6[tag6] = tag_map_6.get(tag6, []) + [neighbour]
            tag_map_7[tag7] = tag_map_7.get(tag7, []) + [neighbour]
                
        prod_sim_graph.nodes[anchor]['nav_tags_1'] = tag_map_1
        prod_sim_graph.nodes[anchor]['nav_tags_2'] = tag_map_2
        prod_sim_graph.nodes[anchor]['nav_dist_ratio'] = tag_map_3
        prod_sim_graph.nodes[anchor]['nav_ratio_env_action_part1'] = tag_map_4
        prod_sim_graph.nodes[anchor]['nav_ratio_env_action_part2'] = tag_map_5
        prod_sim_graph.nodes[anchor]['nav_cos_dist_eucl'] = tag_map_6
        prod_sim_graph.nodes[anchor]['nav_cos_dist_eta'] = tag_map_7
        
    return prod_sim_graph           

if True:
    
    # Construct Product Similarity Matrix
    prod_sim_mat_1 = product_similarity_matrix_part1(products)
    prod_sim_mat_2 = product_similarity_matrix_part2(products)
    
    # Construct Product Similarity Graph
    PSG = product_similarity_graph(prod_sim_mat_1, prod_sim_mat_2, products)
    
    # Generate difference attributes and attach to edges
    PSG = generate_diff_attrs(PSG)
    
    # Generate navigation tags and attach to nodes
    PSG = generate_nav_tags(PSG)
    
# parameters:
# Draw the resulting graph
param_ = 'cos_dist_eta'
# all possibilities for variable 'param_: 'diff_tags_1', 'diff_tags_2','_ratio_env_action_part1','diff_ratio_env_action_part2',
# 'diff_dist_ratio','cos_dist_eucl','cos_dist_eta'
if True:
    # Setup simulated user inputs
    anchor = myCountries[1] # random choice
    tag_selections = list(dic_part2.keys()) 
    product_selections = myCountries[2] 
    
    # Run through simulated inputs    
    for selection in tag_selections:
        try:
            products = PSG.node[anchor]['attr3'][selection]
            if len(products)>1:
                product_selection = product_selections.pop(0)
                assert product_selection in products, "Bad selection"
                anchor = product_selection
            else:
                anchor, = products
        except:
            pass
        

    edge_labels=dict([((u,v,),'{:0.4f}'.format(d[param_]))
                 for u,v,d in PSG.edges(data=True)])

    edge_colors = []
    edge_widths = []
    for edge in edge_labels:
        val_ = np.float(edge_labels.get(edge))
        
        if val_ < 1.:
            edge_colors.append('green')
            edge_widths.append(10)
        if val_ == 1.:
            edge_colors.append('black')
            edge_widths.append(5)
        if (val_ > 1.) & (val_ <= np.sqrt(3.)):
            edge_colors.append('blue')
            edge_widths.append(2)
        if (val_ > np.sqrt(3.)):
            edge_colors.append('red')
            edge_widths.append(5)

    pos=nx.spring_layout(PSG, k=0.02, weight=param_, seed=2)
    #pos=nx.spectral_layout(PSG, weight=param_, scale=0.1)
    
    val_map = { myCountries[i] : 123 for i in range(len(myCountries)) } # the same color of node
    values = [val_map.get(node, 0.) for node in PSG.nodes()]
    
    plt.figure(figsize=(20,20))
    #nx.draw_networkx_edge_labels(PSG,pos,alpha=0.8, edge_labels=edge_labels)
    nx.draw(PSG,pos,with_labels=True, node_color = values, node_size=3500, font_size=28, 
            edge_color=edge_colors, width=edge_widths, edge_cmap=plt.cm.Reds, alpha=0.6)
    plt.show()

    print("\n\tDone !.\n")
# it provides a long list, it disturbs the evaluation of the code
#for edge in edge_labels:
#    print (edge)
#    print (np.float(edge_labels.get(edge)))
# spectral clustering:
from sklearn.cluster import SpectralClustering
from sklearn import metrics
np.random.seed(1)

# Get adjacency-matrix as numpy-array
weight_ = param_ 
adj_mat = nx.adjacency_matrix(PSG, nodelist=list(PSG.nodes()), weight=weight_).todense()
node_list = list(PSG.nodes())

IndexArray = []
NrOfElem = []
myXList = range(2,15,1)
affinity_ = 'precomputed'
n = 5
for NrOfClusters in myXList:
    #print (NrOfClusters)
    sc = SpectralClustering(NrOfClusters, affinity= affinity_, n_init=100, assign_labels='discretize',random_state=0)
    clusters = sc.fit_predict(adj_mat)
    labels = sc.labels_    
    # plot each nth result:
    if NrOfClusters % n == 0:
        plt.scatter(node_list,clusters,c=clusters, s=50, cmap='viridis')
        plt.xticks(node_list,node_list, rotation='vertical')
        plt.show()
    # Compare ground-truth and clustering-results
    print('spectral clustering')
    print ('nr of clusters: ', np.unique(labels))
    
    indexValue = metrics.silhouette_score(adj_mat, labels)
    IndexArray.append(indexValue) 
    NrOfElem.append(len(labels[labels == np.unique(labels)[-1]]))
    print ('nr of elements in the last class:',len(labels[labels == np.unique(labels)[-1]]),indexValue)
    
plt.figure(figsize=(12,12))
plt.title('silhouette_score')
plt.plot(myXList,IndexArray,'o')
plt.xlabel('nr of clusters')
plt.ylabel('silhouette_score')
plt.show()
# single run for the best cluster
# Spectral Clustering
clusters_ = np.argmax(IndexArray) + 2
clusters = SpectralClustering(affinity = affinity_, assign_labels="discretize",
                              random_state=0,n_clusters=clusters_).fit_predict(adj_mat)
plt.figure(figsize=(10,10))
plt.scatter(node_list,clusters,c=clusters, s=50, cmap='viridis')
plt.xticks(node_list,node_list, rotation='vertical')
plt.xlabel('Country')
plt.ylabel('Cluster')
plt.show()
# AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering

# Get adjacency-matrix as numpy-array
weight_ = param_ 
adj_mat = nx.adjacency_matrix(PSG, nodelist=list(PSG.nodes()), weight=weight_).todense()
node_list = list(PSG.nodes())

IndexArray = []
NrOfElem = []
myXList = range(2,15,1)
affinity_ = 'precomputed' 
n = 5
for NrOfClusters in myXList:
    print (NrOfClusters)
    sc = AgglomerativeClustering(n_clusters=NrOfClusters,linkage='average',affinity=affinity_)
    clusters = sc.fit_predict(adj_mat)
    labels = sc.labels_    
    # plot each nth result:
    if NrOfClusters % n == 0:
        plt.scatter(node_list,clusters,c=clusters, s=50, cmap='viridis')
        plt.xticks(node_list,node_list, rotation='vertical')
        plt.show()
    # Compare ground-truth and clustering-results
    print('spectral clustering')
    print ('nr of clusters: ', np.unique(labels))
    indexValue = metrics.silhouette_score(adj_mat, labels)
    IndexArray.append(indexValue) 
    NrOfElem.append(len(labels[labels == np.unique(labels)[-1]]))
    print ('nr of elements in the last class:',len(labels[labels == np.unique(labels)[-1]]),indexValue)
    
plt.figure(figsize=(10,10))
plt.title('silhouette_score')
plt.plot(myXList,IndexArray,'o')
plt.xlabel('nr of clusters')
plt.ylabel('silhouette_score')
plt.show()
# single run
# agglomerative Clustering 
clusters_ = np.argmax(IndexArray) + 2
affinity_ = 'precomputed'
clusters = SpectralClustering(affinity = affinity_, assign_labels="discretize",
                              random_state=0,n_clusters=clusters_).fit_predict(adj_mat)
plt.figure(figsize=(10,10))
plt.scatter(node_list,clusters,c=clusters, s=50, cmap='viridis')
plt.xticks(node_list,node_list, rotation='vertical')
plt.xlabel('Country')
plt.ylabel('Cluster')
plt.show()
# PageRank Algorithm
# PageRank estimates a current node’s importance from its linked neighbors and 
# then again from their respective neighbors.

weight_ = param_ 
rank_list = nx.pagerank(PSG, weight=weight_, alpha=0.9)
lists = sorted(rank_list.items()) 
x, y = zip(*lists) 

plt.plot(x, y,'o--')
plt.xticks(x,x, rotation='vertical')
plt.ylabel('rank_value')
plt.xlabel('country')
plt.show()
# Kernighan-Lin Paritioning
weight_ = param_ 
parts = nx.community.kernighan_lin_bisection(PSG, weight=weight_, max_iter=100, seed=1234)
print ('parts=',parts)
node_colors_map = {}
for i, lg in enumerate(parts):
    for node in lg:
        node_colors_map[node] = i
node_colors = [node_colors_map[n] for n in PSG.nodes]

pos_=nx.spring_layout(PSG, weight=param_)
node_list = list(PSG.nodes())
plt.figure(figsize=(12, 12))  
plt.axis('off')
nx.draw_networkx_nodes(PSG, pos=pos_, with_labels=False, node_size=600, node_color=node_colors)
nx.draw_networkx_labels(PSG,pos_,font_size=16,font_color='r',alpha=0.9)
nx.draw_networkx_edges(PSG, pos_, alpha=0.1)
plt.show(PSG)


