%load_ext autoreload
%autoreload 2
from importlib import reload
    
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.cm import Greys


%matplotlib inline  
import numpy as np

from sklearn.manifold import TSNE
from sklearn.metrics import pairwise
import sklearn as sk
import community
import seaborn as sns

import networkx as nx
import pandas as pd
import subprocess
import sys
import pickle
from scipy.spatial import distance
G = nx.read_gexf("../input/road-cleaned/road_cleaned.gexf")
#G = nx.karate_club_graph()
G = nx.convert_node_labels_to_integers(G,label_attribute="ID")
"""
 -i:Input graph path (default:'graph/karate.edgelist')
   -o:Output graph path (default:'emb/karate.emb')
   -d:Number of dimensions. Default is 128 (default:128)
   -l:Length of walk per source. Default is 80 (default:80)
   -r:Number of walks per source. Default is 10 (default:10)
   -k:Context size for optimization. Default is 10 (default:10)
   -e:Number of epochs in SGD. Default is 1 (default:1)
   -p:Return hyperparameter. Default is 1 (default:1)
   -q:Inout hyperparameter. Default is 1 (default:1)
   -v Verbose output. 
   -dr Graph is directed. 
   -w Graph is weighted. 
   -ow Output random walks instead of embeddings. 
"""

nx.write_edgelist(G,"temp/tempGraph.graph",data=False)
#nx.write_edgelist(G,"temp/tempGraph.graph")

args = ["algorithms/node2vec"]
args.append("-i:temp/tempGraph.graph")
args.append("-o:temp/node2vec.emb")
args.append("-d:%d" % 128) #dimension
#args.append("-l:%d" % self._walk_len) #walk length
#args.append("-r:%d" % self._num_walks) #number of walks
#args.append("-k:%d" % self._con_size) #context size
#args.append("-e:%d" % self._max_iter) #max iterations
#args.append("-p:%f" % self._ret_p) #
#args.append("-q:%f" % self._inout_p)
#args.append("-v")
args.append("-dr")
#args.append("-w")
        
string =""
for x in args:
    string+=x+" "
subprocess.check_output(string,shell=True)
G.adj[0]
embeddingFile = "../input/node2vec/node2vec128.emb"
#embeddingFile = "temp/struc2vec.emb"


def loadEmbedding(file_name):
    with open(file_name, 'r') as f:
        n, d = f.readline().strip().split()
        X = np.zeros((int(n), int(d)))
        for line in f:
            emb = line.strip().split()
            emb_fl = [float(emb_i) for emb_i in emb[1:]]
            X[int(float(emb[0])), :] = emb_fl
    return X
theEmbedding = loadEmbedding(embeddingFile)
theEmbedding[0]
#maxFlow = nx.get_edge_attributes(G,"max_flow")
length = nx.get_edge_attributes(G,"length")
width = nx.get_edge_attributes(G,"width")
maxSpeed = nx.get_edge_attributes(G,"max_speed")
features =[]
featuresANDEmb =[]
toPredict=[]
for e in maxSpeed:
    features.append([width[e],length[e]])
    featuresANDEmb.append([width[e],length[e]]+list(theEmbedding[e[0]])+list(theEmbedding[e[1]]))
    toPredict.append(maxSpeed[e])
toPredict = [1 if x==130 else 0 for x in toPredict]
print(features[:1])
print(toPredict[:1])
print(featuresANDEmb[:1])
#X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(features,toPredict,train_size=0.80,shuffle=True)
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(featuresANDEmb,toPredict,train_size=0.80,shuffle=True)
print(sum(y_train))
print(sum(y_test))
linModel = sk.linear_model.LogisticRegression()
linModel.fit(X_train,y_train)
scores = linModel.decision_function(X_test)
sk.metrics.roc_auc_score(y_test,scores)
#names= nx.get_node_attributes(G,"name")
sortedPredictions = sorted([(scores[i],y_test[i])for i in range(len(scores))],reverse=True)
sortedPredictions[:50]
#sns.scatterplot(scores,y_test)
