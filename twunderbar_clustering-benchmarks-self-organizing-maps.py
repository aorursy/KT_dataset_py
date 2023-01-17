import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data = np.loadtxt("../input/s1.txt")

import SimpSOM as sps



#Build a network 20x20 with a weights format taken from the raw_data and activate Periodic Boundary Conditions. 

net = sps.somNet(20, 20, data, PBC=True)



#Train the network for 10000 epochs and with initial learning rate of 0.1. 

net.train(0.01, 10000)
#Print a map of the network nodes and colour them according to the first feature (column number 0) of the dataset

#and then according to the distance between each node and its neighbours.

net.nodes_graph(colnum=0)

net.diff_graph()



#Project the datapoints on the new 2D network map.

projection = net.project(data)



#Cluster the datapoints according to the Quality Threshold algorithm.

cluster = net.cluster(data, type='qthresh', show=True)