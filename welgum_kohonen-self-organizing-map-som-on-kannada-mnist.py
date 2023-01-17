import numpy as np 

import pandas as pd 

import seaborn as sns

from imageio import imwrite

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from matplotlib.cbook import get_sample_data

import random

%matplotlib inline

from sklearn.preprocessing import StandardScaler

from PIL import Image, ImageChops

!pip install SimpSOM
import SimpSOM as sps
np.random.seed(0)



# get part of the dataset

train = pd.read_csv('../input/Kannada-MNIST/train.csv')

train = train.sample(n=600, random_state=0)

labels = train['label']

train = train.drop("label",axis=1)



# check distribution

sns.countplot(labels)



# standardization of a dataset

train_st = StandardScaler().fit_transform(train.values)
# build a network 20x20 with a weights format taken from the train_st and activate Periodic Boundary Conditions. 

som = sps.somNet(20, 20, train_st, PBC=True)



# train it with 0.1 learning rate for 10000 epochs

som.train(0.05, 10000)



# print unified distance matrix

som.diff_graph(show=True, printout=True)
fig, axs = plt.subplots(10, 10, figsize=(20, 20))

axs = axs.flatten()



some_nodes_indxs = random.sample(range(len(som.nodeList)), len(axs))





for i, ax in enumerate(axs):

    ax.imshow(np.asarray(som.nodeList[some_nodes_indxs[i]].weights).reshape(28,28))

    ax.axis('off')

# print picked coordinates

', '.join([f'({i // 20}, {i % 20})' for i in some_nodes_indxs])

som.cluster(train_st, type='qthresh', show=True);