!pip install -q efficientnet
import os
import gc
import re

import cv2
import math
import numpy as np
import scipy as sp
import pandas as pd

import tensorflow as tf
from IPython.display import SVG
import efficientnet.tfkeras as efn
from keras.utils import plot_model
import tensorflow.keras.layers as L
from keras.utils import model_to_dot
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from kaggle_datasets import KaggleDatasets
from tensorflow.keras.applications import DenseNet121

import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

tqdm.pandas()
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

np.random.seed(0)
tf.random.set_seed(0)

import warnings
warnings.filterwarnings("ignore")
inputpath='/kaggle/input/plant-pathology-2020-fgvc7/'
imagepath=inputpath+'images/'
traindata=inputpath+'train.csv'
testdata=inputpath+'test.csv'
samplesub=inputpath+'sample_submission.csv'
train_df=pd.read_csv(traindata)
test_df=pd.read_csv(testdata)
train_df

train_df.head()
train_df.shape
shape = (512,256)

def getImage(image_id,SHAPE=shape):
    img = cv2.imread(imagepath + image_id + '.jpg')
    img = cv2.resize(img,SHAPE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img
fig1 = px.imshow(cv2.resize(train_images[1], (205, 136)))
fig1.show()
#blue values high imply brown part high. Greater the brown part(decay),greater the blue values. 
healthy = [getImage(image_id) for image_id in train_df[train_df['healthy']==1].iloc[:,0]]

multiple_diseases = [getImage(image_id) for image_id in train_df[train_df['multiple_diseases']==1].iloc[:,0]]

rust = [getImage(image_id) for image_id in train_df[train_df['rust']==1].iloc[:,0]]

scab = [getImage(image_id) for image_id in train_df[train_df['scab']==1].iloc[:,0]]


def displayImages(condition='healthy'):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))
    for i in range(9):
        image = random.choice(classes[condition])
        ax[i//3,i%3].imshow(image)
    fig.suptitle(f'{condition.capitalize()} Class Leaves',fontsize=20)
    plt.show()
    
classes = {'healthy':healthy, 'multiple_diseases':multiple_diseases, 'rust':rust, 'scab': scab} 

displayImages('healthy')
colors = ['rgb(200, 0, 0)', 'rgb(0, 200, 0)', 'rgb(0,0,200)']

def plotChannelDistribution(condition):
    
    distributions = []
        
    for channel in range(3):
        distributions.append([np.mean(img[:,:,channel]) for img in classes[condition]])
    
    fig = ff.create_distplot(distributions,
                            group_labels=['red','green','blue'],
                            colors=colors)
    
    fig.update_layout(title=f'{condition.capitalize()} leaves channel distribution')
    
    fig.show()
plotChannelDistribution('healthy')
#less blue part 
#greater the green part,greater the disease(?)
displayImages('multiple_diseases')
plotChannelDistribution('multiple_diseases')
#confirm:greater the blue part,greater the disease
channelDict = {'red':0,'green':1,'blue':2}

group_labels=[train_df.columns[i] for i in range(1,5)]

colors_cw = {'red':['rgb(250,0,0)','rgb(190,0,0)','rgb(130,0,0)','rgb(50,0,0)'],
         'green':['rgb(0,250,0)','rgb(0,190,0)','rgb(0,130,0)','rgb(0,50,0)'],
         'blue':['rgb(0,0,250)','rgb(0,0,190)','rgb(0,0,130)','rgb(0,0,50)']}

def plotChannelWiseDistribution(channel):
    
    distributions = []
    
    for c in [healthy, multiple_diseases, rust, scab]:
        distributions.append([np.mean(img[:,:,channelDict[channel]]) for img in c])
    
    fig = ff.create_distplot(distributions,
                            group_labels=group_labels,
                            colors=colors_cw[channel],
                            show_hist=False)
    
    fig.update_layout(title=f'{channel.capitalize()} channel distribution for all Classes')
    
    fig.show()
plotChannelWiseDistribution('blue')
#our prediction of greater the blue part,greater the disease turned out to be true from vis
plotChannelWiseDistribution('red')
#almost same for all,doesn't matter 
plotChannelWiseDistribution('green')
#almost same for all.Thus,green channel plays no role.
