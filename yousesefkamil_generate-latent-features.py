import pandas as pd 

import random 

import os

import numpy as np 

import matplotlib.pyplot as plt 

%matplotlib inline 

import statsmodels.api as sm 

import tensorflow as tf 

from sklearn import ensemble 

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

from tqdm import tqdm 

from sklearn.model_selection import train_test_split 

import seaborn as sns

from tensorflow import keras 

#! conda install -c conda-forge gdcm -y

#! pip install pylibjpeg pylibjpeg-libjpe
import pickle

with open('../input/segmented-data/segmentedDataDict.pkl', 'rb') as f:

    dic= pickle.load(f)

with open('../input/segmented-data/segmentedData.npy', 'rb') as f:

    CTs= np.load(f)

AE=tf.keras.models.load_model("../input/autoencoder/AE.h5")



CTs[CTs<-2000]=-2000

CTs=CTs*-1

CTs=CTs/2000



CTs=CTs.reshape(len(CTs),32,256,256,1)



from keras import backend as K



# with a Sequential model

get_3rd_layer_output = K.function([AE.layers[0].input],

                                  [AE.layers[10].output])

for i in range(len(CTs)):

    LF=get_3rd_layer_output([CTs[i].reshape(1,32,256,256,1)])[0] #LF latent features of every image

    #print(LF)

    if i==0:

        latent_features=LF

    else:

        latent_features=np.concatenate((latent_features,LF))



        latent_f_tabular=[]

        

ROOT = "../input/osic-pulmonary-fibrosis-progression"



tr = pd.read_csv(f"{ROOT}/train.csv")

tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])

tr=tr[tr.Patient.isin(dic.keys())]

latent_f_tabular=[]

for i in range(len(tr)):

    ind=dic[tr["Patient"].iloc[i]]

    latent_f_tabular.append(latent_features[ind])

latent_f_tabular=np.array(latent_f_tabular)
latent_f_tabular.shape
for i in range(20):

    tr.insert(i+1,i,latent_f_tabular[:,i])
tr.drop(['Weeks','FVC','Percent','Age','Sex','SmokingStatus'],axis='columns', inplace=True)
tr.to_csv("/kaggle/working/latent features.csv", index=False)