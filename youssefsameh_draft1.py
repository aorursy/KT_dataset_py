# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


train = pd.read_csv("../input/forest-cover-type-kernels-only/train.csv.zip")
test = pd.read_csv("../input/forest-cover-type-kernels-only/test.csv.zip")
print(train.isnull().sum())
print("The number of traning examples(data points) = %i " % train.shape[0])
print("The number of features we have = %i " % train.shape[1])
import seaborn as sns


import matplotlib.pyplot as plt

corr = train.corr()
f, ax = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.75, center=0,
            square=True, linewidths=.3)
train.drop(['Id'], inplace = True, axis = 1 )
train.drop(['Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
test.drop(['Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
import time
import matplotlib.pyplot as plt
classes = np.array(list(train.Cover_Type.values))

def plotRelation(first_feature, sec_feature):
    
    plt.scatter(first_feature, sec_feature, c = classes, s=10)
    plt.xlabel(first_feature.name)
    plt.ylabel(sec_feature.name)
f = plt.figure(figsize=(30,40))
k = 0
for i in train.columns:
    for j in train.columns:
            f.add_subplot(332)
            coll = train.loc[:,i]
            row = train.loc[:,j]
            if coll.corr(row) > 0.2 and coll.corr(row)<0.9999  : #this is gonna take some time just change the lower bound of the corr 
                plotRelation(coll,row)

        