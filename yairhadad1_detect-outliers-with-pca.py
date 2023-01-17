# Basic lib 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



# PCA

from sklearn.decomposition import PCA



#ignore warning messages 

import warnings

warnings.filterwarnings('ignore') 



sns.set()
# Get the train and test data 

train = pd.read_csv("../input/train.csv")

# test = pd.read_csv("../input/train.csv")



# Splite data the X - Our data , and y - the prdict label

X = train.drop('label',axis = 1)

y = train['label'].astype('category')



X.head()
dataset_7 = X[y == 7].reset_index().drop("index",axis = 1)



plt.figure(figsize = (10,8))

row, colums = 3, 3

for i in range(9):  

    plt.subplot(colums, row, i+1)

    plt.imshow(dataset_7.iloc[i].values.reshape(28,28),interpolation='nearest', cmap='Greys')

plt.show()

n_components = 5 # the components number can change to every number that you want.



# Create PCA with components number

pca = PCA(n_components = n_components)

# fit transform with PCA on dataset

pca_dataset_7 = pca.fit_transform(dataset_7)

# inverse transform back to regular dataset 

inverse_transform_dataset_7 = pca.inverse_transform(pca_dataset_7)



print("dataset_7 shape",dataset_7.shape)

print("pca_dataset_7 shape",pca_dataset_7.shape)

print("inverse_transform_dataset_7 shape",inverse_transform_dataset_7.shape)
# Check the diffrent between X and the inverse_transform_X

# (X-inverse_transform_X)**2) = MSE for any pixel, and we make sum() for get image MSE.

MSE_score = ((dataset_7-inverse_transform_dataset_7)**2).sum(axis=1)



MSE_score.head()
MSE_max_scores = MSE_score.nlargest(9).index



plt.figure(figsize = (10,8))

row, colums = 3, 3

for i in range(9):  

    plt.subplot(colums, row, i+1)

    plt.imshow(dataset_7.iloc[MSE_max_scores[i]].values.reshape(28,28),interpolation='nearest', cmap='Greys')

plt.show()
plt.figure(figsize = (10,8))

row, colums = 5, 10

    

for number in range(10):

    # Get the current number

    dataset = pd.DataFrame(X[(y == number)].reset_index().drop("index",axis = 1))

    # Create PCA with components number

    pca = PCA(n_components = n_components)

    # fit transform with PCA on dataset

    pca_dataset = pca.fit_transform(dataset)

    # inverse transform back to regular dataset 

    inverse_transform_dataset = pca.inverse_transform(pca_dataset)

    MSE_score = ((dataset-inverse_transform_dataset)**2).sum(axis=1)

    MSE_worst = MSE_score.nlargest(5).index # get max

    for number2 in range(0,5):

        plt.subplot(colums, row, (number2+(number*5))+ 1)

        plt.imshow(dataset.iloc[MSE_worst[number2]].values.reshape(28,28),interpolation='nearest', cmap='Greys')

plt.show()