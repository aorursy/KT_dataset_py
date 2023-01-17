# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import math

import array

from IPython import display

from matplotlib import cm

from matplotlib import gridspec

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

from sklearn import metrics

import tensorflow as tf

from tensorflow.python.data import Dataset

import datetime

import random

import seaborn as sns

from functools import reduce

rng = np.random

tf.logging.set_verbosity(tf.logging.ERROR)

pd.options.display.max_rows = 10

pd.options.display.float_format = '{:.2f}'.format
train = pd.read_csv("../input/train.csv", sep=",")

test = pd.read_csv("../input/test.csv", sep=",")

full = train.append(test)



train = train.reindex(np.random.permutation(train.index))

full.info()
train.head(5)
train.describe()
corrmat = train.corr()

f, ax = plt.subplots(figsize=(80, 60))

colormap = plt.cm.RdBu

sns.heatmap(corrmat,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)



k=11



feature_cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cmk = train[feature_cols].corr()

f, ax = plt.subplots(figsize=(20, 15))

colormap = plt.cm.RdBu

sns.heatmap(cmk,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
sns.set()

sns.pairplot(train[feature_cols], size = 2)

plt.show();
train[feature_cols].head(10)
train[feature_cols].describe()
def combine_feature(train):

    train['Garage'] = train['GarageCars'] * train['GarageArea']

    train['SF'] = train['TotalBsmtSF'] + train['1stFlrSF']

    train['TG'] = train['TotRmsAbvGrd'] * train['GrLivArea']



    

combine_feature(train)

combine_feature(test)

train = train.fillna(0)

test = test.fillna(0)
feature_cols2 = feature_cols.tolist()

feature_cols2.extend(['Garage','SF','TG'])

print(feature_cols2)

cmk2 = train[feature_cols2].corr()

f, ax = plt.subplots(figsize=(20, 15))

colormap = plt.cm.RdBu

sns.heatmap(cmk2,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
cols = feature_cols2

for e in ['SalePrice','GarageCars', 'GarageArea','TotalBsmtSF', '1stFlrSF','TG']:

    cols.remove(e)
cols
var = 'GrLivArea'

train.plot.scatter(x=var, y='SalePrice');
train = train[train['GrLivArea']<4500]

train = train[train['SalePrice']< 700000]
var = 'GrLivArea'

train.plot.scatter(x=var, y='SalePrice');
def normalize(data, col):

    data[col] = (data[col] - data[col].mean()) / (data[col].max() - data[col].mean())

    

def trans(data, feature_columns):

    trans = []

    for col in feature_columns:

        trans.append(list(data[col])) 

    return trans
training_example = train.head(1000)

validation_example = train.tail(train.shape[0] - 1000)

test_example = test.copy()



train.shape[0]
# 参数

learning_rate = 0.1

training_epochs = 100

display_step = 10



feature_columns = cols

features_num = len(feature_columns)



training_X = []

validation_X = []



for col in feature_columns:

    normalize(training_example, col)

    normalize(validation_example, col)

    normalize(test_example, col)



training_X = training_example[feature_columns]

validation_X = validation_example[feature_columns]

test_X = test_example[feature_columns]

#print(training_X)

training_Y = training_example['SalePrice']

validation_Y = validation_example['SalePrice']

n_samples = tf.placeholder("float")



X = tf.placeholder("float",[features_num, None])

Y = tf.placeholder("float")



W =  tf.Variable(tf.zeros([1, features_num],name="weight"))  

b = tf.Variable(tf.zeros([1]), name="bias")



pred = tf.add(tf.matmul(W, X), b)

cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (n_samples)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

optimizer = tf.train.MomentumOptimizer(learning_rate,momentum = 0.9).minimize(cost)

init = tf.global_variables_initializer()
result = []

with tf.Session() as sess:

    sess.run(init)

    training_XT = sess.run(tf.transpose(training_X[feature_columns]))

    validation_XT = sess.run(tf.transpose(validation_X[feature_columns]))

    for epoch in range(training_epochs):

        for index, row in training_X.iterrows():

            x = [ [i] for i in row[feature_columns]]

            sess.run(optimizer, feed_dict={X: x, Y: training_Y[index], n_samples:training_Y.shape[0]})

        if (epoch + 1) % display_step == 0:

            c = sess.run(cost, feed_dict={X: training_XT, Y: training_Y, n_samples:training_Y.shape[0]})            

            vc = sess.run(cost, feed_dict={X: validation_XT, Y: validation_Y, n_samples:validation_Y.shape[0]})

            print("Epoch:", '%04d' % (epoch + 1), "cost=", c, "v_cost=", vc,"W=", sess.run(W), "b=", sess.run(b))

            result.append([epoch, c, vc])

    print("Optimization Finished!")

    training_cost = sess.run(cost, feed_dict={X: training_XT, Y: training_Y, n_samples:training_Y.shape[0]})

    validation_cost = sess.run(cost, feed_dict={X: validation_XT, Y: validation_Y, n_samples:validation_Y.shape[0]})

    print("Training cost=", training_cost,"Validation cost=",validation_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    

    r = pd.DataFrame(result, columns = ['epoch', 'cost', 'validation cost']) 

    sns.pointplot(x="epoch",y="cost",data=r, markers="^")

    sns.pointplot(x="epoch",y="validation cost",data=r)

    

    p = sess.run(pred, feed_dict={X: trans(test_X,feature_columns)})

    print('pred: ',p)

    
result = pd.DataFrame()

result['Id'] = test.Id

result['SalePrice'] = p[0]

result.to_csv('result.csv', index=False)