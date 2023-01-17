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
train = pd.read_csv('../input/train.csv', parse_dates=['date'])
test = pd.read_csv('../input/test.csv', parse_dates=['date'])
print("Train shape: ", train.shape)
print("Test shape: ", test.shape)
df = pd.concat([train,test])
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.dayofweek
df['year'] = df['date'].dt.year
df['week_of_year']  = train.date.dt.weekofyear

df.drop('date', axis=1, inplace=True)
df.head()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df["median_store_item_month"] = df.groupby(['month',"item","store"])["sales"].transform("median")
df["mean_store_item_week"] = df.groupby(['week_of_year',"item","store"])["sales"].transform("mean")
df["item_month_sum"] = df.groupby(['month',"item"])["sales"].transform("sum")
df["store_month_sum"] = df.groupby(['month',"store"])["sales"].transform("sum")
df["item_week_shifted_90"] = df.groupby(['week_of_year',"item"])["sales"].transform(lambda x:x.shift(12).sum()) 
df["store_week_shifted_90"] = df.groupby(['week_of_year',"store"])["sales"].transform(lambda x:x.shift(12).sum()) 
df["item_week_shifted_90"] = df.groupby(['week_of_year',"item"])["sales"].transform(lambda x:x.shift(12).mean()) 
df["store_week_shifted_90"] = df.groupby(['week_of_year',"store"])["sales"].transform(lambda x:x.shift(12).mean())

train = df.loc[~df.sales.isna()]
train4 = train.copy()
train4.drop('id', axis=1, inplace=True)
train4.head()
corr = train4.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)
train = df.loc[~df.sales.isna()]
col = [i for i in train.columns if i not in ['id','store','item']]

from sklearn.preprocessing import LabelEncoder
train = train[col].apply(LabelEncoder().fit_transform)
train.head()
col = [i for i in train.columns if i not in ['id','sales','store','item']]
X_train=train[col].values

Y_train=train['sales'].values
Y_train=Y_train.reshape((913000,1))

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
from sklearn import cross_validation
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X_train,Y_train, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
import tensorflow as tf
def layer(output_dim,input_dim,inputs,activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W)+b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs
X = tf.placeholder("float", [None, 10])
h1 = layer(20,10,X,activation=tf.nn.relu)
y_predict = layer(1, 20, h1, activation=None)
y_label = tf.placeholder("float", [None, 1]) 
MSE=tf.losses.mean_squared_error(labels=y_label,predictions=y_predict)
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(MSE)
SMAPE = tf.reduce_mean(tf.divide(tf.abs(y_predict-y_label),tf.add(y_label,y_predict)))
import math
def batches(batch_size, features,labels):
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch1 = features[start_i:end_i]
        batch2 = labels[start_i:end_i]
    return batch1,batch2
trainEpochs = 20
batchSizes = 1000
totalBatchs = int(913000/batchSizes)

epoch_list = []
MSE_list = []
SMAPE_list = []
from time import time
startTime = time()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x, batch_y = batches(batchSizes, x_train, y_train)
        sess.run(optimizer,feed_dict={X: batch_x, y_label: batch_y})
    mse,smape = sess.run([MSE,SMAPE],feed_dict={X: x_test,y_label: y_test})
    epoch_list.append(epoch)
    MSE_list.append(mse)
    SMAPE_list.append(smape)
    print("Train Epoch:", '%02d' % (epoch+1), "MES=", "{:.9f}".format(mse), "SMAPE=", smape)
duration = time() - startTime
print("Train Finished takes:", duration)
%matplotlib inline
fig = plt.gcf()
fig.set_size_inches(10,6)
plt.plot(epoch_list, MSE_list, label='MES')
plt.ylabel('mean square error')
plt.xlabel('epoch')
plt.legend(['mean square error'], loc='upper left')
%matplotlib inline
fig = plt.gcf()
fig.set_size_inches(10,6)
plt.plot(epoch_list, SMAPE_list, label='SMAPE')
plt.ylabel('Symmetric mean absolute percentage error')
plt.xlabel('epoch')
plt.legend(['SMAPE'], loc='upper left')