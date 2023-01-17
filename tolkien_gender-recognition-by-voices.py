import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
_data = pd.read_csv('../input/voice.csv')
v_data = _data.reindex(columns=list(_data.columns) + ['gender'])
#v_data.dtypes
_len = len(v_data)
print(_len)
print(v_data.at[3167, 'label'])
for i in range(_len):
    if v_data.at[i, 'label'] == 'male':
        v_data.at[i, 'gender'] = 1.0
    else:
        v_data.at[i, 'gender'] = 0.0
print(v_data.at[3167, 'gender'])
v_data = v_data.drop(['label'], axis=1)
# shuffling
v_data = v_data.reindex(np.random.permutation(v_data.index))
print(len(v_data.columns), v_data.columns)
_split = _len - 1000
train = v_data[0:_split]
test = v_data[_split:-1]
print(len(train), len(test))
print(train.tail(2))
# print(test.head(2))
# print(v_data.tail(2))
# print(test.tail(2))
# v_data.size
# v_data.columns
print(train.describe())
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(train.corr(), vmax=.8, square=True);
def strongerRelationSalePrice(f1, f2):
    f1Corr = train.corr().loc[f1,'gender']
    f2Corr = train.corr().loc[f2,'gender']
#     print(f1Corr, f2Corr)
    return (f1, f2) if (f1Corr >= f2Corr) else (f2, f1)

def print_stronger(f1, f2):
    print('{} > {}'.format(strongerRelationSalePrice(f1, f2)[0], strongerRelationSalePrice(f1, f2)[1]))
print_stronger('meanfreq', 'median')
print_stronger('meanfreq', 'Q25')
print_stronger('meanfreq', 'Q75')
print_stronger('meanfreq', 'mode')
print_stronger('meanfreq', 'centroid')
print_stronger('sd', 'IQR')
print_stronger('sd', 'sfm')
print_stronger('median', 'Q25')
print_stronger('median', 'Q75')
print_stronger('median', 'mode')
print_stronger('median', 'centroid')
print_stronger('Q25', 'centroid')
print_stronger('Q75', 'centroid')
print_stronger('skew', 'kurt')
print_stronger('sp.ent', 'sfm')
print_stronger('mode', 'centroid')
print_stronger('meandom', 'maxdom')
print_stronger('meandom', 'dfrange')
print_stronger('maxdom', 'dfrange')
print_stronger('mode', 'Q75')
train = train.drop(['mode', 'meanfreq', 'centroid', 'median', 'Q25', 'sd', 'sfm', 'skew', 'sfm', 'dfrange', 'maxdom'], axis=1)
test = test.drop(['mode', 'meanfreq', 'centroid', 'median', 'Q25', 'sd', 'sfm', 'skew', 'sfm', 'dfrange', 'maxdom'], axis=1)
print(len(train.columns), train.columns)
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(train.corr(), vmax=.8, square=True)
%matplotlib inline
import matplotlib.pyplot as plt

# I think this graph is more elegant than pandas.hist()
# train['SalePrice'].hist(bins=100)
sns.distplot(train['gender'])
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 6, figsize=(15, 7), sharey=True)
for col, a in zip(train.columns, axes.flatten()):
    if col == 'gender':   
        a.set_title(col)
        a.scatter(df['gender'], df['gender'])
    else:
        df = train[['gender', col]].dropna()
        a.set_title(col)
        a.scatter(df[col], df['gender'])
# Lab 5 Logistic Regression Classifier
import tensorflow as tf
import numpy as np
tf.set_random_seed(743)  # for reproducibility

# collect data
x_data = train.loc[:,['Q75','IQR','kurt','sp.ent']].values
y_data = train.loc[:,['gender']].values
print(x_data[0],y_data[0])
#len(x_data)
type(x_data)
#y_data
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2000):
    cost_val, _ = sess.run([cost, optimizer], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, cost_val)

x_test = test.loc[:,['Q75','IQR','kurt','sp.ent']].values
y_test = test.loc[:,['gender']].values
h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_test, Y: y_test})
print("Accuracy: ", a)
print(c[0:8], y_test[0:8])