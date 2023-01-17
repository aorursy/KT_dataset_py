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

#full.info()
train.shape
training = train.head(40000)

validation = train.tail(2000)



training_Y = training["label"]

training_X = training.drop(labels=["label"], axis=1)





validation_Y = validation["label"]

validation_X = validation.drop(labels=["label"], axis=1)

#設定學習率,學次數等參數

learning_rate = 0.001

training_epochs = 100

display_step = 10

batch_size = 100



#設定X的輸入數,神經網絡層等參數

n_inputs = 28*28 # MNIST

#units = [900, 400, 100]

#units = [900, 900, 900, 900]

units = [100,100,100,100,100,100, 100, 100, 100]

n_outputs = 10

dropout_rate = 0.5





X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

y = tf.placeholder(tf.int64, shape=(None), name="y")



tr = tf.placeholder_with_default(False, shape=(), name='training')



#循環構建神經網絡層,激活函數設為relu,並使用dropout函數提高訓練效果

dense = X

he_init = tf.keras.initializers.he_normal()

for unit in units:

    dropout = tf.layers.dropout(dense, dropout_rate, training=tr)

    dense = tf.layers.dense(inputs=dropout, units=unit, activation=tf.nn.relu)

logits= tf.layers.dense(inputs=dense, units=n_outputs, activation=None)



#sparse_softmax_cross_entropy_with_logits函數將求出logits的輸出和實際結果的y之間的交叉熵,以此作為損失函數

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,logits = logits)

loss = tf.reduce_mean(xentropy, name='loss')



#計算出logits的結果和y是否相同,計算出模型準確率

correct = tf.nn.in_top_k(logits ,y ,1)

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



#經過測試後AdamOptimizer的結果最佳,亦可在此處改用其他優化器作測試

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)

#optimizer = tf.train.FtrlOptimizer(learning_rate).minimize(loss)







threshold = 1.0



def ClipIfNotNone(grad, threshold):

    if grad is None:

        return grad

    return tf.clip_by_value(grad, -threshold, threshold)

optimizer = tf.train.AdamOptimizer(learning_rate)

grads_and_vars = optimizer.compute_gradients(loss)

capped_gvs = [(ClipIfNotNone(grad, threshold), var)

              for grad, var in grads_and_vars]

training_op = optimizer.apply_gradients(capped_gvs)

optimizer = training_op



init = tf.global_variables_initializer()

#開始訓練

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(training_epochs):

        for iteration in range(training.shape[0] // batch_size):

            sample = training.sample(batch_size)

            y_batch = sample["label"]

            X_batch = sample.drop(labels=["label"], axis=1)

            sess.run(optimizer, feed_dict={X: X_batch, y: y_batch})

        if (epoch + 1) % display_step == 0:

            a = sess.run(accuracy, feed_dict={X: training_X, y: training_Y})

            va = sess.run(accuracy, feed_dict={X: validation_X, y: validation_Y})

            print("Epoch:", '%04d' % (epoch + 1), "accuracy=", a, "v_accuracy=", va)

    print("Optimization Finished!")

    

    l = sess.run(logits, feed_dict={X: test})

    p = sess.run(tf.argmax(l, axis=1))
result = pd.DataFrame()

result['ImageId'] = test.index + 1

result['label'] = p

result.to_csv('result.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(result)