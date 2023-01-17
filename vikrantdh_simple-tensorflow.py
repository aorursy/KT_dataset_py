import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
dftrain = pd.read_csv("../input/train.csv")

dftest = pd.read_csv("../input/test.csv")
dftrain.head()
dftest.head()
X_train = dftrain.drop('label',axis=1)

y_train = dftrain['label']

X_test = dftest
plt.imshow(X_train.iloc[26].values.reshape(28,28),cmap='gist_gray')

plt.show()

from keras.utils.np_utils import to_categorical

y_train= to_categorical(y_train)
x = tf.placeholder(tf.float32,shape=[None,784])

W = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x,W) + b 
y_true = tf.placeholder(tf.float32,shape=[None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()

session = tf.Session()

session.run(init)
for step in range(1500):

        

    session.run(train,feed_dict={x:X_train,y_true:y_train})

        

pred = tf.argmax(y,1)
submissions=pd.DataFrame({"ImageId":range(1,28001),"Label": pred.eval(feed_dict={x: X_test}, session=session)})

submissions.to_csv("submissions.csv", index=False, header=True)