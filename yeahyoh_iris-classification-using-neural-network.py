import pandas as pd

import tensorflow as tf

from numpy import array
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
irisdf = pd.read_csv("../input/Iris.csv")

irisdf["Species"] = irisdf["Species"].map({

    "Iris-setosa": 0,

    "Iris-versicolor": 1,

    "Iris-virginica": 2

}).astype(int)
x_train = irisdf[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

y_train = irisdf['Species']
#encode the y_train to one-hot form

new_y = []

for i in y_train:

    a = [0,0,0]

    a[i] = 1

    new_y.append(a)

y_train = array(new_y)
print(x_train.shape)

print(y_train.shape)
x = tf.placeholder(tf.float32, [None,4])

y_ = tf.placeholder(tf.float32, [None,3])
W = tf.Variable(tf.zeros([4,3]))

b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.matmul(x,W)+ b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()

sess.run(init)
for i in range(2000):

    sess.run(train_step,feed_dict={x:x_train, y_:y_train})
print(sess.run(accuracy, feed_dict={x:x_train,y_:y_train}))