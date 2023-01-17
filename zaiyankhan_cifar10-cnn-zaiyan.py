from keras.datasets import cifar10

from __future__ import print_function

(x_train, y_train), (x_test, y_test) = cifar10.load_data()



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

import tensorflow as tf



tf.reset_default_graph()



NUM_CLASSES = 10



X = tf.placeholder(tf.float32, [None, 32, 32, 3])

Y = tf.placeholder(tf.int64, [None,1])



X_extend = tf.reshape(X, [-1, 32, 32, 3])

Y_onehot = tf.one_hot(indices=Y, depth=NUM_CLASSES)



"""First Conv Layer"""

con1_w = tf.get_variable("con1_w", [3,3,3,32], initializer=tf.random_normal_initializer(stddev=1e-2))

con1 = tf.nn.conv2d(X_extend, con1_w, strides=[1,1,1,1], padding='SAME')

relu1 = tf.nn.relu(con1)

pool1 = tf.nn.max_pool(value=relu1,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')



"""Second Conv Layer"""

con2_w = tf.get_variable("con2_w", [5,5,32,64], initializer=tf.random_normal_initializer(stddev=1e-2))

con2 = tf.nn.conv2d(pool1, con2_w, strides=[1,1,1,1], padding='SAME')

relu2 = tf.nn.relu(con2)

pool2 = tf.nn.max_pool(value=relu2,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')



"""Third Conv Layer"""

con3_w = tf.get_variable("con3_w", [3,3,64,64], initializer=tf.random_normal_initializer(stddev=1e-2))

con3 = tf.nn.conv2d(pool2, con3_w, strides=[1,1,1,1], padding='SAME')

relu3 = tf.nn.relu(con3)



print(relu3)
"""Flatten Layer"""

flatten = tf.reshape(relu3, [-1, 8*8*64])



"""FC 1"""

fc1 = tf.layers.dense(inputs=flatten,units=512,activation=tf.nn.relu,use_bias=True)



"""FC 2"""

fc2 = tf.layers.dense(inputs=fc1,units=512,activation=tf.nn.relu,use_bias=True)



"""Output Layer"""

out = tf.layers.dense(inputs=fc2,units=NUM_CLASSES,activation=None,use_bias=True)



"""Loss Func"""

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=out))



"""Acc Func"""

acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out,axis=1),Y[:,0]), dtype=tf.float32))



"""Optimizer"""

opt = tf.train.AdamOptimizer(0.0001).minimize(loss)
"""INIT"""



sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)
from tqdm import tqdm_notebook as tqdm



EPOCHS = 10

BATCH_SIZE = 64



for epoch in range(0,EPOCHS):

    for step in tqdm(range(int(len(x_train)/BATCH_SIZE)), desc=('Epoch '+str(epoch))):

        x_batch = x_train[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE]

        y_batch = y_train[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE]

        

        loss_val = sess.run([loss, opt], feed_dict={X: x_batch, Y: y_batch})

        

    loss_val, acc_val = sess.run([loss,acc], feed_dict={X: x_test[:1000], Y: y_test[:1000]})    

    print('Epoch Loss: ', loss_val, 'Accuracy: ', acc_val)
### import numpy as np

import matplotlib.pyplot as plt

import numpy as np



con1_ex = sess.run(con1_w)

print(con1_ex.shape)

plt.figure(figsize=(20,20))

for i in range(10):

    plt.subplot(1,10,i+1)

    plt.imshow(np.reshape(con1_ex[:,:,:,i]*100, [3,3,3]))
conv1_fmaps = sess.run(relu1, feed_dict={X:[x_train[0]]})

print(conv1_fmaps.shape)



plt.figure(figsize=(20,20))

for i in range(10):

    plt.subplot(1, 10, i+1)

    plt.imshow(np.reshape(conv1_fmaps[0,:,:,i]*100, [32,32]))