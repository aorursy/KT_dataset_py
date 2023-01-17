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
import tensorflow as tf
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')
no_of_classes = 10
batch_size= 128
def conv2d(x, W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1] ,padding ='SAME')
def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1] , padding ='SAME')
def convloutional_neural_network(x):
    weights ={ 'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
              'W_conv2': tf.Variable(tf.random_normal([5,5,32,64])),
              'W_fc': tf.Variable(tf.random_normal([7*7*64,1024])),
              'out': tf.Variable(tf.random_normal([1024,no_of_classes]))
             }
    biases ={ 'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([no_of_classes]))
             }
    x=tf.reshape(x,shape=[-1,28,28,1])
    
    conv1=conv2d(x,weights['W_conv1'])
    conv1=maxpool2d(conv1)
    
    conv2=conv2d(conv1,weights['W_conv2'])
    conv2=maxpool2d(conv2)
    
    fc=tf.reshape(conv2,[-1,7*7*64])
    fc=tf.nn.relu(tf.matmul(fc,weights['W_fc']) + biases['b_fc'])
    
    output = tf.matmul(fc,weights['out']) + biases['out']
    return output
def train_network(x):
    prediction= convloutional_neural_network(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    epocks=10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epok in tqdm(range(epocks)):
            epok_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epok_x,epok_y=mnist.train.next_batch(batch_size)
                _, c=sess.run([optimizer,cost],feed_dict={x : epok_x, y: epok_y})
                epok_loss+=c
            print('Epok ',epok, 'Completed out of ',epocks, 'With loss =',epok_loss)
        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy : ',accuracy.eval({x:mnist.test.images,y: mnist.test.labels}))
train_network(x)
