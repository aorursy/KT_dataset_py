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
import scipy
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
tf.__version__
mnist=input_data.read_data_sets("../input",one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)
print(mnist.train.images[0,:])
"""
save_dir="MNIST_data/raw/"
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)
for i in range(20):
    image_array=mnist.train.images[i,:]
    image_array=image_array.reshape(28,28)
    filename=save_dir+"mnist_train_%d.jpg"%i
    scipy.misc.toimage(image_array,cmin=0.0,cmax=1.0).save(filename)
for i in range(20):
    one_hot_label=mnist.train.labels[i,:]
    label=np.argmax(one_hot_label)
    print("mnist_train_%d.jpg label:%d"%(i,label))
"""
print(mnist.train.labels[0,:])
x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("../input",one_hot=True)
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_img=tf.reshape(x,[-1,28,28,1])
print(x_img.shape)
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
#第一层卷积
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_img,W_conv1)+b_conv1)
print(h_conv1.shape)
#tf.nn.conv2d(x=[?,28,28,1],w=[5,5,1,32],strides=[1,1,1,1],padding="SAME")
#卷积核大小为5*5,卷积核数量为32个
#垂直方向步长为1,水平方向步长为1,填充方式为全零.
#(28*28*32)
h_pool1=max_pool_2x2(h_conv1)
print(h_pool1.shape)
#tf.nn.max_pool(x=[50,24,24,32],ksize=[1,2,2,1],strides=[1,2,2,1],padding=["SAME"])
#(14*14*32)
#第二层卷积
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
print(h_conv2.shape)
#(12*12*64)
h_pool2=max_pool_2x2(h_conv2)
print(h_pool2.shape)
#(7*7*64)
#全连接层
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
print(h_pool2_flat.shape)
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
print(h_fc1.shape)
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
print(h_fc1_drop.shape)
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2
print(y_conv.shape)
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(1000):
    batch=mnist.train.next_batch(50)
    if i%100==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,
                                                  y_:mnist.test.labels,keep_prob:1.0}))
