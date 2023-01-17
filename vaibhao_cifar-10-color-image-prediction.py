# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd, matplotlib.pyplot as plt

%matplotlib inline
def unpickle(file):

    import pickle

    with open(file, 'rb') as fo:

        cifar_dict = pickle.load(fo, encoding='bytes')

    return cifar_dict



dirs = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']

all_data = [0,1,2,3,4,5]

for i,direc in zip(all_data,dirs):

    all_data[i] = unpickle('../input/cifar10/'+ direc)



metaData = unpickle('../input/batches-meta-for-cfar10/'+ 'batches.meta')

test_batch = unpickle('../input/cifar10/'+ 'test_batch')
#Data Setup in variables 

data_batch0 = metaData

data_batch1 = all_data[0]

data_batch2 = all_data[1]

data_batch2 = all_data[2]

data_batch3 = all_data[3]

data_batch4 = all_data[4]

data_batch5 = all_data[5]

test_batch = test_batch
data_batch1.keys()
plt.imshow(data_batch1[b'data'][4].reshape(3,32,32).transpose(1,2,0))
#Preprocessing before showing the Data

data_batch1[b'data'].shape  #----->  color image of 10000 samples and 3072 pixel (RGB) ((10000, 3072))

X = data_batch1[b'data']
X = X.reshape(10000,3,32,32)
X.shape
X = X.transpose(0,2,3,1).astype("uint8")
plt.imshow(X[1].astype("uint8"))
data_batch2[b'data'].shape
def one_hot_encoding(vec, val= 10):

    n = len(vec)

    out = np.zeros((n,val))

    out[range(n),vec] = 1

    return out
one_hot_encoding([4,2,3],10)
#class for batch setup

class Cifar():

    def __init__(self):

        self.i = 0

        #Grabs a list of all the data batches for training

        self.all_Images = [data_batch1,data_batch2,data_batch3, data_batch4,data_batch5]

        #Grabs a list of all the test batches for testing

        self.test_batch = [test_batch]

        #Initialising some variables for later on Purpose

        self.training_images =None

        self.training_labels = None

        self.test_images =None

        self.test_labels = None



        

    def set_up_Image(self):

        #Stacking up the image data vertically

        self.training_images = np.vstack([data[b'data'] for data in self.all_Images])

        self.train_length = len(self.training_images)

        #Setting up the Training data

        self.training_images = self.training_images.reshape(self.train_length,3,32,32).transpose(0,2,3,1)/255.0

        self.training_labels = one_hot_encoding(np.hstack([label[b'labels'] for label in self.all_Images]),10)

    #------------------------------------------

    #setting up Test Images

        self.test_images = np.vstack([data[b'data'] for data in self.test_batch])

        self.test_length = len(self.test_images)

        self.test_images = self.test_images.reshape(self.test_length,3,32,32).transpose(0,2,3,1)/255.0

        self.test_labels = one_hot_encoding(np.hstack([label[b'labels'] for label in self.test_batch]),10)

        

        

    def next_Batch(self,size):

        x = self.training_images[self.i : (self.i + size)].reshape(size,32,32,3)  # need to verify (size,3,32,32).transpose(0,2,3,1) as wll

        y = self.training_labels[self.i : (self.i + size)]

        self.i = int(self.i + size) % self.train_length

        return (x,y)

        
#setting the images 

cp =Cifar()

cp.set_up_Image()
import tensorflow as tf
def init_weights(shape):

    init_Weight = tf.truncated_normal(shape= shape, stddev=0.2)

    return tf.Variable(init_Weight)

#---------------------------

def init_bias(shape):

    init_Bias = tf.constant(0.2,shape=shape)

    return tf.Variable(init_Bias)

#---------------------------

def conv2d(x,W):

    '''x ---> input tensor of 4d length (tensor length, ht,widht, color channel)

        w ---> filter used for Image processing of 4 D (filter ht,filter width, input channel, outpul channel) 

    '''

    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

#---------------------------

def maxPooling(x):

    return tf.nn.max_pool(x,ksize=[1,1,1,1],strides=[1,2,2,1],padding="SAME")



#---------------------------

def convolution_layer(input_x, shape):

    init_Weight = init_weights(shape)

    init_Bias = init_bias ([shape[3]])

    return tf.nn.relu(conv2d(input_x,init_Weight)+ init_Bias)

#---------------------------

def fully_connected_layer(input_x, size):

    init_size = int(input_x.get_shape()[1])

    init_Weight = init_weights([init_size,size])

    init_Bias = init_bias([size])

    return tf.matmul(input_x,init_Weight)+init_Bias

#---------------------------

X = tf.placeholder(dtype=tf.float32,shape=[None,32,32,3])

y_true = tf.placeholder(dtype=tf.float32,shape=[None,10])

hold_prob = tf.placeholder(dtype=tf.float32)
X_image = tf.reshape(X,[-1,32,32,3]) # reshaping the image 

#layers Design

con1 = convolution_layer(X_image,[6,6,3,32])

pool1 = maxPooling(con1)

con2 = convolution_layer(pool1,[6,6,32,64])

pool2 = maxPooling(con2)

#Fully connected Layer desing but we have to flatten the data before sending it to final layer

flatData = tf.reshape(pool2,[-1, 8*8*64])

fullLayer = tf.nn.relu(fully_connected_layer(flatData,1024))



#Create a drop out 

fullDropout = tf.nn.dropout(fullLayer,keep_prob=hold_prob)



# Create prediction model

y_pred = fully_connected_layer(fullDropout,10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

optim = tf.train.AdamOptimizer(learning_rate=0.005)

train = optim.minimize(cross_entropy)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    steps = 10000

    for i in range(steps):

        batch_x, batch_y = cp.next_Batch(200)

        sess.run(train,feed_dict= {X : batch_x, y_true : batch_y, hold_prob:0.5})

        #printing Results for every 100 iterations

        if i%100 ==0:

            print('on step number {}'.format(i))

            print('Accuracy is :')

            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            print(sess.run(acc, feed_dict={X :cp.test_images,y_true: cp.test_labels, hold_prob: 1.0}))

        

        

    