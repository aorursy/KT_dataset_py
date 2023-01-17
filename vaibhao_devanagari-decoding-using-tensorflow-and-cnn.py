# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = shuffle(pd.read_csv('../input/data.csv'))
data.head()
#visualise the data

plt.imshow(data.drop('character', axis=1).iloc[0].reshape(32,32))
#Character encoding

def convert_to_Number(label):

    if 'character' in label.split('_'):

         return int(label.split('_')[1]) - 1  #making index from 0

    else:

        for j in range(0,10):

            if  'digit_' + str(j) in label:

                return int(j+36)

#creating the Extra column with encoded values for all devanagari fonts

data['encodedLabels'] = data['character'].apply(convert_to_Number)
np.sort(data['encodedLabels'].unique())
#One hot encoding function : as the y leabels are in the fomrs of characters, one hot encoding is required to convert them from character to vector

def one_hot_encode(vec, vals=46):

    '''

    For use to one-hot encode the 46- possible labels of devanagari fonts

    '''

    n = len(vec)

    out = np.zeros((n, vals))

    out[range(n), vec] = 1

    return out



#creating a dataframe for ONe hot encoded values

y_encoded = pd.DataFrame(one_hot_encode(np.hstack([label for label in data['encodedLabels']]),46))
data.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop(['character','encodedLabels'],axis=1), y_encoded, test_size=0.3, random_state=42)
'''def reshape_Data(data,length):

    x = []

    for i in range(0,len(data)):

        x.append(data.iloc[i].reshape(32,32)) # sending data in normalised form  

    return x

'''
#len(np.array(X_train).reshape(len(X_train),32,32,1))



#temp = (X_train.values/255.0)
#temp[0:2]
#creating a class for batches

class Next_Batch():

    #initialising

    def __init__(self):

        self.i = 0

        self.train_image = None

        self.test_image = None

        self.train_label = None

        self.test_label = None

#---------------------------------------

    def setUpImage(self):

        train_len = len(X_train)

        test_len = len(X_test)

        #self.train_image = X_train.values.reshape(train_len,32,32,1)/255.0

        self.train_image = X_train.values/255.0

        #reshape_Data(np.vstack(X_train))#.reshape(train_len,1,32,32).transpose(0,2,3,1)

        #self.train_image = self.train_image.reshape(len(self.train_image),1,32,32)

        #self.test_image = X_test.values.reshape(test_len,32,32)/255.0#.reshape(train_len,1,32,32).transpose(0,2,3,1)/255

        self.test_image = X_test.values/255.0

        #self.test_image = self.test_image.reshape(len(self.test_image),1, 32,32)

        self.train_label = y_train

        self.test_label = y_test

#---------------------------------------    

    def next_Batch(self, batch_size):

        x = self.train_image[self.i : self.i + batch_size] # grabbing the images till the batch size

        y = self.train_label[self.i : self.i + batch_size] #grabbing the labels till the batch size

        self.i = int(self.i + batch_size) % len(self.train_image)

        return (x,y)

    

    
#creating object of class and initialising th variables 

nb = Next_Batch()

#setting up the images

nb.setUpImage()
#Plotting some data

plt.figure(figsize=(14,10))

a,b = 8,8

temp = data.drop(['character','encodedLabels'],axis=1)

for i in range(0,(a*b)):

    plt.subplot(a,b,i+1)

    plt.imshow(temp.iloc[i].values.reshape(32,32))

    plt.title(data['character'].iloc[i])

    #plt.title(X_train.iloc[i].get_index)

    #plt.tight_layout()

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline
# Weight initialisation 

def init_weight(shape):

    init_weights = tf.truncated_normal(shape=shape, stddev=0.1)

    return tf.Variable(init_weights)
# Bias initialisation

def init_bais(shape):

    init_biases = tf.constant(0.1,shape=shape,dtype=tf.float32)

    return tf.Variable(init_biases)
#convolution function

def conv2d(x, W):

    '''

    x --> image data as tensor [Batch, ht, wd, color channel]

    W --> Filter [filter ht, filter wt, input cannel, output channel]

    '''

    return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME')
#Pooling Function

def maxPoolling(x):

    return tf.nn.max_pool(x,ksize=[1,1,1,1],strides=[1,1,1,1],padding="SAME")
# Convolution Layer

def convolution_layer(input_x, shape):

    W = init_weight(shape)

    b = init_bais([shape[3]])

    return tf.nn.relu(conv2d(input_x,W) + b)
#Output fully connected function

def normalFullyConnected(input_x, size):

    input_size = int(input_x.get_shape()[1])

    W = init_weight([input_size,size])

    b = init_bais([size])

    return tf.matmul(input_x,W) + b
#creating a Place Holder

X = tf.placeholder(dtype=tf.float32,shape=[None,1024])

y_true = tf.placeholder(dtype=tf.float32,shape=[None,46])
X_image = tf.reshape(X,[-1,32,32,1])
# 1st convolution layer

#filter design [filter ht , filter wd, in cannel, out channel]

conv1_layer = convolution_layer(X_image,[6,6,1,32])

maxPool1 = maxPoolling(conv1_layer)
# 2nd Convolution layer

#filter design [filter ht , filter wd, in cannel, out channel]

conv2_layer = convolution_layer(maxPool1,[6,6,32,64])

maxPool2 = maxPoolling(conv2_layer)
flatData = tf.reshape(maxPool2,[-1,8*8*64])

full_layer = tf.nn.relu(normalFullyConnected(flatData,1024))
#drop out layer

hold_prob = tf.placeholder(dtype=tf.float32)

full_one_dropout = tf.nn.dropout(full_layer,keep_prob=hold_prob)
y_pred = normalFullyConnected(full_one_dropout,46)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
optim = tf.train.AdamOptimizer(learning_rate=0.001)
train = optim.minimize(cross_entropy)
init = tf.global_variables_initializer()
#t1, t2 = nb.next_Batch(2)
#with tf.Session() as sess:

 #   print(sess.run(tf.argmax(t2,1)))
#nb.test_image[0:100]
steps = 2000

with tf.Session() as sess:

    sess.run(init)

    for i in range(steps):

        batch_x , batch_y = nb.next_Batch(300)

        print(batch_x.shape)

        print(batch_y.shape)

        sess.run(train,feed_dict={X:batch_x,y_true:batch_y,hold_prob:0.5})

        #print('training done')

        # PRINT OUT A MESSAGE EVERY 100 STEPS

        if i%100 == 0:

            print('Currently on step {}'.format(i))

            print('Accuracy is:')

            # Test the Train Model

            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            #print('matching done')

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            #print(sess.run(acc,feed_dict={X:batch_x,y_true:batch_y,hold_prob:1.0}))

            print(sess.run(acc,feed_dict={X:nb.test_image[0:200],y_true:nb.test_label[0:200],hold_prob:1.0}))

           
#nb.test_image.shape
# on GPU



#Processing on GPU

'''

#sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

sess = tf.Session(config=config)



steps = 5000

sess.run(init)

for step in range(steps):

    batch_x , batch_y = nb.next_Batch(100)



    sess.run(train, feed_dict={X:batch_x,y_true:batch_y,hold_prob:0.5})

    # PRINT OUT A MESSAGE EVERY 100 STEPS

    #print('started step {}'.format(step))

    if step%100 == 0:

        print('Currently on step {}'.format(step))

        print('Accuracy is:')

        matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

        acc = tf.reduce_mean(tf.cast(matches,dtype=tf.float32))



        print(sess.run(acc,feed_dict = {X : nb.test_image, y_true : nb.test_label,hold_prob:1.0}))'''

    
#nb.test_image[0:-1]#