# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import cv2

from matplotlib import pyplot as plt

import gc

print(os.listdir("../input"))

gc.collect()



# Any results you write to the current directory are saved as output.
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os,cv2

from IPython.display import Image

from keras.preprocessing import image

from keras import optimizers

from keras import layers,models

from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt

import seaborn as sns

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16

import tensorflow as tf

import numpy as np

import time

import albumentations
os.getcwd()
train_y=pd.read_csv('../input/train.csv')

test_y=pd.read_csv('../input/sample_submission.csv')

train_y['has_cactus']=train_y['has_cactus'].astype(str)



train_y['has_cactus'][16500:].value_counts()

def load_data(x):

    train_dir="../input/train/train"

    test_dir="../input/test/test"

    train_x=[]

    test_x=[]

    filename=[]

    filenamey=[]

    for datefile in os.listdir('../input/train/train'):

                                

        img=cv2.imread(os.path.join(train_dir,datefile))

        filename.append(datefile)

        train_x.append(img)





    for datefile in os.listdir('../input/test/test'):

                                

        img=cv2.imread(os.path.join(test_dir,datefile))

        filenamey.append(datefile)

        test_x.append(img)

    

    

    label=pd.DataFrame({'id':filename})

    train_y=label.merge(x,how='left',on='id')



    train_Y=train_y['has_cactus']

    train_Y=np.array(pd.get_dummies(train_Y))

    

    trm_x=train_x.copy()

    trm_y=train_Y

    for r in range(8):

        for i in train_y[train_y['has_cactus']=='0'].index:

            img=augs()(image=train_x[i])["image"]

            trm_x.append(img)

            trm_y=np.append(trm_y,[train_Y[i]],axis=0)   

         

    for r in range(2):    

        for i in train_y[train_y['has_cactus']=='1'].index:

            img=augs()(image=train_x[i])["image"]

            trm_x.append(img)

            trm_y=np.append(trm_y,[train_Y[i]],axis=0)  

    return trm_x,trm_y,test_x,filenamey

def augs(p=0.5):

    return albumentations.Compose([

        albumentations.HorizontalFlip(),

        albumentations.VerticalFlip(),

        albumentations.RandomBrightness(),

    ], p=p)

trm_x,trm_y,test_x,filenamey=load_data(train_y)

sum(train_y['has_cactus']=='0')

len(trm_y)
np.sum(trm_y,axis=0)
trm_x=[i/255 for i in trm_x]

trm_x=np.array(trm_x)

test_x=[i/255 for i in test_x]

test=np.array(test_x)
trm_x.shape
keepprob=0.8

tf.reset_default_graph()





def max_pool_2x2(x):

    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')





x=tf.placeholder(tf.float32,[None,32,32,3],name='x_image')

y=tf.placeholder(tf.float32,[None,2],name='y')



W_conv1=tf.get_variable('W1',shape=[3,3,3,16],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

b_conv1=tf.get_variable('b1',shape=[16],initializer=tf.constant_initializer(0.1))



h_conv1=tf.nn.relu(tf.nn.conv2d(x,W_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)





W_conv12=tf.get_variable('W12',shape=[3,3,16,16],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

b_conv12=tf.get_variable('b12',shape=[16],initializer=tf.constant_initializer(0.1))



h_conv12=tf.nn.relu(tf.nn.conv2d(h_conv1,W_conv12,strides=[1,1,1,1],padding='SAME')+b_conv12)



h_pool1=max_pool_2x2(h_conv12)



W_conv2=tf.get_variable('W2_1',shape=[3,3,16,32],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

b_conv2=tf.get_variable('b2_1',shape=[32],initializer=tf.constant_initializer(0.1))



h_conv2=tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)



W_conv22=tf.get_variable('W2_2',shape=[3,3,32,32],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

b_conv22=tf.get_variable('b2_2',shape=[32],initializer=tf.constant_initializer(0.1))



h_conv22=tf.nn.relu(tf.nn.conv2d(h_conv2,W_conv22,strides=[1,1,1,1],padding='SAME')+b_conv22)

h_pool2=max_pool_2x2(h_conv22)



W_conv3=tf.get_variable('W3',shape=[3,3,32,64],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

b_conv3=tf.get_variable('b3',shape=[64],initializer=tf.constant_initializer(0.1))

h_conv3=tf.nn.relu(tf.nn.conv2d(h_pool2,W_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)



W_conv32=tf.get_variable('W32',shape=[3,3,64,64],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

b_conv32=tf.get_variable('b32',shape=[64],initializer=tf.constant_initializer(0.1))

h_conv32=tf.nn.relu(tf.nn.conv2d(h_conv3,W_conv32,strides=[1,1,1,1],padding='SAME')+b_conv32)



W_conv33=tf.get_variable('W33',shape=[3,3,64,64],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

b_conv33=tf.get_variable('b33',shape=[64],initializer=tf.constant_initializer(0.1))

h_conv33=tf.nn.relu(tf.nn.conv2d(h_conv32,W_conv33,strides=[1,1,1,1],padding='SAME')+b_conv33)



h_pool3=max_pool_2x2(h_conv33)





W_conv4=tf.get_variable('W4',shape=[3,3,64,128],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

b_conv4=tf.get_variable('b4',shape=[128],initializer=tf.constant_initializer(0.1))

h_conv4=tf.nn.relu(tf.nn.conv2d(h_pool3,W_conv4,strides=[1,1,1,1],padding='SAME')+b_conv4)



W_conv42=tf.get_variable('W42',shape=[3,3,128,128],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

b_conv42=tf.get_variable('b42',shape=[128],initializer=tf.constant_initializer(0.1))

h_conv42=tf.nn.relu(tf.nn.conv2d(h_conv4,W_conv42,strides=[1,1,1,1],padding='SAME')+b_conv42)





h_pool4=max_pool_2x2(h_conv42)









W_fc1=tf.get_variable("Wf1_2",shape=[2*2*128,512],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

b_fc1=tf.get_variable('bf2',shape=[512],initializer=tf.constant_initializer(0.1))

h_pool_flat=tf.reshape(h_pool4,[-1,2*2*128])

h_fc1=tf.nn.relu(tf.matmul(h_pool_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32,name='keep_prob')

h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=tf.get_variable('Wf3_2',shape=[512,256],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

b_fc2=tf.get_variable('bf3',shape=[256],initializer=tf.constant_initializer(0.1))

h_fc2=tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

h_fc2_drop=tf.nn.dropout(h_fc2,keep_prob)



W_fc3=tf.get_variable('Wf4_2',shape=[256,2],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

b_fc3=tf.get_variable('bf4',shape=[2],initializer=tf.constant_initializer(0.1))



y_conv=tf.matmul(h_fc2_drop,W_fc3)+b_fc3





cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_conv))

train_step=tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9).minimize(cross_entropy)

correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



tf.add_to_collection('y_conv',y_conv)

tf.add_to_collection('cross_entropy',cross_entropy)
mybatch=50

iterations=20



def train_model(mybatch,iterations,m,n,keepprob):

    

    init = tf.global_variables_initializer()

    

    #saver=tf.train.Saver(max_to_keep=1)



    with tf.Session() as sess:      

        sess.run(init)

        a=0.8

        indice=list(range(len(m)))

        np.random.shuffle(indice)

        m=m[indice]

        n=n[indice]

        x1=m[15000:]

        y1=n[15000:]

        x2=m[:15000]

        y2=n[:15000]

        for s in range(iterations):

            t0=time.time()

            myindice=list(range(len(x1)))

            np.random.shuffle(myindice)

           

            x_=x1[myindice]

            y_=y1[myindice]



            for i in range(len(x1)//mybatch):

                sess.run(train_step,feed_dict={x:x_[i*mybatch:i*mybatch+mybatch],y:y_[i*mybatch:i*mybatch+mybatch],keep_prob:keepprob})

      

            t1=time.time()

            print("traintime:%.2f s"%(t1-t0))

            print(sess.run(cross_entropy,feed_dict={x:x_[:1000],y:y_[:1000],keep_prob:1.0}))  

            

            #train_acc=sess.run(accuracy,feed_dict={x:x2,y:y2,keep_prob:1.0})

            t2=time.time()

            print("evaltime:%.2f s"%(t2-t0))

            #print("eval:%1.4f"%(train_acc))

            cross_val=sess.run(cross_entropy,feed_dict={x:x2,y:y2,keep_prob:1.0})

            print("evalcross:%1.4f"%(cross_val))

            if cross_val<a:

                #saver.save(sess,'basemode.ckpt')               

                a=cross_val

                pre_plate=sess.run(y_conv,feed_dict={x:test,keep_prob:1.0})

                predict=np.argmax(pre_plate,axis=1)

        return predict
predict=train_model(mybatch,iterations,trm_x,trm_y,keepprob)
sub=pd.DataFrame({"id":filenamey,'has_cactus':predict})

realsub=test_y[['id']]

sub=realsub.merge(sub,how='left',on='id')



sub.to_csv("sample_submission.csv",index=False)
predict