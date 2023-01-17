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


import cv2

import numpy as np

import os

from random import shuffle



TRAIN_DIR = "../input/train/train"

TEST_DIR = "../input/test/test"

IMG_SIZE = 50

LR = 1e-3

MODEL_NAME = 'dogsvscats1-{}-{}.model'.format(LR,'2conv-basic-video')
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
def label_img(img):

    word_label = img.split('.')[-3]

    print(word_label)

    if word_label == 'cat': return [1,0]

    elif word_label == 'dog': return [0,1]
def create_train_data():

    training_data = []

    for img in os.listdir(TRAIN_DIR):

        path = os.path.join(TRAIN_DIR,img)

        print(path)

        label = label_img(img)

        print(label)

        #path = os.path.join(TRAIN_DIR,img)

        #print(path)

        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))

        training_data.append([np.array(img),np.array(label)])

    shuffle(training_data)

    np.save('train_data.npy', training_data)

    return training_data

   

    
print(TRAIN_DIR)

print(TEST_DIR)

#os.listdir(TRAIN_DIR)

#os.listdir(TEST_DIR)
def process_test_data():

    testing_data = []

    for img in os.listdir(TEST_DIR):

        path = os.path.join(TEST_DIR,img)

        img_num = img.split('.')[0]

        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))

        testing_data.append([np.array(img),img_num])

    np.save('test_data.npy', testing_data)  

    return testing_data



    
train_data = create_train_data()

# if you already have train data you could say 

#train_data = np.load('train_data.npy')

import tflearn

from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.layers.core import input_data, dropout, fully_connected

from tflearn.layers.estimator import regression



import tensorflow as tf

tf.reset_default_graph()



convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')



convnet = conv_2d(convnet, 32, 2, activation='relu')

convnet = max_pool_2d(convnet, 2)



convnet = conv_2d(convnet, 64, 2, activation='relu')

convnet = max_pool_2d(convnet, 2)



convnet = conv_2d(convnet, 32, 2, activation='relu')

convnet = max_pool_2d(convnet, 2)



convnet = conv_2d(convnet, 64, 2, activation='relu')

convnet = max_pool_2d(convnet, 2)



convnet = conv_2d(convnet, 32, 2, activation='relu')

convnet = max_pool_2d(convnet, 2)



convnet = conv_2d(convnet, 64, 2, activation='relu')

convnet = max_pool_2d(convnet, 2)



convnet = fully_connected(convnet, 1024, activation='relu')

convnet = dropout(convnet, 0.8)



convnet = fully_connected(convnet, 2, activation='softmax')

convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')



model = tflearn.DNN(convnet, tensorboard_dir='log')

os.path.exists('{}.meta'.format(MODEL_NAME))
if(os.path.exists('{}.meta'.format(MODEL_NAME))):

    model.load(MODEL_NAME)

    print('model loaded !')
train = train_data[:-500]

test = train_data[-500:]
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)

Y = np.array([i[1] for i in train])



test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)

test_y = np.array([i[1] for i in test])
model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 

    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
#tensorboard --logdir=foo:D:\DogsVsCats\log
model.save(MODEL_NAME)
import matplotlib.pyplot as plt



test_data = process_test_data()



fig = plt.figure()



for num,data in enumerate(test_data[:12]):

    # cat = [1,0]

    #dog = [0,1]

    img_num = data[1]

    img_data = data[0]

    

    y = fig.add_subplot(3,4,num+1)

    orig = img_data

    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)

    

    model_out = model.predict([data])[0]

    

    if np.argmax(model_out) == 1: str_label = 'Dog'

    else: str_label = 'Cat'

    

    y.imshow(orig, cmap='gray')

    

    plt.title(str_label)

    y.axes.get_xaxis().set_visible(False)

    y.axes.get_yaxis().set_visible(False)

plt.show()
