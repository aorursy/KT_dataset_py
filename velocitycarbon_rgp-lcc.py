# Import libraries

import os,cv2

import numpy as np

import matplotlib.pyplot as plt



from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split



from keras.utils import np_utils

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.optimizers import SGD,RMSprop,adam
from keras.models import Sequential,Model,load_model

from keras.optimizers import SGD

from keras.layers import BatchNormalization, Lambda, Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from keras.layers.merge import Concatenate

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

import numpy as np

import keras.backend as K
# First, look at everything.

from subprocess import check_output
PATH = os.getcwd()

# Define data path

data_path = '../input/onlyrgb/onlyrgb/'

data_dir_list = os.listdir(data_path)

print(check_output(["ls", "../input/onlyrgb/onlyrgb"]).decode("utf8"))

data_dir_list.sort()

data_dir_list
data_dir_list
labels_name={'L1':0,'L2':1,'L3':2,'L4':3}

labels_name['L1']
img_data_list=[]

labels_list = []
for dataset in data_dir_list:

    img_list=os.listdir(data_path+'/'+ dataset)

    print ('Loading the images of dataset-'+'{}\n'.format(dataset))

    label = labels_name[dataset]

    for img in img_list:

        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )

        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        input_img_resize=cv2.resize(input_img,(128,128))

        img_data_list.append(input_img_resize)

        labels_list.append(label)
img_data = np.array(img_data_list)

img_data = img_data.astype('float32')
print (img_data.shape)
#img_data_list
#labels_list
img_data = np.array(img_data_list)
print (img_data.shape)
labels = np.array(labels_list)
print(np.unique(labels,return_counts=True))
img_rows=120

img_cols=70

num_channel=3

num_epoch=20



# Define the number of classes

num_classes = 4
Y = np_utils.to_categorical(labels, num_classes)
#Shuffle the dataset

x,y = shuffle(img_data,Y, random_state=101)
# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=221)
input_shape=img_data[0].shape
input_shape
input_image = Input(shape=input_shape)
 # first top convolution layer

top_conv1 = Convolution2D(filters=48,kernel_size=(11,11),strides=(4,4),

                          input_shape=input_shape,activation='relu')(input_image)

top_conv1 = BatchNormalization()(top_conv1)

top_conv1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_conv1)



# second top convolution layer

# split feature map by half

top_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(top_conv1)

top_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(top_conv1)



top_top_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv2)

top_top_conv2 = BatchNormalization()(top_top_conv2)

top_top_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_top_conv2)



top_bot_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv2)

top_bot_conv2 = BatchNormalization()(top_bot_conv2)

top_bot_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_bot_conv2)



# third top convolution layer

# concat 2 feature map

top_conv3 = Concatenate()([top_top_conv2,top_bot_conv2])

top_conv3 = Convolution2D(filters=192,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_conv3)



# fourth top convolution layer

# split feature map by half

top_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(top_conv3)

top_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(top_conv3)



top_top_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv4)

top_bot_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv4)



# fifth top convolution layer

top_top_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv4)

top_top_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_top_conv5) 



top_bot_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv4)

top_bot_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_bot_conv5)



# ============================================= TOP BOTTOM ===================================================

# first bottom convolution layer

bottom_conv1 = Convolution2D(filters=48,kernel_size=(11,11),strides=(4,4),

                          input_shape=(227,227,3),activation='relu')(input_image)

bottom_conv1 = BatchNormalization()(bottom_conv1)

bottom_conv1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_conv1)



# second bottom convolution layer

# split feature map by half

bottom_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(bottom_conv1)

bottom_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(bottom_conv1)



bottom_top_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv2)

bottom_top_conv2 = BatchNormalization()(bottom_top_conv2)

bottom_top_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_top_conv2)



bottom_bot_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv2)

bottom_bot_conv2 = BatchNormalization()(bottom_bot_conv2)

bottom_bot_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_bot_conv2)



# third bottom convolution layer

# concat 2 feature map

bottom_conv3 = Concatenate()([bottom_top_conv2,bottom_bot_conv2])

bottom_conv3 = Convolution2D(filters=192,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_conv3)



# fourth bottom convolution layer

# split feature map by half

bottom_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(bottom_conv3)

bottom_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(bottom_conv3)



bottom_top_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv4)

bottom_bot_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv4)



# fifth bottom convolution layer

bottom_top_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv4)

bottom_top_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_top_conv5) 



bottom_bot_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv4)

bottom_bot_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_bot_conv5)



# ======================================== CONCATENATE TOP AND BOTTOM BRANCH =================================

conv_output = Concatenate()([top_top_conv5,top_bot_conv5,bottom_top_conv5,bottom_bot_conv5])



# Flatten

flatten = Flatten()(conv_output)



# Fully-connected layer

FC_1 = Dense(units=4096, activation='relu')(flatten)

FC_1 = Dropout(0.6)(FC_1)

FC_2 = Dense(units=4096, activation='relu')(FC_1)

FC_2 = Dropout(0.6)(FC_2)

output = Dense(units=num_classes, activation='softmax')(FC_2)



model = Model(inputs=input_image,outputs=output)

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

# sgd = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=True)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
num_epoch=3

num_epoch
# Training

hist = model.fit(X_train, y_train, batch_size=50, epochs=num_epoch, verbose=True, validation_data=(X_test, y_test))