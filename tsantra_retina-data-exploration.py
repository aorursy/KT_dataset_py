# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import os

import keras

from keras import layers

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.models import Model, load_model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from keras.initializers import glorot_uniform

import scipy.misc

from matplotlib.pyplot import imshow

from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)
def normalize_histograms(im): #normalizes the histogram of images

    im1=im.copy()

    for i in range(3):

        imi=im[:,:,i]

        #print(imi.shape)

        minval=np.min(imi)

        maxval=np.max(imi)

        #print(minval,maxval)

        imrange=maxval-minval

        im1[:,:,i]=(255/(imrange+0.0001)*(imi-minval)) # imi-minval will turn the color range between 0-imrange, and the scaleing will stretch the range between 0-255

    return im1
def read_and_process_image(filename,im_size):

        im=cv2.imread(filename) #read image from file 

        

        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) # convert 2 grayscale

        _,thresh = cv2.threshold(gray,10,255,cv2.THRESH_BINARY) # turn it into a binary image

        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # find contours

        if len(contours) != 0:

            #find the biggest area

            cnt = max(contours, key = cv2.contourArea)

                      

            #find the bounding rect

            x,y,w,h = cv2.boundingRect(cnt)                  



            crop = im[y:y+h,x:x+w]# crop image

            crop1=cv2.resize(crop,(im_size,im_size)) # resize to im_size X im_size size

            crop1=normalize_histograms(crop1)

            return crop1

        else:

            return( normalize_histograms(cv2.resize(im,(im_size,im_size))) )         
def prepare_data(files,labels_orig,im_size):

    images=[]

    labels=[]

    for i,f in enumerate(files):

        im=read_and_process_image(f,im_size)

        l=labels_orig[i]

        

        imb=im+0.05*im # brighter image

        

        imd=im-0.05*im #deemer image

        

        imlr= cv2.flip(im,0)

        imud= cv2.flip(im,1)

        

        imblr=cv2.flip(imb,0)

        imbud=cv2.flip(imb,1)

        

        imdlr=cv2.flip(imd,0)

        imdud=cv2.flip(imd,1)

        

        #add all the images an labels   

        images.append(im)

        labels.append(l)

        



        images.append(imb)

        labels.append(l)

        



        images.append(imd)

        labels.append(l)

        

  

        images.append(imlr)

        labels.append(l)

        

       

        images.append(imud)

        labels.append(l)

        

        images.append(imblr)

        labels.append(l)

        

        

        images.append(imbud)

        labels.append(l)

        

        images.append(imdlr)

        labels.append(l)

        

        

        images.append(imdud)

        labels.append(l)

        

        

    return(np.array(images),np.array(labels))

    
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#Create a list of image files and labels



files=[] #store the filenames here

labels=[] #store the labels here

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith('.png'):

            files.append(os.path.join(dirname, filename))

            l=np.zeros(4)

            if filename.startswith('NL'):

                l[0]=1

            elif filename.startswith('ca'):

                l[1]=1

            elif filename.startswith('Gl'):

                l[2]=1

            elif filename.startswith('Re'):

                l[3]=1

            labels.append(l)



        

print(len(labels),len(files))        



#Shuffle the files and labels

combined = list(zip(files, labels)) # combine the lists

np.random.shuffle(combined) # shuffle two lists together to keep order

files[:], labels[:] = zip(*combined) #unzip the shuffled lists

#print(files,labels)



# Train test devide (70:30)

index=int(len(files)*0.7)



#size of the images

im_size=128



# training data

files_train=files[:index]

labels_train=labels[:index]



X_train,Y_train=prepare_data(files_train,labels_train,im_size)

X_train=X_train/255

# test data

files_test=files[index:]

labels_test=labels[index:]



X_test,Y_test=prepare_data(files_test,labels_test,im_size)

X_test=X_test/255
index=19

print(X_test[index].shape)

#im1=normalize_histograms(X_test[index])

plt.imshow(X_test[index])

#plt.imshow(X_test[index])

print(Y_test[index], np.sum(X_test[index]),np.max(X_test[index]))

# GRADED FUNCTION: identity_block



def identity_block(X, f, filters, stage, block):

    """

    X -- input tensor 

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    

    Returns:

    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)

    """

    

    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    # Retrieve Filters

    F1, F2, F3 = filters

    

    # Save the input value. You'll need this later to add back to the main path. 

    X_shortcut = X

    

    # First component of main path

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)  



    

    # Second component of main path (≈3 lines)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    # Third component of main path (≈2 lines)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)

    X = Add()([X, X_shortcut])

    X = Activation('relu')(X)

    

    return X
def convolutional_block(X, f, filters, stage, block, s = 2):

    """    

    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    s -- Integer, specifying the stride to be used

    

    Returns:

    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)

    """

    

    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    # Retrieve Filters

    F1, F2, F3 = filters

    

    # Save the input value

    X_shortcut = X





    ##### MAIN PATH #####

    # First component of main path 

    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)





    # Second component of main path (≈3 lines)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    # Third component of main path (≈2 lines)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)



    ##### SHORTCUT PATH #### (≈2 lines)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)

    X = Add()([X, X_shortcut])

    X = Activation('relu')(X)



    

    return X
def ResNetS(input_shape = (128, 128, 3), classes = 4,filters=[5,5,10]):

    """

    Implementation of a simpler version (ResNet10 if you may) of the popular ResNet50:

    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3

    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    """

    # Define the input as a tensor with shape input_shape

    X_input = Input(input_shape)



    

    # Zero-Padding

    X = ZeroPadding2D((2, 2))(X_input)

    

    # Stage 1

    X = Conv2D(10, (5, 5), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)



    # Stage 2

    X = convolutional_block(X, f = 3, filters = filters, stage = 2, block='a', s = 1)

    X = identity_block(X, 3, filters, stage=2, block='b')

    X = identity_block(X, 3, filters, stage=2, block='c')



    # Stage 3 

    X = convolutional_block(X, f=3, filters=filters, stage=3, block='a', s=2)

    X = identity_block(X, 3, filters, stage=3, block='b')

    X = identity_block(X, 3, filters, stage=3, block='c')

    X = identity_block(X, 3, filters, stage=3, block='d')



    # Stage 4 

    X = convolutional_block(X, f=3, filters=filters, stage=4, block='a', s=2)

    X = identity_block(X, 3, filters, stage=4, block='b')

    X = identity_block(X, 3, filters, stage=4, block='c')

    X = identity_block(X, 3, filters, stage=4, block='d')

    X = identity_block(X, 3, filters, stage=4, block='e')

    X = identity_block(X, 3, filters, stage=4, block='f')

    

    # Stage 5 

    X = X = convolutional_block(X, f=3, filters=filters, stage=5, block='a', s=2)

    X = identity_block(X, 3, filters, stage=5, block='b')

    X = identity_block(X, 3, filters, stage=5, block='c')



    # AVGPOOL 

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)



    # output layer

    X = Flatten()(X)

    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0),kernel_regularizer=keras.regularizers.l1_l2(l1=0.5,l2=0.25))(X)

    X = BatchNormalization(name = 'output')(X)

    X = Activation('softmax')(X)

    

    # Create model

    model = Model(inputs = X_input, outputs = X, name='ResNetS')



    return model
def genCNN(input_shape = (128, 128, 3), classes = 4, filters=[16,32,64,128]):



    # Define the input as a tensor with shape input_shape

    X_input = Input(input_shape)



    

    # Zero-Padding

    X = ZeroPadding2D((2, 2))(X_input)

    

    # Stage 1

    X = Conv2D(filters[0], (5, 5), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)



    # Stage 2

    X = Conv2D(filters[1], (5, 5), strides = (2, 2), padding='same',name = 'conv2', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    

    # Stage 3

    X = Conv2D(filters[2], (5, 5), strides = (1, 1), padding='same',name = 'conv3', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = 'bn_conv3')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), padding='same',strides=(1, 1))(X)

   

    # Stage 4

    X = Conv2D(filters[3], (5, 5), padding='same',strides = (1, 1), name = 'conv4', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = 'bn_conv4')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), padding='same',strides=(1, 1))(X)

  

    #dense layer

    X = Flatten()(X)

    X = Dense(classes,name='fc_l2', kernel_initializer = glorot_uniform(seed=0),kernel_regularizer=keras.regularizers.l1_l2(l1=0.5,l2=0.5))(X)

    X = BatchNormalization(name = 'dense')(X)

    X = Activation('relu')(X)

    

    X = Dense(classes,name='fc_l3', kernel_initializer = glorot_uniform(seed=0),kernel_regularizer=keras.regularizers.l1_l2(l1=0.5,l2=0.5))(X)

    X = BatchNormalization(name = 'dense1')(X)

    X = Activation('relu')(X)

    

    # output layer

    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0),kernel_regularizer=keras.regularizers.l1_l2(l1=0.5,l2=0.5))(X)

    X = BatchNormalization(name = 'output')(X)

    X = Activation('softmax')(X)

    

    # Create model

    model = Model(inputs = X_input, outputs = X, name='ResNetS')



    return model
# Create and compile the ResNet models

resnet_s = ResNetS(input_shape = (128, 128, 3), classes = 4,filters=[16,16,32])

resnet_s.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the resnet model

print('Training resnet....')

history=resnet_s.fit(X_train, Y_train, validation_data=(X_test,Y_test),epochs = 20, batch_size = 32)

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

#print('Saving resnet...')

#resnet_s.save('resnet_s_200.h5')
preds1 = resnet_s.evaluate(X_test, Y_test)

print ("resnet Loss = " + str(preds1[0]))

print ("resnet Test Accuracy = " + str(preds1[1]))


#Create and compile the generic CNN models

gcnn=genCNN(input_shape = (128, 128, 3), classes = 4,filters=[16,16,32,32])

gcnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#train the general cnn model



print(' Training generic CNN..')

history=gcnn.fit(X_train, Y_train, epochs = 50, validation_data=(X_test,Y_test), batch_size = 32)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

#print('Saving resnet...')

#resnet_s.save('resnet_s_200.h5')

#print('Saving generic CNN...')

#gcnn.save('gcnn.h5')