# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import scipy as sp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

basePath = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'

# Any results you write to the current directory are saved as output.
trainInfo = pd.read_csv(basePath+'stage_1_train.csv')

print(trainInfo.head(10))
splitData = trainInfo['ID'].str.split('_', expand = True)

trainInfo['class'] = splitData[2]

trainInfo['fileName'] = splitData[0] + '_' + splitData[1]

trainInfo = trainInfo.drop(columns=['ID'],axis=1)

del splitData

print(trainInfo.head(10))
pivot_trainInfo = trainInfo[['Label', 'fileName', 'class']].drop_duplicates().pivot_table(index = 'fileName',columns=['class'], values='Label')

pivot_trainInfo = pd.DataFrame(pivot_trainInfo.to_records())

print(pivot_trainInfo.head(10))
import matplotlib.image as pltimg

import pydicom



fig = plt.figure(figsize = (20,10))

rows = 5

columns = 5

trainImages = os.listdir(basePath + 'stage_1_train_images')

for i in range(rows*columns):

    ds = pydicom.dcmread(basePath + 'stage_1_train_images/' + trainImages[i*100+1])

    fig.add_subplot(rows, columns, i+1)

    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

    fig.add_subplot    

        
colsToPlot = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']

rows = 5

columns = 5

for i_col in colsToPlot:

    fig = plt.figure(figsize = (20,10))

    trainImages = list(pivot_trainInfo.loc[pivot_trainInfo[i_col]==1,'fileName'])

    plt.title(i_col + ' Images')

    for i in range(rows*columns):

        ds = pydicom.dcmread(basePath + 'stage_1_train_images/' + trainImages[i*100+1] +'.dcm')

        fig.add_subplot(rows, columns, i+1)

        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)        

        fig.add_subplot    
for i_col in colsToPlot:

    plt.figure()

    ax = sns.countplot(pivot_trainInfo[i_col])

    ax.set_title(i_col + ' class count')
#dropping of corrupted image from dataset

pivot_trainInfo = pivot_trainInfo.drop(list(pivot_trainInfo['fileName']).index('ID_6431af929'))
import keras

from keras.layers import Dense, Activation,Dropout,Conv2D,MaxPooling2D,Flatten,Input,BatchNormalization,AveragePooling2D,LeakyReLU,ZeroPadding2D,Add,GlobalAveragePooling2D

from keras.models import Sequential, Model

from keras.initializers import glorot_uniform

from keras import optimizers

from keras.applications.resnet import ResNet50

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import cv2

import gc



pivot_trainInfo = pivot_trainInfo.sample(frac=1).reset_index(drop=True)

train_df,val_df = train_test_split(pivot_trainInfo,test_size = 0.03, random_state = 42)

batch_size = 64
y_train = train_df[['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]

y_val = val_df[['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]

train_files = list(train_df['fileName'])

gc.collect()

def scaleAndconvertImage(ds,windowLength,windowWidth):

    image = ds.pixel_array

    # Set outside-of-scan pixels to 1

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)

    intercept = ds.RescaleIntercept

    slope = ds.RescaleSlope    

    image = image * slope + intercept

    #min_value = windowLength - (windowWidth/2)

    #max_value = windowLength + (windowWidth/2)     

    #image[image <= min_value] = 0

    #image[image > max_value] = 255   

    #image[(image > min_value) & (image <= max_value)] = \

     #   ((image[(image > min_value) & (image <= max_value)] - 

     #     (windowLength - 0.5)) / (windowWidth - 1) + 0.5) * (255 - 0) + 0  

    U=1.0 

    eps=(1.0 / 255.0)

    ue = log((U / eps) - 1.0)

    W = (2 / windowWidth) * ue

    b = ((-2 * windowLength) / windowWidth) * ue

    z = W * image + b

    image = U / (1 + np.power(np.e, -1.0 * z))    

    x_max = image.max()

    x_min = image.min()

    if x_max != x_min:

        image = (image - x_min) / (x_max - x_min)

        return image

    del x_max,x_min,slope,intercept,ds

    return np.zeros(image.shape) 

    



def get_pixels_hu(ds):

    brain_img = scaleAndconvertImage(ds,40,80)

    subdural_img = scaleAndconvertImage(ds,80,200)

    bone_img = scaleAndconvertImage(ds,600,2000)

    image = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))

    image[:, :, 0] = brain_img

    image[:, :, 1] = subdural_img

    image[:, :, 2] = bone_img   

    del brain_img,subdural_img,bone_img

    return image    



def readDCMFile(fileName):

    ds = pydicom.read_file(fileName) # read dicom image

    #img = ds.pixel_array # get image array

    img = get_pixels_hu(ds)

    img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA) 

    return img



def generateImageData(train_files,y_train):

    numBatches = int(np.ceil(len(train_files)/batch_size))

    while True:

        x_batch_data = []

        y_batch_data = []

        for i in range(numBatches):

            batchFiles = train_files[i*batch_size : (i+1)*batch_size]

            x_batch_data = np.array([readDCMFile(basePath + 'stage_1_train_images/' + i_f +'.dcm') for i_f in batchFiles])

            y_batch_data = y_train[i*batch_size : (i+1)*batch_size]

            #x_batch_data = np.reshape(x_batch_data,(x_batch_data.shape[0],x_batch_data.shape[1],x_batch_data.shape[2],1))            

            yield x_batch_data,y_batch_data

            x_batch_data = []

            y_batch_data = []                  

            

def generateTestImageData(test_files):

    numBatches = int(np.ceil(len(test_files)/batch_size))

    while True:

        x_batch_data = []        

        for i in range(numBatches):

            batchFiles = test_files[i*batch_size : (i+1)*batch_size]

            x_batch_data = np.array([readDCMFile(basePath + 'stage_1_test_images/' + i_f +'.dcm') for i_f in batchFiles])

            #x_batch_data = np.reshape(x_batch_data,(x_batch_data.shape[0],x_batch_data.shape[1],x_batch_data.shape[2],1))

            yield x_batch_data            

            x_batch_data = []
for i_col in colsToPlot:

    fig = plt.figure(figsize = (20,10))

    trainImages = list(pivot_trainInfo.loc[pivot_trainInfo[i_col]==1,'fileName'])

    plt.title(i_col + ' Images')

    for i in range(rows*columns):

        img = readDCMFile(basePath + 'stage_1_train_images/' + trainImages[i*100+1] +'.dcm')

        fig.add_subplot(rows, columns, i+1)

        plt.imshow(img,cmap=plt.cm.bone)        

        fig.add_subplot    

        del img
dataGenerator = generateImageData(train_files,train_df[colsToPlot])

val_files = list(val_df['fileName'])
x_val = np.array([readDCMFile(basePath + 'stage_1_train_images/' + i_f +'.dcm') for i_f in tqdm(val_files)])

y_val = val_df[colsToPlot]
# loss function definition courtesy https://www.kaggle.com/akensert/resnet50-keras-baseline-model

from keras import backend as K

def logloss(y_true,y_pred):      

    eps = K.epsilon()

    

    class_weights = np.array([2., 1., 1., 1., 1., 1.])

    

    y_pred = K.clip(y_pred, eps, 1.0-eps)



    #compute logloss function (vectorised)  

    out = -( y_true *K.log(y_pred)*class_weights

            + (1.0 - y_true) * K.log(1.0 - y_pred)*class_weights)

    return K.mean(out, axis=-1)



def _normalized_weighted_average(arr, weights=None):

    """

    A simple Keras implementation that mimics that of 

    numpy.average(), specifically for the this competition

    """

    

    if weights is not None:

        scl = K.sum(weights)

        weights = K.expand_dims(weights, axis=1)

        return K.sum(K.dot(arr, weights), axis=1) / scl

    return K.mean(arr, axis=1)



def weighted_loss(y_true, y_pred):

    """

    Will be used as the metric in model.compile()

    ---------------------------------------------

    

    Similar to the custom loss function 'weighted_log_loss()' above

    but with normalized weights, which should be very similar 

    to the official competition metric:

        https://www.kaggle.com/kambarakun/lb-probe-weights-n-of-positives-scoring

    and hence:

        sklearn.metrics.log_loss with sample weights

    """      

    

    eps = K.epsilon()

    

    class_weights = K.variable([2., 1., 1., 1., 1., 1.])

    

    y_pred = K.clip(y_pred, eps, 1.0-eps)



    loss = -(y_true*K.log(y_pred)

            + (1.0 - y_true) * K.log(1.0 - y_pred))

    

    loss_samples = _normalized_weighted_average(loss,class_weights)

    

    return K.mean(loss_samples)
def convolutionBlock(X,f,filters,stage,block,s):

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    X_shortcut = X

    F1,F2,F3 = filters

    X = Conv2D(filters = F1, kernel_size = (1,1),strides = s, padding = 'valid',name = conv_name_base + '2a',

               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2a')(X)

    X = Activation('relu')(X)

    

    X = Conv2D(filters = F2, kernel_size = (f,f),strides = 1, padding = 'same',name = conv_name_base + '2b',

               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2b')(X)

    X = Activation('relu')(X)

    

    X = Conv2D(filters = F3, kernel_size = (1,1),strides = 1, padding = 'valid',name = conv_name_base + '2c',

               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2c')(X)



    X_shortcut = Conv2D(filters = F3, kernel_size = (1,1),strides = s, padding = 'valid',name = conv_name_base + '1',

               kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'1')(X_shortcut)

    

    X = Add()([X,X_shortcut])

    X = Activation('relu')(X)

    

    return X



def identityBlock(X,f,filters,stage,block):

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    X_shortcut = X

    F1,F2,F3 = filters

    X = Conv2D(filters = F1, kernel_size = (1,1),strides = 1, padding = 'valid',name = conv_name_base + '2a',

               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2a')(X)

    X = Activation('relu')(X)

    

    X = Conv2D(filters = F2, kernel_size = (f,f),strides = 1, padding = 'same',name = conv_name_base + '2b',

               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2b')(X)

    X = Activation('relu')(X)

    

    X = Conv2D(filters = F3, kernel_size = (1,1),strides = 1, padding = 'valid',name = conv_name_base + '2c',

               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2c')(X)

    

    X = Add()([X,X_shortcut])

    X = Activation('relu')(X)

    

    return X
input_img = Input((64,64,3))

#X = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), name="initial_conv2d")(input_img)

#X = BatchNormalization(axis=3, name='initial_bn')(X)

#X = Activation('relu', name='initial_relu')(X)

X = ZeroPadding2D((3, 3))(input_img)



# Stage 1

X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)

X = BatchNormalization(axis=3, name='bn_conv1')(X)

X = Activation('relu')(X)

X = MaxPooling2D((3, 3), strides=(2, 2))(X)



# Stage 2

X = convolutionBlock(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)

X = identityBlock(X, 3, [64, 64, 256], stage=2, block='b')

X = identityBlock(X, 3, [64, 64, 256], stage=2, block='c')



# Stage 3 (≈4 lines)

X = convolutionBlock(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)

X = identityBlock(X, 3, [128, 128, 512], stage=3, block='b')

X = identityBlock(X, 3, [128, 128, 512], stage=3, block='c')

X = identityBlock(X, 3, [128, 128, 512], stage=3, block='d')



# Stage 4 (≈4 lines)

X = convolutionBlock(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)

X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='b')

X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='c')

X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='d')

X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='e')

X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='f')



# Stage 5 (≈4 lines)

X = convolutionBlock(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)

X = identityBlock(X, 3, [512, 512, 2048], stage=5, block='b')

X = identityBlock(X, 3, [512, 512, 2048], stage=5, block='c')





# AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"

X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

# output layer

X = Flatten()(X)

out = Dense(6,name='fc' + str(6),activation='sigmoid')(X)
model_conv = Model(inputs = input_img, outputs = out)

#model_conv.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

model_conv.compile(optimizer='Adam',loss = logloss,metrics=[weighted_loss])

model_conv.summary()

history_conv = model_conv.fit_generator(dataGenerator,steps_per_epoch=500, epochs=20,validation_data = (x_val,y_val),verbose = True)
testInfo = pd.read_csv(basePath+'stage_1_sample_submission.csv')

splitData = testInfo['ID'].str.split('_', expand = True)

testInfo['class'] = splitData[2]

testInfo['fileName'] = splitData[0] + '_' + splitData[1]

testInfo = testInfo.drop(columns=['ID'],axis=1)

del splitData

pivot_testInfo = testInfo[['fileName', 'class','Label']].drop_duplicates().pivot_table(index = 'fileName',columns=['class'], values='Label')

pivot_testInfo = pd.DataFrame(pivot_testInfo.to_records())

test_files = list(pivot_testInfo['fileName'])

testDataGenerator = generateTestImageData(test_files)

temp_pred = model_conv.predict_generator(testDataGenerator,steps = pivot_testInfo.shape[0]/batch_size,verbose = True)
temp_pred.shape
submission_df = pivot_testInfo

submission_df['any'] = temp_pred[:,0]

submission_df['epidural'] = temp_pred[:,1]

submission_df['intraparenchymal'] = temp_pred[:,2]

submission_df['intraventricular'] = temp_pred[:,3]

submission_df['subarachnoid'] = temp_pred[:,4]

submission_df['subdural'] = temp_pred[:,5]
submission_df = submission_df.melt(id_vars=['fileName'])

submission_df['ID'] = submission_df.fileName + '_' + submission_df.variable

submission_df['Label'] = submission_df['value']

print(submission_df.head(20))
submission_df = submission_df.drop(['fileName','variable','value'],axis = 1)

print(submission_df.head(20))
submission_df.to_csv('sample_submission.csv', index=False)