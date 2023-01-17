



import keras

import cv2



import random

import matplotlib.pyplot as plt

import numpy as np

import os,glob

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.keras import backend as K



from tensorflow.keras.models import Model

from tensorflow.keras.layers import *

LABEL_DICT = {"0_normal":"0","1_black_smut":"1","2_peeled":"2"}



import numpy as np

import cv2

from matplotlib import pyplot as plt

import glob,os

import random

from sklearn.utils import shuffle

from tqdm import tqdm

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from keras.preprocessing.image import ImageDataGenerator

from keras.applications.densenet import DenseNet121

from keras.layers import Dense, GlobalAveragePooling2D

from keras import backend as K

import os

# os.environ["CUDA_VISIBLE_DEVICES"]="3"

import sys

import cv2,glob

import time

from PIL import Image

from tqdm import tqdm_notebook as tqdm

from keras.activations import relu

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau,ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.regularizers import l1,l2

from keras import regularizers

from keras.models import Model,load_model

from keras.layers import *

from sklearn.utils import shuffle
len(os.listdir('../input/cell-images-for-detecting-malaria/cell_images/Parasitized'))
import glob

from keras.utils import to_categorical

w,h = 100,100

def getData(impath):

    x_data,y_data = [],[]

    labels = {'Parasitized':0,'Uninfected':1}

    for im in tqdm(impath):

        if im.endswith('.png'):

            img = cv2.imread(im)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (w,h))/255.

            label = im.split('/')[-2]

            if label in labels:

                x_data.append(img)

                y_data.append(labels[label])

            else:

                print(label)

                continue

    x_data = np.asarray(x_data)

    y_data = to_categorical(y_data, num_classes=2)

    return x_data,y_data
impath_train = glob.glob('../input/cell-images-for-detecting-malaria/cell_images/*/*')



X_data,Y_data = getData(impath_train)

print(X_data.shape,Y_data.shape)
impath = '../input/cell-images-for-detecting-malaria/cell_images/Parasitized/'

data, labels = [],[]

for i in os.listdir(impath):

    try:

        image = cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/Parasitized/"+i)

        image_from_array= Image.fromarray(image , "RGB")

        size_image =image_from_array.resize((100,100))

        #resize45=size_image.rotate(15)

        #resize75 = size_image.rotate(25)

        #blur =cv2.blur(np.array(size_image),(10,10))

        data.append(np.array(size_image))

        labels.append(0)

        #labels.append(0)

        #labels.append(0)

        #labels.append(0)

        

    except AttributeError:

        print("")

Uninfected = os.listdir("../input/cell-images-for-detecting-malaria/cell_images/Uninfected/")

for b in Uninfected:

    try :

        image = cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/Uninfected/"+b)

        array_image=Image.fromarray(image,"RGB")

        size_image=array_image.resize((100,100))

        resize45= size_image.rotate(15)

        resize75 = size_image.rotate(25)

        #blur =cv2.blur(np.array(size_image),(10,10))

        data.append(np.array(size_image))

        #data.append(np.array(resize45))

        #data.append(np.array(resize75))

        #data.append(np.array(blur))

        #labels.append(1)

        #labels.append(1)

        #labels.append(1)

        labels.append(1)

    except AttributeError:

        print("")
Cells =np.array(data)

labels =np.array(labels)

labels =keras.utils.to_categorical(labels)

print(Cells.shape, labels.shape)
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(Cells, labels, test_size=0.3, random_state=0)
x_test.shape
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.3)
x_test.shape
def TNet(h=(100,100,3), alpha=1, classes=2):



    img_input = Input(shape = h)







    x = Conv2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.0002),activation="relu")(img_input)

    x = BatchNormalization()(x)





    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', activation="relu")(x)

    x = BatchNormalization()(x)

    x = Conv2D(int(32 * alpha), (1, 1), strides=(1, 1), padding='same',kernel_regularizer=regularizers.l2(0.0002),activation="relu" )(x)

    x = BatchNormalization()(x)





    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', activation="relu")(x)

    x = BatchNormalization()(x)

    x = Conv2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same',kernel_regularizer=regularizers.l2(0.0002) ,activation="relu")(x)

    x = BatchNormalization()(x)

 



    x = DepthwiseConv2D( (3, 3), strides=(1, 1), padding='same', activation="relu")(x)

    x = BatchNormalization()(x)

    x = Conv2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.0002),activation="relu")(x)

    x = BatchNormalization()(x)





    x = DepthwiseConv2D( (3, 3), strides=(2, 2), padding='same',activation="relu" )(x)

    x = BatchNormalization()(x)

    x = Conv2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same',kernel_regularizer=regularizers.l2(0.0002) ,activation="relu")(x)

    x = BatchNormalization()(x)





    x = DepthwiseConv2D( (3, 3), strides=(1, 1), padding='same',activation="relu" )(x)

    x = BatchNormalization()(x)

    x = Conv2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same',kernel_regularizer=regularizers.l2(0.0002) ,activation="relu")(x)

    x = BatchNormalization()(x)





    x = DepthwiseConv2D( (3, 3), strides=(2, 2), padding='same' ,activation="relu")(x)

    x = BatchNormalization()(x)

    x = Conv2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same',kernel_regularizer=regularizers.l2(0.0002) ,activation="relu")(x)

    x = BatchNormalization()(x)







    x = DepthwiseConv2D((3, 3), strides=(1,1), padding='same',activation="relu" )(x)

    x = BatchNormalization()(x)

    x = Conv2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.0002),activation="relu")(x)

    x = BatchNormalization()(x)





    

    x = DepthwiseConv2D( (3, 3), strides=(2,2), padding='same',activation="relu")(x)

    x = BatchNormalization()(x)

    x = Conv2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same',kernel_regularizer=regularizers.l2(0.002) ,activation="relu")(x)

    x = BatchNormalization()(x)



    x = DepthwiseConv2D((3, 3), strides=(1,1), padding='same',activation="relu" )(x)

    x = BatchNormalization()(x)

    x = Conv2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.002),activation="relu")(x)

    x = BatchNormalization()(x)

    



    

    x = DepthwiseConv2D( (3, 3), strides=(2,2), padding='same',activation="relu")(x)

    x = BatchNormalization()(x)

    x = Conv2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same',kernel_regularizer=regularizers.l2(0.002) ,activation="relu")(x)

    x = BatchNormalization()(x)



    x = DepthwiseConv2D((3, 3), strides=(1,1), padding='same',activation="relu" )(x)

    x = BatchNormalization()(x)

    x = Conv2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.002),activation="relu")(x)

    x = BatchNormalization()(x)

    

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.1)(x)

    out = Dense(classes,activation = 'sigmoid')(x)

#     out = Activation("sigmoid")(out)



    

    model = Model(img_input, out)



    return model

model = TNet()

model.summary()
from keras.optimizers import Adam, SGD

model.compile(loss='binary_crossentropy',

          optimizer=Adam(lr=0.001), 

          metrics=['acc'])
LR_Reduce_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto')

EarlyStop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=20)

model_ckpt = ModelCheckpoint('model_weights_9jul20.h5', monitor='val_loss', save_weights_only=True,save_best_only=True, period=1)
hist = model.fit(x_train, y_train,batch_size=16,validation_data=(x_val, y_val),epochs=80,

                 callbacks=[LR_Reduce_callback,EarlyStop,model_ckpt],verbose=1)





plt.plot(hist.history['loss'], label='train_loss')

plt.plot(hist.history['val_loss'], label='val_loss')

plt.legend()

plt.show()



plt.plot(hist.history['acc'], label='train_acc')

plt.plot(hist.history['val_acc'], label='val_acc')

plt.legend()

plt.show()
model.save('maleria_model.h5')
# x_test, y_test = np.array(x_test)/255,np.array(y_test)

score = model.evaluate(x_test, y_test, verbose=1)

print('Accuracy:',score[1])

print('Loss:',score[0])
from sklearn.metrics import classification_report,accuracy_score

from sklearn.metrics import confusion_matrix

labels = ['Parasitized','Uninfected']

y_pred = model.predict(x_test, verbose=1)

y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(np.argmax(y_test,axis=1), y_pred_bool))