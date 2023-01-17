# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

 

#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



#model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder



#preprocess.

from keras.preprocessing.image import ImageDataGenerator



#dl libraraies

from keras import backend as K

from keras import regularizers

from keras.models import Sequential

from keras.models import Model

from keras.layers import Dense

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical

from keras.callbacks import ReduceLROnPlateau



# specifically for cnn

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.layers import InputLayer

 

import tensorflow as tf

import random as rn



# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.

import cv2                  

import numpy as np  

from tqdm import tqdm

import os                   

from random import shuffle  

from zipfile import ZipFile

from PIL import Image

import keras.preprocessing.image as img

from keras.applications.resnet50 import ResNet50
os.listdir('../input/pothole-detection-small/My Dataset/train')
def make_train_data(label,DIR):

    for img in tqdm(os.listdir(DIR)):

        path = os.path.join(DIR,img)

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

    

        X.append(np.array(img))

        Z.append(str(label))
X=[]

Z=[]

IMG_SIZE=256

Plain='../input/pothole-detection-small/My Dataset/train/Plain'

Pothole='../input/pothole-detection-small/My Dataset/train/Pothole'



make_train_data('Plain',Plain)

make_train_data('Pothole',Pothole)
fig,ax=plt.subplots(2,5)

plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)

fig.set_size_inches(10,10)



for i in range(2):

    for j in range (5):

        l=rn.randint(0,len(Z))

        ax[i,j].imshow(X[l][:])

        ax[i,j].set_title(Z[l])

        ax[i,j].set_aspect('equal')
le=LabelEncoder()

Y=le.fit_transform(Z)

Y=to_categorical(Y,2)

print(Y)

X=np.array(X)



x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1337)



np.random.seed(42)

rn.seed(42)
base_model=ResNet50(include_top=False, weights='imagenet',input_shape=(256,256,3), pooling='max')

base_model.summary()
model=Sequential()

model.add(base_model)

model.add(Dropout(0.20))

model.add(Dense(2048,activation='relu'))

model.add(Dense(1024,activation='relu'))

model.add(Dense(512,activation='relu'))

model.add(Dense(2,activation='softmax'))
epochs=50

batch_size=128

red_lr=ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=0.0001, patience=2, verbose=1)

base_model.trainable=True # setting the VGG model to be trainable.

model.compile(optimizer=Adam(lr=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
History = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test,y_test))
model.save('../working/model.h5')

plt.plot(History.history['accuracy'])

plt.plot(History.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
plt.plot(History.history['loss'])

plt.plot(History.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
path = "/kaggle/input/pothole-detection-small/My Dataset/test/Pothole"

files = os.listdir(path)

files
for i in tqdm(files):

    pth = os.path.join(path,i)

    X = cv2.imread(pth,cv2.IMREAD_COLOR)

    X = cv2.resize(X,(256,256))

    plt.figure()

    plt.imshow(X[:,:,::-1]) 

    plt.show()  



    X = np.array(X)

    X = np.expand_dims(X, axis=0)



    y_pred = np.round(model.predict(X))

    if y_pred[0][0] == 1:

        print("Plain Road")

    else:

        print("Pothole Road")
path = "/kaggle/input/pothole-detection-small/My Dataset/test/Plain"

files = os.listdir(path)

files
for i in tqdm(files):

    pth = os.path.join(path,i)

    X = cv2.imread(pth,cv2.IMREAD_COLOR)

    X = cv2.resize(X,(256,256))

    plt.figure()

    plt.imshow(X[:,:,::-1]) 

    plt.show()  





    X = np.array(X)

    X = np.expand_dims(X, axis=0)



    y_pred = np.round(model.predict(X))

    if y_pred[0][0] == 1:

        print("Plain Road")

    else:

        print("Pothole Road")