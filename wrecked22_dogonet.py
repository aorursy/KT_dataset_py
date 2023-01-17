import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')

import tensorflow as tf

import random as rn

import itertools

import cv2                  

import numpy as np         

import os      

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical

from keras.layers import Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization             

from random import shuffle  

from zipfile import ZipFile

from PIL import Image

from tqdm import tqdm 
TRAIN_DIR = '../input/train'

TEST_DIR = '../input/test'

IMG_SIZE=100
def label_img(img):

    word_label = img.split('.')[0]

    return word_label
def create_train_data():

    training_data = []

    for img in tqdm(os.listdir(TRAIN_DIR)):

        label=label_img(img)

        path = os.path.join(TRAIN_DIR,img)

        img_num = img.split('.')[0]

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        training_data.append([np.array(img),str(label)])

        

    shuffle(training_data)

    return training_data
train_data=create_train_data()

train_data=np.array(train_data)

print(train_data.shape)

X= np.array([i[0] for i in train_data]).reshape(-1,IMG_SIZE,IMG_SIZE,3)

Y= np.array([i[1] for i in train_data])
fig,ax=plt.subplots(6,2)

fig.set_size_inches(15,15)

for i in range(6):

    for j in range (2):

        k=rn.randint(0,len(Y))

        ax[i,j].imshow(X[k])

        ax[i,j].set_title('Pet: '+Y[k])

        

plt.tight_layout()
sns.countplot(Y)

plt.title('Categories')
l =[]

for i in range(25000):

    if Y[i]=="dog":

                   l.append(1)

    elif Y[i]=="cat":

                   l.append(0)
X_train, X_test, y_train, y_test = train_test_split(X, l, shuffle=True, test_size =0.3, random_state = 32)
model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Conv2D(128, (3,3), input_shape=(100,100,3), activation = tf.nn.relu, padding = "valid"))

model.add(tf.keras.layers.MaxPool2D( pool_size = (3,3), strides = None ))



model.add(tf.keras.layers.Conv2D(128, (3,3), activation = tf.nn.relu, padding = "same"))

model.add(tf.keras.layers.MaxPool2D( pool_size = (3,3), strides = None ))



model.add(tf.keras.layers.Conv2D(128, (3,3), activation = tf.nn.relu, padding = "same"))

model.add(tf.keras.layers.MaxPool2D( pool_size = (3,3), strides = None ))



model.add(tf.keras.layers.Flatten())



model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))

model.add(tf.keras.layers.Dropout(0.25))



model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))

model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))
model.summary() #Architecture of Network
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"])
Model= model.fit(X_train, y_train,

                 validation_split=0.1,

                 epochs=13,

                 batch_size=32)
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
plt.plot(Model.history['acc'])

plt.plot(Model.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()
plt.plot(Model.history['val_loss'])

plt.plot(Model.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Test set'], loc='upper left')

plt.show()
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred,axis = 1) 

confusion_mtx = confusion_matrix(y_test, y_pred_classes) 

plot_confusion_matrix(confusion_mtx, classes = range(2)) 
def create_test_data():

    testing_data = []

    for img in tqdm(os.listdir(TEST_DIR)):

        path = os.path.join(TEST_DIR,img)

        img_num = img.split('.')[0]

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        testing_data.append([np.array(img)])

        

    shuffle(testing_data)

    return testing_data
test_data=create_test_data()

test_data=np.array(test_data)

print(test_data.shape)

test= np.array([i[0] for i in test_data]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
pred=model.predict(test)
imageid=[]

prob=[]

for i in range(12500):

    imageid.append(i+1)

    prob.append(pred[i,1])

   

d={'id':imageid,'label':prob}

ans=pd.DataFrame(d)

ans.to_csv('prediction.csv',index=False)