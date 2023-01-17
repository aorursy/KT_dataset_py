# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from keras.datasets import cifar10

(X_train,y_train), (X_test,y_test)=cifar10.load_data()
print(X_train.shape,X_test.shape)

print(y_train.shape,y_test.shape)
l_grid,w_grid=15,15



fig,axes=plt.subplots(l_grid,w_grid,figsize=(25,25))



axes=axes.ravel() #used to flatten the image into 25*25

n_training=len(X_train)

for i in np.arange(0,l_grid*w_grid):

  index=np.random.randint(0,n_training)

  axes[i].imshow(X_train[index])

  axes[i].set_title(y_train[index])

  axes[i].axis('off')

plt.subplots_adjust(hspace=0.6)
X_train=X_train.astype('float32')

X_test=X_test.astype('float32')
import keras

y_train=keras.utils.to_categorical(y_train,10)

y_test=keras.utils.to_categorical(y_test,10)
X_train=X_train/255

X_test=X_test/255

X_train
Input_shape=X_train.shape[1:]

Input_shape
from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard
cnn_model= Sequential()

cnn_model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=Input_shape))

cnn_model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))

cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Dropout(0.25))



cnn_model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))

cnn_model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))

cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Dropout(0.5))



cnn_model.add(Flatten())



cnn_model.add(Dense(512,activation='relu'))

cnn_model.add(Dense(512,activation='relu'))

cnn_model.add(Dense(10,activation='softmax'))
cnn_model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.RMSprop(lr=0.001),metrics=['accuracy'])
cnn_model.summary()
history= cnn_model.fit(X_train,y_train, batch_size =32,epochs=5,shuffle=True)
evaluation= cnn_model.evaluate(X_test,y_test)

print("Test Accuracy: {}".format(evaluation[1]))
y_test=y_test.argmax(-1)
sample=pd.read_csv('/kaggle/input/cifar-10/sampleSubmission.csv')

sample.head()
y_tes=pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')

y_tes
from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()

y_tes['label']=le.fit_transform(y_tes['label'])

y_tes
predicted_class= cnn_model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_test,predicted_class)

plt.figure(figsize=(10,10))

sns.heatmap(cm,annot=True)
from sklearn.metrics import classification_report

print(classification_report(y_test,predicted_class))
predicted_class= le.inverse_transform(predicted_class)

labels=np.unique(predicted_class)

print(labels)

predicted_class

l,w=7,7

fig,axes=plt.subplots(l,w,figsize=(15,15))

y_test=le.inverse_transform(y_test)

axes=axes.ravel() #used to flatten the image

n_testing=len(X_test)

for i in np.arange(0,l*w):

    axes[i].imshow(X_test[i])

    axes[i].set_title("Prediction: {}\n True: {}".format(predicted_class[i],y_test[i]))

    axes[i].axis('off')

plt.subplots_adjust(wspace=1)
import os

directory= os.path.join(os.getcwd(),'saved_models')

if not os.path.isdir(directory):

    os.makedirs(directory)

model_path=os.path.join(directory,'keras_cifar10.h5')

cnn_model.save(model_path)