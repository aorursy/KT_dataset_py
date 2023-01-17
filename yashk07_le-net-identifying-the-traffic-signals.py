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
import pickle
with open("../input/traffic-sign-classification/train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("../input/traffic-sign-classification/valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("../input/traffic-sign-classification/test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)
X_train,y_train = train['features'],train['labels']
X_validation,y_validation = valid['features'],valid['labels']
X_test,y_test = test['features'],test['labels']
X_train.shape
y_train.shape
X_test.shape
y_test.shape
X_validation.shape
y_validation.shape
i = 234
plt.imshow(X_train[i]) #will show the image
y_train[i] #this will show to which class does the image belongs
i = 12232
plt.imshow(X_train[i])
y_train[i]
from sklearn.utils import shuffle
X_train,y_train = shuffle(X_train,y_train)
X_train_gray = np.sum(X_train/3,axis=3,keepdims=True) #While keeping the dimensions same, we are averaging the 3 colours RGB into one(gray).
X_train_gray.shape #now the dimension is 32 by 32 by 1.

X_test_gray = np.sum(X_test/3,axis=3,keepdims=True)
X_validation_gray = np.sum(X_validation/3,axis=3,keepdims=True)
X_test_gray.shape
X_validation_gray.shape
X_train_gray_norm = (X_train_gray - 128)/128
X_test_gray_norm = (X_test_gray - 128)/128
X_validation_gray_norm = (X_validation_gray - 128)/128
X_train_gray_norm
i = 300
plt.imshow(X_train_gray[i].squeeze(),cmap='gray')#we use squeeze becuase we dont want the dimns to be 32 by 32 by 1 but 32 by 32.
y_train[i]
plt.figure() #create new image
plt.imshow(X_train[i]) #actual image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,AveragePooling2D,Dense,Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
cnn_model = Sequential()
cnn_model.add(Conv2D(filters = 6,kernel_size = (5,5),activation = 'relu',input_shape =(32,32,1))) #filters represent the depth,kernal size is size of filter layer,input shape is dimns of the input image
cnn_model.add(AveragePooling2D())

cnn_model.add(Conv2D(filters = 16,kernel_size = (5,5),activation = 'relu')) #filters represent the depth,kernal size is size of filer layer,input shape is dimns of the input image
cnn_model.add(AveragePooling2D())

cnn_model.add(Flatten())
#creating an artificial neural network
cnn_model.add(Dense(120,activation='relu'))
cnn_model.add(Dense(84,activation='relu'))
cnn_model.add(Dense(43,activation='softmax')) #output - hence 43 neurons corresponding to 43 classes


cnn_model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
history = cnn_model.fit(X_train_gray_norm,y_train,batch_size=500,epochs = 20,verbose=1,validation_data=(X_validation_gray_norm,y_validation))
score = cnn_model.evaluate(X_test_gray_norm,y_test)
print('Test Accuracy:{}'.format(score[1]))
plott = pd.DataFrame(history.history)
plott.plot()
prediction = cnn_model.predict_classes(X_test_gray_norm)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,prediction)
plt.figure(figsize=(25,25))
sns.heatmap(cm,annot=True)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)
