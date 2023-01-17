# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head()
test.head()
#Check Null Values
train.isnull().any().sum()
test.isnull().any().sum()
Xtrain = train.iloc[:,1:]
ytrain = train.iloc[:,0]
Xtrain
ytrain
ytrain.value_counts()
sns.countplot(ytrain)
#Normalize the data
Xtrain = Xtrain / 255.0
test = test / 255.0
#Reshape to  3D DataFrame

Xtrain = Xtrain.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
from keras.utils.np_utils import to_categorical
ytrain = to_categorical(ytrain, num_classes=10)
ytrain.shape
#Splitting into training and validation set
from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val = train_test_split(Xtrain,ytrain, test_size = 0.1, random_state=13)
X_train.shape

plt.imshow(X_train[0][:,:,0])

#Build CNN Model
# 1. VGG 16 Model

from keras.layers import Input, Conv2D , MaxPooling2D, Dropout
from keras.layers import Dense, Flatten
from keras.models import Model, Sequential
from keras import regularizers

model = Sequential()
model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = 'Same',activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 128,kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(Conv2D(filters = 128,kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 256,kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(Conv2D(filters = 256,kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(Conv2D(filters = 256,kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(4096 , activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(4096 , activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10 , activation='softmax',kernel_regularizer=regularizers.l2(0.001)))





#Optimizers 
from keras.optimizers import RMSprop

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer , loss='categorical_crossentropy', metrics = ['accuracy'])
#Callbacks
from keras.callbacks.callbacks import ReduceLROnPlateau
callbacks = ReduceLROnPlateau(monitor='val_accuracy',patience = 3,verbose=1,factor=0.5,min_lr=0.00001)
batch_size  = 100
epoch = 20
#Data Augmentation

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
            featurewise_center = False, samplewise_center = False,featurewise_std_normalization = False,
            samplewise_std_normalization = False,zca_whitening = False,
            rotation_range = 10, zoom_range = 0.15, height_shift_range = 0.1,
            horizontal_flip = False,vertical_flip = False)

datagen.fit(X_train)

#fit Model
history = model.fit_generator(datagen.flow(X_train,Y_train,batch_size = batch_size),
                              epochs = epoch, validation_data=(X_val,Y_val),verbose= 2,
                              steps_per_epoch = X_train.shape[0] // batch_size,
                              callbacks = [callbacks] )
#Plot accuracy plot and validation plot

fig,ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'],color = 'b',label = 'Training loss')
ax[0].plot(history.history['val_loss'],color = 'r', label = 'validation loss',axes = ax[0])
legend = ax[0].legend(loc = 'best',shadow= True)

ax[1].plot(history.history['accuracy'],color = 'b',label = 'Training Accuracy')
ax[1].plot(history.history['val_accuracy'], color = 'r',label = 'validation accuracy')
legend = ax[1].legend(loc='best',shadow=True)



#confusion matrix
from sklearn.metrics import confusion_matrix
y_true = np.argmax(Y_val,axis=1)
Y_pred = model.predict(X_val)
y_pred = np.argmax(Y_pred,axis=1) 
cm = confusion_matrix(y_true,y_pred)

sns.heatmap(cm, annot = True, fmt = 'd',xticklabels = 1, yticklabels = 1)
plt.xlabel('Predicted digit')
plt.ylabel('Actual digit')
#Predict result
result = model.predict(test)
results = np.argmax(result,axis = 1)
submission = pd.DataFrame({ 'ImageId' : list(range(1,len(results)+1)),
             'Label': results})

submission.to_csv('Submission file',index=False,header = True)