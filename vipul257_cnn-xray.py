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
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate
import os
labels = []
for i in os.listdir('../input/chest-xray-pneumonia/chest_xray/train/NORMAL'):
    labels.append(0)
for i in os.listdir('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'):
    labels.append(1)
import cv2
loc1 = '../input/chest-xray-pneumonia/chest_xray/train/NORMAL'
loc2 = '../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'
features = []
from tqdm import tqdm
for i in tqdm(os.listdir(loc1)):
    f1 = cv2.imread(os.path.join(loc1,i),0)
    f1 = cv2.resize(f1,(100,100))
    features.append(f1)
    
for i in tqdm(os.listdir(loc2)):
    f2 = cv2.imread(os.path.join(loc2,i),0)
    f2 = cv2.resize(f2,(100,100))
    features.append(f2)
import numpy as np
Y = np.array(labels)
X = np.array(features)
X.shape = (5216,100,100,1)
Xt = (X - X.mean())/X.std()
Yt = np_utils.to_categorical(Y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(Xt,Yt)
 
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
 
model.summary()
#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(x_train)
#training
batch_size = 64
 
adam = keras.optimizers.Adam(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=30,verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])
#save to disk
model.fit(x_train,y_train,epochs=15)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5') 
#testing
scores = model.evaluate(x_test, y_test)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

x_test.shape
status = np.array(['Normal' , 'Viral Pneumonia' , 'Bacterial Pneumonia'])
prediction = np.argmax(model.predict(x_test[50].reshape(1,100,100,3)))
print(prediction)
print(status[prediction])
print(np.argmax(y_test[50]))
import matplotlib.pyplot as plt
plt.imshow(x_test[50])
plt.show()
import matplotlib.pyplot as plt
plt.imshow(X[50])
plt.show()