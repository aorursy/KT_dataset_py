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
    f1 = cv2.imread(os.path.join(loc1,i))
    f1 = cv2.resize(f1,(100,100))
    features.append(f1)
    
for i in tqdm(os.listdir(loc2)):
    f2 = cv2.imread(os.path.join(loc2,i))
    f2 = cv2.resize(f2,(100,100))
    features.append(f2)
import numpy as np
Y = np.array(labels)
X = np.array(features)
from keras.utils import np_utils
Xt = (X - X.mean())/X.std()        #Normalised the data
Yt = np_utils.to_categorical(Y)    #Categorical representation
Xt = Xt.reshape(5216,30000)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(Xt,Yt)
x_train.shape[1]
model = keras.Sequential()
model.add(layers.Dense(300, activation="relu" , input_dim = x_train.shape[1]))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

adam = keras.optimizers.Adam(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam , metrics = ['Accuracy'])
model.fit(x_train,y_train,epochs=200)
model.evaluate(x_test,y_test)
model.summary()
from sklearn.ensemble import RandomForestClassifier
rmodel = RandomForestClassifier()
rmodel.fit(x_train,y_train)
print(rmodel.score(x_train,y_train))
print(rmodel.score(x_test,y_test))
import matplotlib.pyplot as plt
plt.imshow(x_test[56].reshape(100,100,3))
plt.show()
p = model.predict(x_test[56].reshape(1,30000))
np.argmax(p)
np.argmax(y_test[56])
model.save('my_model.h5')
from keras.models import load_model
model_d = load_model('./my_model.h5')
model_d.summary()
