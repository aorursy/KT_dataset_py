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
    if (i.split('_')[1] ==  'virus'):
        labels.append(1)
    else:
        labels.append(2)
import numpy as np
Y = np.array(labels)
Y.shape
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
X = np.array(features)
X.shape
from keras import models
from keras import layers
from keras.utils import to_categorical
Xt = X.reshape(5216,30000)
Yt = to_categorical(Y)
Xt = Xt/Xt.max()
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(Xt,Yt,test_size=0.2)
model = models.Sequential()
model.add(layers.Dense(400, activation = 'relu', input_dim=Xt.shape[1]))
model.add(layers.Dense(300, activation = 'relu'))
model.add(layers.Dense(200, activation = 'relu'))
model.add(layers.Dense(200, activation = 'relu'))
model.add(layers.Dense(100, activation = 'relu'))
model.add(layers.Dense(100, activation = 'relu'))
model.add(layers.Dense(100, activation = 'relu'))
model.add(layers.Dense(3, activation = 'softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(xtrain,ytrain,epochs=10)
model.evaluate(xtest,ytest)
