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

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import Sequential


data=pd.read_csv('../input/digit-recognizer/train.csv')
data.head(20)
x=data.iloc[:,1:] / 255
y=data.iloc[:,0]
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=0)
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid'),
])
model.compile(optimizer='adam',metrics=['accuracy'],loss='sparse_categorical_crossentropy')
model.fit(train_x,train_y,epochs=20,)
test_loss,test_acc=model.evaluate(test_x,test_y)
prediction=model.predict(test_x)
np.argmax(prediction[1])
np.array(test_y)[1]
plt.figure()
plt.imshow(np.reshape(np.array(test_x)[1],(28,28)))
