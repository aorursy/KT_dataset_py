# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import tensorflow as tf

from tensorflow.keras import losses, layers

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
def show_num(img,lbl):

    print("Label ", lbl)

    plt.imshow(img.reshape((28,28)))
X = train.iloc[:,1:].to_numpy()

y = train.iloc[:,0]
i = np.random.randint(train.shape[0])

show_num(X[i],y[i])
X = X.reshape(-1,28,28)

X = np.expand_dims(X,axis=-1)
X.shape
model = tf.keras.Sequential([

    layers.Conv2D(32,3,activation='relu',input_shape=(28,28,1)),

    layers.MaxPool2D(2),

    layers.Conv2D(64,3,activation='relu'),

    layers.GlobalAveragePooling2D(),

    layers.Dense(512,activation='relu'),

    layers.Dense(10,activation='softmax')

])



model.compile(optimizer='adam',loss=losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
model.summary()
tf.keras.utils.plot_model(model)
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=.2,random_state=0)
epochs = 10

batch = 32



model.fit(xtrain,ytrain,epochs=epochs,batch_size=batch,validation_data=(xtest,ytest))
xtest = np.expand_dims(test.to_numpy().reshape((-1,28,28)),axis=-1)
xtest.shape
pred = model.predict_classes(xtest)
sub['Label'] = pred
sub.to_csv("submission.csv",index=False)