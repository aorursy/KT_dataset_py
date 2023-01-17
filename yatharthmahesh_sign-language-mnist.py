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
import tensorflow as tf

import matplotlib.pyplot as plt

from PIL import Image
plt.imshow(np.array(Image.open('/kaggle/input/sign-language-mnist/american_sign_language.PNG')))

#plt.imshow(np.array(Image.open('/kaggle/input/sign-language-mnist/amer_sign3.png')))

df1=pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')

df2=pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')

print(df1.shape)

print(df2.shape)

y=np.array(df1['label'].values)

x=df1.drop(['label'],axis=1).values

print(x.shape)

x=np.array([np.reshape(i,(28,28,1)) for i in x])

x=x/255

y=y.reshape(-1,1)

#print(x)

from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()

y= label_binrizer.fit_transform(y)

print(x[0].shape)

print(x.shape)

print(y)
model=tf.keras.Sequential([tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),

                           tf.keras.layers.MaxPooling2D(2,2),

                           tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

                           tf.keras.layers.MaxPooling2D(2,2),

                            tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

                           tf.keras.layers.MaxPooling2D(2,2),

                           tf.keras.layers.Flatten(),

                           tf.keras.layers.Dense(128,activation='relu'),

                           tf.keras.layers.Dense(24,activation='softmax')

                          ])
y_test=np.array(df2['label'].values).reshape(-1,1)

x_test=df2.drop(['label'],axis=1).values

x_test=np.array([np.reshape(i,(28,28,1)) for i in x_test])/255



from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()

y_test= label_binrizer.fit_transform(y_test)

model.compile(optimizer="rmsprop",loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
history=model.fit(x,y,epochs=25,batch_size=128,validation_data=(x_test,y_test))
history