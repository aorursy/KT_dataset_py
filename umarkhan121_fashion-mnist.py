import numpy as np
import pandas as pd 
import os 
for x in os.listdir('/kaggle/input/fashionmnist'):
    print(x)
data=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
data.head()
data=data.to_numpy()
X=data[:,1:].reshape(-1,28,28,1)/255.0
Y=data[:,0].astype(np.int32)

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
Y=to_categorical(Y)
Y
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(126,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.summary()
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
v_data=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
v_data=v_data.to_numpy()
vx=v_data[:,1:].reshape(-1,28,28,1)/255.0
vy=v_data[:,0].astype(np.int32)
vy=to_categorical(vy)
H=model.fit(X,Y,epochs=10,validation_data=(vx,vy))
from matplotlib import pyplot as plt 
plt.plot(H.history['accuracy'],label='acc')
plt.plot(H.history['val_accuracy'],label='val_acc')
plt.show()
