import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
print('numpy version:',np.__version__)
print('pandas version:',pd.__version__)

import tensorflow as tf
print('tensorflow version:',tf.__version__)

from tensorflow import keras
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()
test = pd.read_csv('../input/test.csv')
print(test.shape)
test.head()
from keras.utils.np_utils import to_categorical

X_train = train.drop(['label'],axis=1).values / 255
y_train = to_categorical(train['label'].values, 10)
print('X_train.shape=',X_train.shape)
print('y_train.shape=',y_train.shape)
model = keras.Sequential([
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax, name='OutLayer')
])
model.compile(optimizer=tf.train.AdamOptimizer(),
             loss='categorical_crossentropy',
             metrics=['accuracy'])
# Early stopping  
early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
history = model.fit(X_train, y_train, epochs=50, batch_size=1000, validation_split=0.1, callbacks=[early_stop])
X_test = test.values/255
pred = model.predict(X_test)
pred
pred = np.argmax(pred,axis=1)
pred
submission = pd.read_csv("../input/sample_submission.csv")
submission["Label"] = pred
submission.to_csv("submission.csv", index=False)
submission.head()
# plot history
import matplotlib.pyplot as plt
%matplotlib inline

def plot_history(history):
    plt.plot(history.history['loss'],"o",label="loss")
    plt.plot(history.history['val_loss'],"x",label="val_loss")
    plt.plot(history.history['acc'],"+",label="acc")
    plt.plot(history.history['val_acc'],"*",label="val_acc")
    plt.title('history')
    plt.xlabel('epoch')
    plt.ylabel('loss/accuracy')
    plt.legend(loc='best')
    plt.show()
plot_history(history)
