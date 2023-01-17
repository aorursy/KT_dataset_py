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
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Dense,BatchNormalization

from tensorflow.keras.layers import Flatten

from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

submission_file = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

train.shape,test.shape,submission_file.shape
X = train.drop('label',axis=1).values

y = train['label'].values





test = test.values
X = X.reshape(-1,28,28,1)

test = test.reshape(-1,28,28,1)
X = X/255.0



test = test/255.0
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
trainX = X_train.reshape((X_train.shape[0], 28, 28, 1))

testX = X_test.reshape((X_test.shape[0], 28, 28, 1))
trainY = y_train

testY = y_test
train_norm = trainX.astype('float32')

test_norm = testX.astype('float32')

# normalize to range 0-1

train_norm = train_norm / 255.0

test_norm = test_norm / 255.0
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))

model.summary()
# compile model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=30, batch_size=128, validation_data=(testX, testY), verbose=1)
plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
test_predicted = model.predict_classes(testX)

test_predicted
model.evaluate(testX, testY)
import tensorflow as tf

cm = tf.math.confusion_matrix(labels = testY, predictions = test_predicted)

cm
import seaborn as sns

plt.figure(figsize=(20,8))

sns.heatmap(cm,annot=True,fmt='d')

plt.ylabel('Actual')

plt.xlabel('Predicted')
submission_pred = model.predict_classes(test)

len(submission_pred)
submission_file['Label'] = submission_pred
submission_file.to_csv('submission_file.csv',index=False)