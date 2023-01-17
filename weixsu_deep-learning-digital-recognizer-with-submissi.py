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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
%matplotlib inline

train_file = '../input/digit-recognizer/train.csv'
raw_train = pd.read_csv(train_file)
raw_train.head()
#Borrowed from Kaggle's Deep Learning course
img_rows, img_cols = 28, 28
num_classes = 10

def data_prep (raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)
    num_images = raw.shape[0]
    x_as_array = raw.values[:, 1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array/255 #a conventional procedure to allow color intensity to be represented by 0 to 1
    return out_x, out_y
X, y = data_prep(raw_train)
model = Sequential()
model.add(Conv2D(100, kernel_size = (3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(100, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(100, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(100, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))#all connected layer before the prediction layer
model.add(Dense(num_classes, activation='softmax'))#this is the prediction layer

#now that the network structure is setup, let's compile it
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=128, epochs=3, validation_split = 0.2)
test_fp = '../input/digit-recognizer/test.csv'
test_data = pd.read_csv(test_fp)
test_data
test_data.shape
x_as_array = test_data.values[:,:]
x_shaped_array = x_as_array.reshape(test_data.shape[0], img_rows, img_cols, 1)
X_test = x_shaped_array/255 #a conventional procedure to allow color intensity to be represented by 0 to 1
y_pred = model.predict(X_test)
sample_data = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sample_data.tail()
y_df = pd.DataFrame(y)
y_df
y_label = np.argmax(y_pred, axis=1)
submission = pd.DataFrame({'ImageId': test_data.index+1, 'Label':y_label})
submission
submission.to_csv('digit_reg_submission_noindex_v4.csv', index=False)