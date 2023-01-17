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
# import library
import glob
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import keras 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten, Dropout
from sklearn.model_selection import train_test_split
# load train data
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_train.head()
Y = df_train['label']
Y_input = to_categorical(Y, num_classes = 10)
data_train = df_train.drop(['label'], axis = 1)
Y_input.shape
# plot the labels
sns.countplot(x=Y)
# load test data
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
df_test.head()
# scale the data
s = MinMaxScaler()
s.fit(data_train)
data_scaled = s.transform(data_train)
data_input = data_scaled.reshape(42000, 28, 28, 1)
test_data = s.transform(df_test)
test_set = test_data.reshape(-1, 28, 28, 1)
# create model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, input_shape = (28, 28, 1), kernel_size = (3,3), strides = (1, 1), padding ='same',  kernel_initializer='he_uniform'))
    model.add(LeakyReLU(0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), padding = 'same', strides =(1, 1), kernel_initializer='he_uniform' ))
    model.add(LeakyReLU(0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.20))
    model.add(Conv2D(128, (3, 3), padding = 'same', strides =(1, 1),kernel_initializer='he_uniform' ))
    model.add(LeakyReLU(0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(250, kernel_initializer='he_uniform'))
    model.add(LeakyReLU(0.1))
    model.add(Dense(100, kernel_initializer='he_uniform'))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.20))
    model.add(Dense(10,kernel_initializer='he_uniform', activation = 'sigmoid'))
    model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adamax')
    return model
# create generator for augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)
# train and save model
nets = 16
models = []
History = []
for i in range(nets):
    model = create_model()
    x_train, x_val, y_train, y_val = train_test_split(data_input, Y_input, test_size = 0.1)
    history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=64),
        epochs = 132, steps_per_epoch = x_train.shape[0]//64,  
        validation_data = (x_val,y_val))
    models.append(model)
    model.save(f'm{i}.h5')
    History.append(history)
# predict the model
models = glob.glob('./*')
models.remove('./__notebook_source__.ipynb')
result = 0
for i in models:
  m = tf.keras.models.load_model(i)
  m_pred = np.argmax(m.predict(test_set), axis = 1)
  m_pred_cat = to_categorical(m_pred, num_classes = 10)
  result = result + m_pred_cat
y_pred = np.argmax(result, axis = 1)
df_output = pd.DataFrame()
df_output['ImageId'] = list(range(1, 28001))
df_output['Label'] = y_pred
df_output.to_csv('output.csv', index = False)