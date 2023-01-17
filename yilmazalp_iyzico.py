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
import matplotlib.pyplot as plt



train_data = pd.read_csv("/kaggle/input/iyzico-projesi/train.csv")

test_data = pd.read_csv("/kaggle/input/iyzico-projesi/test.csv")
train_data.head()
train_data.tail()
train_data.shape
test_data.tail()
test_data.shape
train_data = train_data.drop(["EMAIL", "CARDBANKID", "MERCHANT_ID"], axis= 1)
count = 0

for row in range(train_data.shape[0]):

    for column in range(train_data.shape[1]):

        if pd.isnull(train_data.iloc[row, column]) == True:

            count += 1

        else:

            continue



if count == 0:

    print('There is no NaN value in the data set')
features = train_data.iloc[:, :-1]

target = train_data.iloc[:, -1]
features.shape
target.shape
import random



from sklearn.model_selection import train_test_split

train_features, test_features, train_target, test_target = train_test_split(features, target, test_size = 0.33, random_state = random.randrange(0,100))
count = 0



for index in range(len(test_target)):

    if target[index] == 1:

        count += 1

        

print(count)
train_features.shape
train_features = np.expand_dims(train_features, axis = 2)

test_features = np.expand_dims(test_features, axis = 2)
train_features.shape
import keras 



train_target = keras.utils.to_categorical(train_target)

test_target = keras.utils.to_categorical(test_target)
train_target.shape
from keras.layers import Dense, Flatten, Dropout, BatchNormalization

from keras.models import Sequential 

from keras.regularizers import l1, l2, l1_l2
model = Sequential()
#model.add(LSTM(50, return_sequences=True, input_shape=(train_features.shape[1], 1), activation='tanh'))

model.add(Dense(64, input_shape=(train_features.shape[1], 1), activation = 'relu', kernel_regularizer='l2'))

model.add(Dense(16, activation = 'tanh'))

model.add(Dense(4, activation = 'relu'))





model.add(Flatten())

#model.add(LSTM(50, return_sequences=True, activation='relu'))

#model.add(LSTM(50, return_sequences=True, activation='tanh'))

#model.add(LSTM(50, return_sequences=False, activation='tanh'))
#model.add(Dropout(0.5))

#model.add(BatchNormalization())

model.add(Dense(2, activation = 'sigmoid', kernel_regularizer='l2'))
opt_param = keras.optimizers.Adam(learning_rate=0.000001)

model.compile(loss='binary_crossentropy', optimizer = opt_param, metrics=['accuracy'])
ann = model.fit(train_features, train_target, batch_size=1024, epochs=20, validation_data=(test_features, test_target))
#Plot the Loss Curves

plt.figure(figsize=[8,6])

train_loss, = plt.plot(ann.history['loss'],'r',linewidth=3.0)

test_loss, = plt.plot(ann.history['val_loss'],'b',linewidth=3.0)

plt.legend([train_loss, test_loss], ['Training Loss', 'Test Loss'],fontsize=12)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves',fontsize=16)

 

#Plot the Accuracy Curves

plt.figure(figsize=[8,6])

train_accuracy, = plt.plot(ann.history['accuracy'],'r',linewidth=3.0)

test_accuracy, = plt.plot(ann.history['val_accuracy'],'b',linewidth=3.0)

plt.legend([train_accuracy, test_accuracy], ['Training Accuracy', 'Test Accuracy'],fontsize=12)



plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)
test_data.head()
test_data = test_data.drop(["EMAIL", "CARDBANKID", "MERCHANT_ID", "ID"], axis=1)
test_data = np.expand_dims(test_data, axis = 2)
test_data.shape
prediction = model.predict(test_data)
prediction
prediction_classes = model.predict_classes(test_data)
predictions = list(prediction_classes)
result_df = pd.DataFrame(columns = ['ID', 'ISFRAUD'])
transaction_id = []



for index in range(0, len(predictions)):

    transaction_id.append(index)
result_df['ID'] = transaction_id

result_df['ISFRAUD'] = predictions
result_df.head()
result_df.tail()
result_df.to_csv('last_submission2.csv', index = False)