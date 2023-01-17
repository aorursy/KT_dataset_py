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
dataset = pd.read_csv("/kaggle/input/international-airline-passengers/international-airline-passengers.csv")

dataset.columns = ['month', 'passengers']
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

ax = sns.lineplot(x=dataset.month, y=dataset.passengers)
def create_dataset(dataset, look_back=1):

	dataX, dataY = [], []

	for i in range(len(dataset)-look_back-1):

		a = dataset[i:(i+look_back)]

		dataX.append(a)

		dataY.append(dataset[i + look_back])

	return np.array(dataX), np.array(dataY)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

dataset_array = scaler.fit_transform(dataset.passengers.values.reshape(-1, 1))
index = int(len(dataset_array) * 0.8)

train = dataset_array[:index]

test = dataset_array[index:]
pasos_atras = 3

X_train, y_train = create_dataset(train, look_back=pasos_atras)

X_test, y_test = create_dataset(test, look_back=pasos_atras)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
from keras.models import Sequential

from keras.layers import Dense, LSTM

from keras import metrics

model = Sequential()

model.add(LSTM(8, input_shape=(1, pasos_atras)))

model.add(Dense(4, activation="linear"))

model.add(Dense(1))



model.compile(loss="mean_squared_error",

             optimizer="Adam",

             metrics=[metrics.mae]

             )

history = model.fit(X_train,

         y_train,

         validation_split=0.2,

         epochs=300, 

         batch_size=50,

         verbose=0)



sns.set()

# Plot training & validation accuracy values

plt.plot(history.history['mean_absolute_error'])

plt.plot(history.history['val_mean_absolute_error'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
train_predict = scaler.inverse_transform(model.predict(X_train))

y_train = scaler.inverse_transform(y_train)



test_predict = scaler.inverse_transform(model.predict(X_test))

y_test = scaler.inverse_transform(y_test)
to_plot = pd.DataFrame({"Real": np.squeeze(np.concatenate((y_train, y_test))), "Predicted": np.squeeze(np.concatenate((train_predict, test_predict)))})

plt.plot(to_plot.index, to_plot.Real)

plt.plot(to_plot.index, to_plot.Predicted, linestyle="-")

plt.plot()