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
import keras

from keras.models import Sequential

from keras.layers import Dense

import matplotlib.pyplot as plt

import math

from sklearn.metrics import mean_squared_error
# Load data

df_train = pd.read_csv('/kaggle/input/house-prices-federico/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/house-prices-federico/test.csv', index_col=0)



# Convert to Numpy array

#df_train_np = df_train.values

#df_test_np = df_test.values
# Normalize dataset

normalized_df_train=(df_train-df_train.mean())/df_train.std()

normalized_df_test=(df_test-df_train.mean())/df_train.std()



# Convert to Numpy array

normalized_df_train_np = normalized_df_train.values

normalized_df_test_np = normalized_df_test.values
# Split dataset

X_train = normalized_df_train_np[:, [0,1,2,3,4,5,6,7]]

y_train = normalized_df_train_np[:, [8]]

X_test = normalized_df_test_np[:, [0,1,2,3,4,5,6,7]]

y_test = normalized_df_test_np[:, [8]]
#X_train = df_train_np[:, [0,1,2,3,4,5,6,7]]

#y_train = df_train_np[:, [8]]

#X_test = df_test_np[:, [0,1,2,3,4,5,6,7]]

#y_test = df_test_np[:, [8]]
# 3 layers containing respectively 10, 30 and 40 neurons

model = Sequential()

model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))

model.add(Dense(30, activation='relu'))

model.add(Dense(40, activation='relu'))

model.add(Dense(1)) 

model.compile(optimizer ='adam', loss = 'mean_squared_error', metrics=['mae']) 

#history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=150, batch_size=32) 

history = model.fit(X_train, y_train, epochs=150, batch_size=32) 
# Review model structure

model.summary()
# Line graph of the Loss

plt.ylim(0,0.6)

plt.plot(history.history['loss'], color = 'b')

plt.title('Loss')

plt.xlabel('Epochs')
# Line graph of the MAE

plt.ylim(0,0.6)

plt.plot(history.history['mae'], color = 'r')

plt.title('MAE')

plt.xlabel('Epochs')
# Scatterplot for real and predicted values

pred = model.predict(X_test)

index = [i for i in range(X_test.shape[0])]

plt.scatter(index, pred, color = ['red'])

plt.scatter(index, y_test, color = ['blue'])
# Scatterplot for the first 20 real and predicted values

plt.xlim(-0.5,20.5)

plt.scatter(index, pred, color = ['red'], label = 'Prediction')

plt.scatter(index, y_test, color = ['blue'], label = 'Real')

plt.title('First 20 house prices prediction')

plt.xlabel('Prices')

plt.ylabel('House')
# Histogram

plt.hist(pred,bins=7)

plt.title('Histogram of Predicted Prices')
# Computing Root mean squared error

RMSE = math.sqrt(mean_squared_error(y_test,pred))
# 1 layer containing a single neuron

model1 = Sequential()

model1.add(Dense(1, input_dim=X_train.shape[1], activation='relu'))

model1.add(Dense(1)) 

model1.compile(optimizer ='adam', loss = 'mean_squared_error', metrics=['mae']) 

history1 = model1.fit(X_train, y_train, epochs=150, batch_size=32) 
# Review model structure

model1.summary()
# 1 layer containing 3 neurons

model2 = Sequential()

model2.add(Dense(3, input_dim=X_train.shape[1], activation='relu'))

model2.add(Dense(1)) 

model2.compile(optimizer ='adam', loss = 'mean_squared_error', metrics=['mae']) 

history2 = model2.fit(X_train, y_train, epochs=150, batch_size=32) 
# Review model structure

model2.summary()
# 1 layer containing 10 neurons

model3 = Sequential()

model3.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))

model3.add(Dense(1)) 

model3.compile(optimizer ='adam', loss = 'mean_squared_error', metrics=['mae']) 

history3 = model3.fit(X_train, y_train, epochs=150, batch_size=32) 
# Review model structure

model3.summary()
# 2 layers containing respectively 10 and 30 neurons

model4 = Sequential()

model4.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))

model4.add(Dense(30, activation='relu'))

model4.add(Dense(1)) 

model4.compile(optimizer ='adam', loss = 'mean_squared_error', metrics=['mae']) 

history4 = model4.fit(X_train, y_train, epochs=150, batch_size=32) 
# Review model structure

model4.summary()
# Compare the training Loss by line graph

plt.ylim(0,1)

plt.plot(history1.history['loss'], color = 'b')

plt.plot(history2.history['loss'], color = 'g')

plt.plot(history3.history['loss'], color = 'r')

plt.plot(history4.history['loss'], color = 'y')

plt.plot(history.history['loss'], color = 'k')

plt.title('Loss')

plt.xlabel('Epochs')
# Compare the training MAE by line graph

plt.ylim(0,1)

plt.plot(history1.history['mae'], color = 'b')

plt.plot(history2.history['mae'], color = 'g')

plt.plot(history3.history['mae'], color = 'r')

plt.plot(history4.history['mae'], color = 'y')

plt.plot(history.history['mae'], color = 'k')

plt.title('MAE')

plt.xlabel('Epochs')
# Computing the test Root mean squared error

pred1 = model1.predict(X_test)

RMSE1 = math.sqrt(mean_squared_error(y_test,pred1))



pred2 = model2.predict(X_test)

RMSE2 = math.sqrt(mean_squared_error(y_test,pred2))



pred3 = model3.predict(X_test)

RMSE3 = math.sqrt(mean_squared_error(y_test,pred3))



pred4 = model4.predict(X_test)

RMSE4 = math.sqrt(mean_squared_error(y_test,pred4))



RMSE_comparision = np.array([RMSE1, RMSE2, RMSE3, RMSE4, RMSE])



# Compare the test RMSE by line graph

plt.ylim(0,1)

plt.plot(RMSE_comparision, color = 'b', marker='^')

plt.title('RMSE COMPARISION')

plt.xlabel('NETWORK INDEX')