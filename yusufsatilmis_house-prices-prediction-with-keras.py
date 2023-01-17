import numpy as np 

import pandas as pd 

import os

from random import sample 



import keras

from keras.layers.advanced_activations import LeakyReLU

from keras import models

from keras import layers

from keras.models import Sequential

from keras.layers import Dense

from keras import backend as K

from keras import optimizers



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



import matplotlib.pyplot as plt

import matplotlib



print(os.listdir("../input"))

train_csv = pd.read_csv('../input/train.csv')

test_csv = pd.read_csv('../input/test.csv')

sample_submission_csv = pd.read_csv('../input/sample_submission.csv')

print (train_csv.shape)

print (train_csv.shape)

print (sample_submission_csv.shape)
train_csv.head()
test_csv.head()
sample_submission_csv.head()
test_csv["SalePrice"]=sample_submission_csv["SalePrice"]

all_data = pd.concat([train_csv, test_csv], axis=0)

train_data=all_data.loc[:,all_data.dtypes!=object]

train_data=train_data.replace(-1,np.NAN)

train_data.fillna((train_data.mean().round()), inplace=True)

train_data.drop('Id',axis = 1, inplace = True)

print (all_data.shape)
train_data.head()
train_column = list(train_data.columns)



train_matrix = np.matrix(train_data)

train_matrix_norm = MinMaxScaler()

train_matrix_norm.fit(train_matrix)





test_matrix = np.matrix(train_data.drop('SalePrice',axis = 1))

test_matrix_norm = MinMaxScaler()

test_matrix_norm.fit(test_matrix)





y_matrix = np.array(train_data.SalePrice).reshape((2919,1))

y_matrix_norm = MinMaxScaler()

y_matrix_norm.fit(y_matrix)







train = pd.DataFrame(train_matrix_norm.transform(train_matrix),columns = train_column)
X = train.drop(columns=["SalePrice"])

Y = train["SalePrice"]
X.head()
Y.head()
x_train , x_test , y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)
print ('X_train.shape',x_train.shape)

print ('X_test.shape',x_test.shape)

print ('Y_train.shape',y_train.shape)

print ('Y_test.shape',y_test.shape)


seed = 7

np.random.seed(seed)



# Model

model = Sequential()

model.add(Dense(200, input_dim=36, kernel_initializer='normal', activation='relu'))

model.add(Dense(100, kernel_initializer='normal', activation='relu'))

model.add(Dense(50, kernel_initializer='normal', activation='relu'))

model.add(Dense(25, kernel_initializer='normal', activation='relu'))

model.add(Dense(1, kernel_initializer='normal'))

# Compile model

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam())

              #optimizer=keras.optimizers.Adadelta())





# increase epochs from 100 to 200

history = model.fit(np.array(x_train), np.array(y_train), epochs=100, batch_size=100)
print(history.history.keys())

plt.plot(history.history['loss'])

#plt.plot(history.history['val_acc'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y = model.predict(np.array(x_test))
predictions = y_matrix_norm.inverse_transform(np.array(y).reshape(584,1))
reality = y_matrix_norm.inverse_transform(np.array(y_test).reshape(584,1))
matplotlib.rc('xtick', labelsize=30) 

matplotlib.rc('ytick', labelsize=30) 



plt.figure(figsize=(20,20))

plt.scatter(predictions, reality)

plt.title('Predictions x Reality',fontsize = 30)

plt.xlabel('Predictions',fontsize = 30)

plt.ylabel('Reality',fontsize = 30)

plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)

plt.show()
error=[]

for i in range (len(reality)):

    error.append(abs(reality[i][0]-predictions[i][0])/reality[i][0]*100)

mean = sum(error)

print ('Hata orani = %',mean/len(error))