import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import MinMaxScaler



filename = '../input/train_data/train_data.csv'

data = pd.read_csv(filename)

engines = data['engine_no'].unique()

data = data.dropna(axis=1)

data['RUL'] = data['RUL']<100

mem = data['engine_no']



filename = '../input/test_data/test_data.csv'

data_test = pd.read_csv(filename)

engines_test = data_test['engine_no'].unique()

data_test = data_test.dropna(axis=1)

mem_test = data_test['engine_no']
data_test_reshaped = np.zeros((np.array(data_test).shape[0],np.array(data_test).shape[1]+1))

data_test_reshaped[:,:-1] = data_test



datagier = np.concatenate((np.array(data), np.array(data_test_reshaped)),axis=0)



trainsize = np.array(data).shape[0]



# data normalization



scaler = MinMaxScaler(feature_range=(0, 1))

scaler_output = MinMaxScaler(feature_range=(0, 1))



data = scaler.fit_transform(datagier)



data_train = data[:trainsize,:]

data_train[:,0] = mem

data_train = pd.DataFrame(data_train)



data_Test = data[trainsize:,:]

data_Test[:,0] = mem_test

data_Test = pd.DataFrame(data_Test)



X_train = []

Y_train = []



look_back = 10



#for eng in engines[:]: uncomment this line

for eng in engines[:1]: # comment this line

    data_pd = data_train.where(data_train[0]==eng)

    data_pd = data_pd.dropna()

    for i in range(len(data_pd)-look_back-1):

        X_train.append(data_pd.values[i:(i+look_back), 1:26])

        Y_train.append(data_pd.values[i + look_back, 26])

    if eng%100==0:

        print('*', end="")



X_train = np.array(X_train)

Y_train = np.array(Y_train)
import keras.metrics as metrics

from keras.models import Sequential

from keras.layers import Dense,Activation,Dropout

from keras.layers import LSTM

import keras.optimizers as optimizers

from keras.models import load_model



model = Sequential()



model.add(LSTM(

         input_shape=(look_back, 25),

         units=200,

         return_sequences=True))

model.add(Dropout(0.3))

model.add(LSTM(

          units=100,

          return_sequences=True))

model.add(Dropout(0.3))

model.add(LSTM(

          units=50))

model.add(Dropout(0.3))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.binary_accuracy])



history = model.fit(X_train, Y_train, epochs=60, batch_size=300, verbose=2)
fichier = open("submission.csv", "w")

fichier.write('engine_no,result\n')



for eng in engines_test[:]:

    count = 0

    data_eng = data_Test.where(data_Test[0]==eng)

    data_eng = data_eng.dropna()

    for i in range(len(data_eng)-look_back-1,len(data_eng)-look_back-7,-1):

        x = data_eng.values[i:(i+look_back), 1:26]

        x = x.reshape((1,x.shape[0], x.shape[1]))

        pred = model.predict(x)

        if pred[0][0]>0.5:

            count += 1

    if count>=3:

        pred = 1

    else:

        pred = 0

    fichier.write(str(eng) + ',' + str(pred) +'\n')

    if eng%100==0:

        print('*',end="")



fichier.close()