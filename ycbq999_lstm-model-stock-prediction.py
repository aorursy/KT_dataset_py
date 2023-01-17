import numpy as np

import datetime

from pandas_datareader import data as web

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential



def get_start_end_date():

    end = datetime.date.today()

    start = datetime.date(end.year-5,1,1)

    return start, end



def loadStockData(ticker, start, end):

    try:   

        data = web.DataReader(ticker, 'yahoo', start, end)

        #data.to_csv(CSV_PATH + ticker +'.csv')

        #print (ticker,' data has data stored to csv file')

        return data



    except:

        print("No information for ticker # and symbol:")

    

     

    

start, end = get_start_end_date()   

df = loadStockData('TSLA', start,end)
df.dropna(inplace = True) 

df.describe()    

prices = round(df['Close'], 2).tolist()

dates = df.index.date.tolist()

#dates = np.arange(len(df)).tolist()



data = df.sort_index(ascending=True, axis=0)

dates = data.index.date.tolist()

new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])




for i in range(0,len(data)):

    new_data['Date'][i] = dates[i]

    new_data['Close'][i] = data['Close'][i]



#setting index

new_data.index = new_data.Date

new_data.drop('Date',axis = 1, inplace = True)

#creating train and test sets



dataset = new_data.values



train = dataset[0:987,:]

valid = dataset[987:,:]



#converting dataset into x_train and y_train



scaler = MinMaxScaler(feature_range = (0,1))

scaled_data = scaler.fit_transform(dataset)





x_train, y_train = [], []

for i in range(60,len(train)):

    x_train.append(scaled_data[i-60:i,0])

    y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)



x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))

from keras.layers import Dense, Dropout,LSTM

# create and fit the LSTM network

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))

model.add(LSTM(units=50))

model.add(Dense(1))





model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
#predicting 246 values, using past 60 from the train data

inputs = new_data[len(new_data) - len(valid) - 60:].values





inputs = inputs.reshape(-1,1)

inputs  = scaler.transform(inputs)    





X_test = []

for i in range(60,inputs.shape[0]):

    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)



X_test2 = np.reshape(X_test[0],(1,X_test.shape[1],1))

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

closing_price = model.predict(X_test)

closing_price = scaler.inverse_transform(closing_price)
# plotting the predicted stock price and true stock price

train = new_data[:987]

valid = new_data[987:]

valid['Predictions'] = closing_price

plt.plot(train['Close'])

plt.plot(valid[['Close','Predictions']])