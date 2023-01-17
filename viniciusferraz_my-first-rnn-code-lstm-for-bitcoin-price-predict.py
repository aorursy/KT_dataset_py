from IPython.display import Image

Image("../input/images/RNN.png")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, SimpleRNN,Embedding
n = 1000

train_point = int(0.8*n)



t = np.arange(0,n)

X = np.sin((0.02*t)+2*np.random.rand(n)) #Multiplications done to make the graph "clean"
df = pd.DataFrame(X)

df.head(3)
plt.plot(df)
step = 4
values = df.values

train,test = values[0:train_point,:], values[train_point:n,:]

#train.shape = (800,1)
step = 4

# add step elements into train and test

test = np.append(test,np.repeat(test[-1,],step))

train = np.append(train,np.repeat(train[-1,],step))

train.shape # = (804,1)
# convert into dataset matrix

def convertToMatrix(data, step):

    X, Y =[], []

    for i in range(len(data)-step):

        d=i+step  

        X.append(data[i:d,])

        Y.append(data[d,])

    return np.array(X), np.array(Y)
trainX,trainY =convertToMatrix(train,step)

print(trainX.shape)

testX,testY =convertToMatrix(test,step)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

trainX.shape
trainY.shape
#(batch_size, timesteps, input_dim)

def RNN():

    model = Sequential()

    model.add(SimpleRNN(units=32, input_shape=(1,step), activation="relu"))

    model.add(Dense(8,activation="relu"))

    model.add(Dense(1))

    model.compile(loss="mean_squared_error",optimizer="adam")

    return model
#print(trainX.shape,trainy.shape)

model = RNN()

model.fit(trainX,trainY, epochs=100, batch_size=16, verbose=0)
trainPredict = model.predict(trainX)

testPredict= model.predict(testX)

predicted=np.concatenate((trainPredict,testPredict),axis=0)
trainScore = model.evaluate(trainX, trainY, verbose=0)

print(trainScore)
index = df.index.values

plt.plot(index,df)

plt.plot(index,predicted)

plt.axvline(df.index[train_point], c="lightgreen",lw=2)

plt.show() 
#dayofweek = ["Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday"]

dayofweek = [1,2,3,4,5,6,7]

#dinner = ["Pizza","Burguer","Pancake","Spaghetti","Meat","Chicken","Sausage"]

dinner = [1,2,3,4,5,6,7]



#Repeating for greater dataset

dayofweek = np.array(144*dayofweek) # = 1008 days

dinner = np.array(144*dinner)
# Time-step

tstep = 2



# train = ~80% | test = ~20%

tp = 800



trainX, trainY = dayofweek[0:tp], dinner[0:tp]

testX, testY = dayofweek[tp:], dinner[tp:]
testX.shape
def makematrix(dataX, dataY, tstep):

    X, Y = [], []

    for i in range(len(dataX)-tstep):

        X.append(dataX[i:i+tstep,])

        Y.append(dataY[i+tstep])

    return np.array(X), np.array(Y)
trainX, trainY = makematrix(trainX, trainY, tstep)

testX, testY = makematrix(testX, testY, tstep)
testX.shape
trainX = np.reshape(trainX, (trainX.shape[0], 1, tstep))

testX = np.reshape(testX, (testX.shape[0], 1, tstep))
def graph(hist):

    # Plots Loss Line.

    plt.plot(hist.history['loss'])

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    

    lss = str(hist.history['loss'][-1])

    plt.title(str('Loss='+lss))

    plt.show()
def RNN():

    model = Sequential()

    import keras

    keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=5)

    model.add(SimpleRNN(units=32, input_shape=(1,tstep), kernel_initializer="normal",activation="relu"))

    model.add(Dense(1))

    keras.optimizers.Adam(decay=1e-6,learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False )

    model.compile(loss="mean_squared_error",optimizer="adam")

    return model
model = RNN()

hist = model.fit(trainX, trainY, epochs=350, batch_size=32, verbose=0)
trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

predicted = np.concatenate((trainPredict,testPredict),axis=0)
trainScore = model.evaluate(trainX, trainY, verbose=0)

print(trainScore)
graph(hist)
seq = np.array([[[1, 2]],[[4, 5]],[[7, 1]]])

# NextDay: [Wednesday],[Saturday],[Tuesday]

# Returns: Pancake, Chicken, Burguer

#           3.0      6.0      2.0



for i in model.predict(seq):

    if round(i[0]) == 1:

        print('\n',round(i[0],3))

        print("Pizza")

    elif round(i[0]) == 2:

        print('\n',round(i[0],3))

        print("Burguer")

    elif round(i[0]) == 3:

        print('\n',round(i[0],3))

        print("Pancake")

    elif round(i[0]) == 4:

        print('\n',round(i[0],3))

        print("Spaghetti")

    elif round(i[0]) == 5:

        print('\n',round(i[0],3))

        print("Meat")

    elif round(i[0]) == 6:

        print('\n',round(i[0],3))

        print("Chicken")

    else:

        print("Sausage")

Image("../input/images/LSTM.png",width=400, height=400)
Image("../input/images/LSTM2.png", width=550, height=550)
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
btcdf = pd.read_csv('../input/bitcoin-historical-price/Bitcoin_Historical_Price.csv')
btcdf['Date'] = pd.to_datetime(btcdf['Date'])

btcdf.head(3)
btcdf['Close'] = btcdf['Close']/btcdf['Close'].max()
group = btcdf.groupby('Date')
Real_Price = group['Close'].mean()
Real_Price[:3]
Real_Price.shape
split_point = 1400

step = 30

X_train = Real_Price[:split_point]

y_train = Real_Price[step:split_point+step] # 30 days after train

X_test = Real_Price[split_point:-step] # Values until the 30th last value

y_test = Real_Price[split_point+step:] # 30 days after test
def makematrix(dataX, dataY, step):

    X, Y = [], []

    for i in range(len(dataX)-step):

        X.append(dataX[i:i+step,])

        Y.append(dataY[i+step])

    return np.array(X), np.array(Y)
X_train, y_train = makematrix(X_train, y_train, step)

X_test, y_test = makematrix(X_test, y_test, step)
X_train.shape
X_train = np.reshape(X_train, (X_train.shape[0], 1, step))

X_test = np.reshape(X_test, (X_test.shape[0], 1, step))
X_train.shape
#X_train = sc.fit_transform(X_train)

#X_test = sc.fit_transform(X_test)
from keras.layers import LSTM
def BTC_RNN():

    model = Sequential()

    model.add(LSTM(units=64, activation="sigmoid", input_shape=(1,step)))

    model.add(Dense(units=1))

    import keras

    keras.optimizers.Adam(decay=1e-6,learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False )

    model.compile(loss="mean_squared_error",optimizer="adam")

    return model
model = BTC_RNN()

hist1 = model.fit(X_train, y_train, batch_size=32, epochs=5)
graph(hist1)
X = y_test

y = model.predict(X_test)

plt.figure(figsize=(20,4))

plt.plot(X,color='orange', label='Real BTC Prices')

plt.plot(y,color='b', label = 'Predicted BTC Prices')

plt.legend(loc=2, prop={'size': 12})

plt.title('BTC Price Prediction', fontsize=20)

plt.xlabel('Time', fontsize=15)

plt.ylabel('BTC Price(USD)', fontsize=15)

plt.show()