# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.metrics import mean_squared_error as mse



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!head -3 /kaggle/input/national-stock-exchange-time-series/infy_stock.csv
df = pd.read_csv("../input/national-stock-exchange-time-series/infy_stock.csv",

                 usecols=['Date', 'Close'], parse_dates=['Date'],index_col='Date')

df.head()
df.info()
print("Min:",df.index.min())

print("Max:",df.index.max())
plt.figure(figsize=(17,5))

df.Close.plot()

plt.title("Closing Price",fontsize=20)

plt.show()
# The Split

plt.figure(figsize=(17,5))

stock_price = pd.concat([df.Close[:'2015-06-12']/2,df.Close['2015-06-15':]]) # adjustment

plt.plot(stock_price)

plt.title("Closing Price Adjusted",fontsize=20)

plt.show()
#helper function to plot the stock prediction

prev_values = stock_price.iloc[:180]

y_test = stock_price.iloc[180:]



def plot_pred(pred,title):

    plt.figure(figsize=(17,5))

    plt.plot(prev_values,label='Train')

    plt.plot(y_test,label='Actual')

    plt.plot(pred,label='Predicted')

    plt.ylabel("Stock prices")

    plt.title(title,fontsize=20)

    plt.legend()

    plt.show()
#Average of previous values

y_av = pd.Series(np.repeat(prev_values.mean(),68),index=y_test.index)

mse(y_av,y_test)
plot_pred(y_av,"Average")
weight = np.array(range(0,180))/180

weighted_train_data =np.multiply(prev_values,weight)



# weighted average is the sum of this weighted train data by the sum of the weight



weighted_average = sum(weighted_train_data)/sum(weight)

y_wa = pd.Series(np.repeat(weighted_average,68),index=y_test.index)



mse(y_wa,y_test)
plot_pred(y_wa,"Weighted Average")
y_train = stock_price[80:180]

y_test = stock_price[180:]

print("y train:",y_train.shape,"\ny test:",y_test.shape)
X_train = pd.DataFrame([list(stock_price[i:i+80]) for i in range(100)],

                       columns=range(80,0,-1),index=y_train.index)

X_test = pd.DataFrame([list(stock_price[i:i+80]) for i in range(100,168)],

                       columns=range(80,0,-1),index=y_test.index)



X_train
y_ma = X_test.mean(axis=1)

mse(y_ma,y_test)
plot_pred(y_ma,"Moving Average")
weight = np.array(range(1,81))/80

#weighted moving average

y_wma = X_test@weight/sum(weight)

mse(y_wma,y_test)
plot_pred(y_wma,"Weighted Moving Average")
from sklearn.linear_model import LinearRegression

lr=LinearRegression()



lr.fit(X_train,y_train)

y_lr = lr.predict(X_test)

y_lr = pd.Series(y_lr,index=y_test.index)



mse(y_test,y_lr)
plot_pred(y_lr,"Linear Regression")
weight = np.array(range(1,101))/100

wlr = LinearRegression()



wlr.fit(X_train,y_train,weight)

y_wlr = wlr.predict(X_test)

y_wlr = pd.Series(y_wlr,index=y_test.index)



mse(y_test,y_wlr)
plot_pred(y_wlr,"Weighted Linear Regression")
from sklearn.linear_model import Lasso

lasso = Lasso()



las = lasso.fit(X_train,y_train)

y_las = las.predict(X_test)

y_las = pd.Series(y_las,index = y_test.index)



mse(y_las,y_test)
plot_pred(y_las,"Lasso Regression")
from keras.models import Sequential

from keras.layers import Dense



#moving average Neural Network

ma_nn = Sequential([Dense(64,input_shape=(80,),activation='relu'), 

                    Dense(32,activation='linear'),Dense(1)])



ma_nn.compile(loss='mse',optimizer='rmsprop',metrics=['mae','mse'])



history = ma_nn.fit(X_train, y_train, epochs=250, batch_size=32, validation_split=0.25)
plt.plot(history.history['mse'],label='Training loss')

plt.plot(history.history['val_mse'], label='Validation loss')

plt.title("Mean Squared error")

plt.xlabel("Number of Epochs")

plt.legend()

plt.show()
loss_nn,mae_nn,mse_nn = ma_nn.evaluate(X_test,y_test)

print("\nloss:",loss_nn,"\nmae:",mae_nn,"\nmse:",mse_nn)
y_nn = ma_nn.predict(X_test)

y_nn = pd.Series(y_nn[:,0],index=y_test.index)

mse(y_nn,y_test)
plot_pred(y_nn,"Moving Average Prediction")