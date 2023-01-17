import numpy as np

import pandas as pd

import matplotlib

from matplotlib import pyplot as plt

import matplotlib.ticker as ticker

import sklearn

from sklearn.linear_model import LinearRegression

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score

import scipy.stats as ss

import seaborn as sns
# read data

df_x=pd.read_excel("../input/dataset/yield.xlsx")

df_x=df_x.dropna()

df_y=pd.read_csv("../input/dataset/spf500.csv")

df_y=df_y.dropna()
dateSPF500=[str(x) for x in df_y["Date"]]

dateBidYield=[str(x) for x in df_x["Date"]]

closePrice=[float(x) for x in df_y["Close"]]

bidYield=[float(x) for x in df_x["Bid Yield"]]
dateStock=[]

dateYield=[]

x_temp=[] # list of bid yield

y_temp=[] # list of close price
# match Yield Date to SPF+500 Date (since we need monthly data) 

for i in range(0,len(dateSPF500)):

	for j in range(len(dateBidYield)-1,0,-1):

		dateBidYield[j]=dateBidYield[j].replace(' 00:00:00','')

		if dateSPF500[i]==dateBidYield[j]:

			dateStock.append(dateSPF500[i])

			dateYield.append(dateBidYield[j])

			y_temp.append(closePrice[i])

			x_temp.append(bidYield[j])
# index of 2009: 24

# index of 2014: -31

# get data from 2009 - 2014

x_temp=x_temp[24:-31]

y_return=y_temp[24:-31]

x=np.array(x_temp)

y=np.array(y_return)
# regression

model = LinearRegression()

model.fit(x.reshape(len(x),1) , y) # Train

print(model.intercept_, model.coef_)

w1=model.coef_[0]

w0=model.intercept_



# linear Regression Prediction

features=pd.DataFrame(x)

target=pd.DataFrame(y)



split_num = int(len(features)*0.7) # Find 70% position



train_x = features[:split_num] # Get features of training set

train_y = target[:split_num] # Get target of training set



test_x = features[split_num:] # Get features of testing set

test_y = target[split_num:] # Get target of testing set



model = LinearRegression() # Establish the model

model.fit(train_x, train_y) # Train

model.coef_, model.intercept_ # Print the parameters

print(model.intercept_, model.coef_)

w1=model.coef_[0]

w0=model.intercept_



preds = model.predict(test_x) # Train and predict

#print(preds) # Show results
# calculate MAE and MSE

def mae_value(y_true, y_pred):

    n = len(y_true)

    mae = sum(np.abs(y_true - y_pred))/n

    return mae



def mse_value(y_true, y_pred):

    n = len(y_true)

    mse = sum(np.square(y_true - y_pred))/n

    return mse



mae = mae_value(test_y.values, preds)

mse = mse_value(test_y.values, preds)

print("MAE: ", mae)

print("MSE: ", mse)
# plot

plt.scatter(x, y)

plt.xlabel("bid yield")

plt.ylabel("Stock Price")
# fitting line

x_temp = np.linspace(0,10,100) # Plot testing points

plt.scatter(x, y)

plt.plot(x, x*w1 + w0, 'r')

plt.show() 
# trend chat

date1=np.array(dateStock[24:-31])

date2=np.array(dateYield[24:-31])



fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(111)

ax1.plot(date1, y)

ax1.set_ylabel('Stock Price')



ax2 = ax1.twinx()  

ax2.plot(date2, x, 'r')

ax2.set_ylabel('Bid Yield')



tick_spacing = 3

ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.show()