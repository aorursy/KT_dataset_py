import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=20,10

from keras.models import Sequential

from keras.layers import LSTM,Dropout,Dense

from sklearn.preprocessing import MinMaxScaler
df=pd.read_csv("../input/stock-ew/EW-MAX.csv")

df.head()
df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")

df.index=df['Date']

plt.figure(figsize=(16,8))

plt.plot(df["Close"],label='Close Price history')
data=df.sort_index(ascending=True,axis=0)

new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

for i in range(0,len(data)):

    new_dataset["Date"][i]=data['Date'][i]

    new_dataset["Close"][i]=data["Close"][i]