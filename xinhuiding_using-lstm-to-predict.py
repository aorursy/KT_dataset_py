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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('../input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')
df.head()
bj_data=df[df['Province/State'].isin(['Beijing'])]
bj_data=bj_data.drop(['Province/State','Country/Region','Lat','Long'],axis=1)
df=pd.DataFrame(bj_data)
df=df.T
#convert argument to datetime
df.index=pd.to_datetime(df.index)
df.info()
#split train and test set
train,test=df[:-13],df[-13:]
scaler=MinMaxScaler()
scaler.fit(train)
train=scaler.transform(train)
test=scaler.transform(test)
n_input=12
n_features=1
generator=TimeseriesGenerator(train,train,length=n_input,batch_size=6)
model=Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_input,n_features)))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

model.fit_generator(generator,epochs=300)
pred_list=[]
batch=train[-n_input:].reshape((1,n_input,n_features))
for i in range(n_input):
  pred_list.append(model.predict(batch)[0])
  batch=np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
df_predict=pd.DataFrame(scaler.inverse_transform(pred_list),index=df[-n_input:].index,columns=['Predictions'])
df_test=pd.concat([df,df_predict],axis=1)
df_test.tail()
plt.figure(figsize=(20,5))
plt.plot(df_test.index,df_test[50])
plt.plot(df_test.index,df_test['Predictions'],color='r')
train=df
scaler.fit(train)
train=scaler.transform(train)
n_input=12
n_features=1
generator=TimeseriesGenerator(train,train,length=n_input,batch_size=6)
model.fit_generator(generator,epochs=300)

pred_list=[]
batch=train[-n_input:].reshape((1,n_input,n_features))
for i in range(n_input):
  pred_list.append(model.predict(batch)[0])
  batch=np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
from pandas.tseries.offsets import DateOffset
add_dates=[df.index[-1]+DateOffset(days=x) for x in range (0,13)]
future_dates=pd.DataFrame(index=add_dates[1:], columns=df.columns)
df_predict=pd.DataFrame(scaler.inverse_transform(pred_list),index=future_dates[-n_input:].index,columns=['Prediction'])
df_proj=pd.concat([df,df_predict],axis=1)
plt.figure(figsize=(10,4))
plt.plot(df_test.index,df_test[50])
plt.plot(df_proj.index,df_proj['Prediction'],color='r')
plt.legend(loc='best',fontsize='large')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()