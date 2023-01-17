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
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
import seaborn as sns
import datetime
df = pd.read_csv("../input/sandp500/all_stocks_5yr.csv")
df.head()
df["date"] = pd.to_datetime(df["date"] ,format='%Y-%m-%d')
print('Min date from data set: %s' % df['date'].min())
print('Max date from data set: %s' % df['date'].max())
df = df.set_index("date")
df.head()
df_apple = df[df.Name=="AAPL"]

df_google = df[df.Name=="GOOGL"]
df_amazon = df[df.Name=="AMZN"]

df_microsoft = df[df.Name=="MSFT"]
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
generator = TimeseriesGenerator(df_microsoft["close"], df_microsoft["close"], length=10, batch_size=1)
X = []
Y = []
for i in range(len(generator)):
    x_gen, y_gen = generator[i]
    X.append(x_gen)
    Y.append(y_gen)
X =np.array(X)
Y = np.array(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25)
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(1, 10)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, Y_train, epochs=500)
predictions = model.predict(X)

model.evaluate(X_test,Y_test)
