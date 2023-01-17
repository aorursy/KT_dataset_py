import numpy as np 

import pandas as pd

from datetime import datetime, date, time



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from fbprophet import Prophet

from sklearn.metrics import *
df = pd.read_csv('../input/predict-train-occupancy-time-series/Occupancy_train.csv')[['Time','Total Occupancy rate (percent)']]
df['Time'] = df['Time'].apply(lambda x: datetime.strptime(x, "%YM%m"))

df.columns = ['ds','y']
df.plot(x='ds', y='y', kind="line")
test_val = 12

df_train = df.iloc[:-test_val]

df_test = df.iloc[-test_val:]

m = Prophet()

m.fit(df_train)
future = m.make_future_dataframe(periods=12, freq='m')

forecast = m.predict(future)
fig1 = m.plot(forecast)
pred = forecast.iloc[-test_val:]['yhat']

df_test = df_test['y']
r2_score(df_test,pred)
fig2 = m.plot_components(forecast)