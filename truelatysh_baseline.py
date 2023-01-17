import sys

sys.path.append('/kaggle/input/ts-course')
!pip install -r /kaggle/input/ts-course/requirements.txt
import matplotlib.pyplot as plt

import pandas as pd

from model import TimeSeriesPredictor

import plotting
%matplotlib inline
ts = pd.read_csv('/kaggle/input/sberbank-timeseries-challenge/train.csv')

test = pd.read_csv('/kaggle/input/sberbank-timeseries-challenge/test.csv', index_col='time', parse_dates=True)
ts.head()
test.head()
ts = pd.Series(data=ts.value.values, index=pd.to_datetime(ts.time))
ts.head()
ts[-100:].plot()
predictor = TimeSeriesPredictor()
predictor.fit(ts)
n_hours = int((test.index[-1] - test.index[0]).total_seconds() / 3600) + 1
n_hours
ts_pred = predictor.predict_next(ts, k=n_hours)
plotting.plot_multiple_ts(ts[-500:], ts_pred[:500])
ts_pred.head()
answer = test.join(ts_pred.to_frame(), how='left')
answer.columns = ['value']
answer.to_csv('answer.csv')