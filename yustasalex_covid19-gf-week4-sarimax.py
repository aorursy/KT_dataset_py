import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
from sklearn.metrics import mean_squared_log_error
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


paths = []

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        #print(os.path.join(dirname, filename))
        
sorted(paths)
train_df = pd.read_csv(sorted(paths)[2])
test_df = pd.read_csv(sorted(paths)[1])
submission = pd.read_csv(sorted(paths)[0])
train_df.head()
test_df.head()
submission.head()
from statsmodels.tsa.statespace.sarimax import SARIMAX
count_len = len(train_df[train_df['Country_Region'] == 'Russia'])

train_cc = []
train_f = []
count = 0
for i in range(int(len(train_df) / count_len)):
    train_cc.append(train_df.ConfirmedCases[count:count+count_len].values.tolist())
    train_f.append(train_df.Fatalities[count:count+count_len].values.tolist())
    count += count_len
from datetime import date
delta = (date(2020, 4, 15) - date(2020, 4, 2)).days
test_count = len(test_df[test_df['Country_Region'] == 'Russia']) - delta - 1
predicted_cc = []

for i in range(len(train_cc)):
    try:
        data1 = train_cc[i]
        model1 =  SARIMAX(data1, order=(1,1,0), seasonal_order=(1,1,0,12), measurement_error=True)
        model1_fit = model1.fit(disp=False)
        predicted1 = model1_fit.predict(len(data1), len(data1)+test_count)
        predicted_cc.append(predicted1.tolist())
    except:
        data1 = train_cc[i]
        model1 =  SARIMAX(data1,order=(1,1,0),seasonal_order=(1,1,0,12),measurement_error=True,enforce_stationarity=False)
        model1_fit = model1.fit(disp=False)
        predicted1 = model1_fit.predict(len(data1), len(data1)+test_count)
        predicted_cc.append(predicted1.tolist())
predicted_f = []
for i in range(len(train_f)):
    try:
        data2 = train_f[i]
        model2 =  SARIMAX(data2,order=(1,1,0), seasonal_order=(1,1,0,12), measurement_error=True)
        model2_fit = model2.fit(disp=False)
        predicted2 = model2_fit.predict(len(data2), len(data2)+test_count)
        predicted_f.append(predicted2.tolist())
    except:
        data2 = train_f[i]
        model2 =  SARIMAX(data2,order=(1,1,0),seasonal_order=(1,1,0,12),measurement_error=True,enforce_stationarity=False)
        model2_fit = model2.fit(disp=False)
        predicted2 = model2_fit.predict(len(data2), len(data2)+test_count)
        predicted_f.append(predicted2.tolist())
check_lenght = len(train_cc[0][-delta:]) + len(predicted_cc[0])
if check_lenght == 43:
    print('Check OK')
else:
    print('Check failed')
import itertools

predicted_ConfirmedCases = []
predicted_Fatalities = []
for i in range(int(len(train_df) / count_len)):
    predicted_ConfirmedCases.append(train_cc[i][-delta:])
    predicted_ConfirmedCases.append(predicted_cc[i])
    predicted_Fatalities.append(train_f[i][-delta:])
    predicted_Fatalities.append(predicted_f[i])
    
predicted_ConfirmedCases = list(itertools.chain.from_iterable(predicted_ConfirmedCases))
predicted_Fatalities = list(itertools.chain.from_iterable(predicted_Fatalities))
submission['ConfirmedCases'] = predicted_ConfirmedCases
submission['Fatalities'] = predicted_Fatalities
submission.to_csv('submission.csv', index=False)
submission.head()
concl_df = pd.read_csv(sorted(paths)[1])
concl_feats = ['ForecastId', 'Country_Region', 'Date']
conclusion = pd.concat([concl_df[concl_feats], submission[['ConfirmedCases', 'Fatalities']]], axis=1)
region = 'Russia'
conclusion[conclusion['Country_Region'] == region]
day_cc = conclusion[conclusion['Country_Region'] == region]['ConfirmedCases'].values

perday_cc = []
for i in range(1, len(day_cc)):
    perday_cc.append(day_cc[i] - day_cc[i-1])
    
perday_cc_df = pd.DataFrame(perday_cc, columns=['CC_per_day'])
perday_cc_df['Date'] = conclusion[conclusion['Country_Region'] == region]['Date'].values[1:]

perday_cc_df.plot(x='Date', y='CC_per_day', kind='bar', grid=True, figsize=(14, 6), title=region);
