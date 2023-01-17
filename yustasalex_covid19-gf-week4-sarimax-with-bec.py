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
        
sorted(paths)
train_df = pd.read_csv(sorted(paths)[-1])
test_df = pd.read_csv(sorted(paths)[-2])
submission = pd.read_csv(sorted(paths)[-3])
sub_08_04 = pd.read_csv(sorted(paths)[-5])
train_df.head()
test_df.head()
submission.head()
sub_08_04.head()
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
delta = (datetime.today().date() - date(2020, 4, 2)).days
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
new_countries_idxs = [121, 166, 183, 210, 221, 299, 310]
days_from = (datetime.today().date() - date(2020, 4, 9)).days
sub_08_04[sub_08_04['Country_Region'] == 'Russia'].iloc[14:14+days_from]
train_df[train_df['Country_Region'] == 'Russia'].iloc[78:78+days_from]
sub_count_len = len(sub_08_04[sub_08_04['Country_Region'] == 'Russia'])
sub_0804_cc_preds = []
sub_0804_f_preds = []
sub_count = 0

for i in range(int(len(sub_08_04) / sub_count_len)):
    sub_0804_cc_preds.append(sub_08_04.ConfirmedCases[sub_count+14:sub_count+14+days_from].values.tolist())
    sub_0804_f_preds.append(sub_08_04.Fatalities[sub_count+14:sub_count+14+days_from].values.tolist())
    sub_count += sub_count_len
train_count_len = len(train_df[train_df['Country_Region'] == 'Russia'])
train_cc_act = []
train_f_act = []
train_count = 0

for i in range(int(len(train_df) / train_count_len)):
    train_cc_act.append(train_df.ConfirmedCases[train_count+78:train_count+78+days_from].values.tolist())
    train_f_act.append(train_df.Fatalities[train_count+78:train_count+78+days_from].values.tolist())
    train_count += train_count_len
train_0804_cc_act = []
train_0804_f_act = []

for i in range(len(train_cc_act)):
    if i not in new_countries_idxs:
        train_0804_cc_act.append(train_cc_act[i])
        train_0804_f_act.append(train_f_act[i])
div_act_sub_cc = []
div_act_sub_f = []

for i in range(len(train_0804_cc_act)):
    div_act_sub_cc.append([train_0804_cc_act[i][0] / sub_0804_cc_preds[i][0], 
                           train_0804_cc_act[i][1] / sub_0804_cc_preds[i][1]])
    div_act_sub_f.append([(train_0804_f_act[i][0]+1) / (sub_0804_f_preds[i][0]+1), 
                          (train_0804_f_act[i][1]+1) / (sub_0804_f_preds[i][1]+1)])
cc_becs = []
f_becs = []

for i in range(len(div_act_sub_cc)):
    cc_becs.append(np.mean(div_act_sub_cc[i]))
    f_becs.append(np.mean(div_act_sub_f[i]))
for idx in new_countries_idxs:
    cc_becs.insert(idx, 1)
    f_becs.insert(idx, 1)
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
mod_predicted_cc = []
mod_predicted_f = []
part = 1

for i in range(len(predicted_cc)):
    mod_predicted_cc.append([x*(cc_becs[i]*part) for x in predicted_cc[i]])
    mod_predicted_f.append([x*(f_becs[i]*part) for x in predicted_f[i]])
m_predicted_ConfirmedCases = []
m_predicted_Fatalities = []
for i in range(int(len(train_df) / count_len)):
    m_predicted_ConfirmedCases.append(train_cc[i][-delta:])
    m_predicted_ConfirmedCases.append(mod_predicted_cc[i])
    m_predicted_Fatalities.append(train_f[i][-delta:])
    m_predicted_Fatalities.append(mod_predicted_f[i])
    
m_predicted_ConfirmedCases = list(itertools.chain.from_iterable(m_predicted_ConfirmedCases))
m_predicted_Fatalities = list(itertools.chain.from_iterable(m_predicted_Fatalities))
mean_ConfirmedCases_preds = []
mean_Fatalities_preds = []

for i in range(int(len(predicted_ConfirmedCases))):
    mean_ConfirmedCases_preds.append(predicted_ConfirmedCases[i] * 0.15 + m_predicted_ConfirmedCases[i] * 0.85)
    mean_Fatalities_preds.append(predicted_Fatalities[i] * 0.15 + m_predicted_Fatalities[i] * 0.85)
submission['ConfirmedCases'] = predicted_ConfirmedCases
submission['Fatalities'] = predicted_Fatalities
submission1 =  pd.DataFrame(data=submission.ForecastId, columns=['ForecastId'])
submission1['ConfirmedCases'] = m_predicted_ConfirmedCases
submission1['Fatalities'] = m_predicted_Fatalities
submission2 =  pd.DataFrame(data=submission.ForecastId, columns=['ForecastId'])
submission2['ConfirmedCases'] = mean_ConfirmedCases_preds
submission2['Fatalities'] = mean_Fatalities_preds
concl_df = pd.read_csv(sorted(paths)[-2])
concl_feats = ['ForecastId', 'Country_Region', 'Date']
conclusion = pd.concat([concl_df[concl_feats], submission[['ConfirmedCases', 'Fatalities']]], axis=1)

concl1_df = pd.read_csv(sorted(paths)[-2])
conclusion1 = pd.concat([concl1_df[concl_feats], submission1[['ConfirmedCases', 'Fatalities']]], axis=1)

concl2_df = pd.read_csv(sorted(paths)[-2])
conclusion2 = pd.concat([concl2_df[concl_feats], submission2[['ConfirmedCases', 'Fatalities']]], axis=1)
region = 'Russia'
conclusion2[conclusion2['Country_Region'] == region]
import matplotlib.pyplot as plt

original_cc = conclusion[conclusion['Country_Region'] == region]['ConfirmedCases'].values
corrected_cc = conclusion1[conclusion1['Country_Region'] == region]['ConfirmedCases'].values
mean_cc = conclusion2[conclusion2['Country_Region'] == region]['ConfirmedCases'].values

original_f = conclusion[conclusion['Country_Region'] == region]['Fatalities'].values
corrected_f = conclusion1[conclusion1['Country_Region'] == region]['Fatalities'].values
mean_f = conclusion2[conclusion2['Country_Region'] == region]['Fatalities'].values

date = conclusion[conclusion['Country_Region'] == region]['Date'].values
for i in range(len(date)):
    date[i] = date[i][5:]

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,14), dpi=120)

ax1.plot(date, original_cc, ':g', label='original SARIMAX')
ax1.plot(date, corrected_cc, ':b', label='corrected SARIMAX')
ax1.plot(date, mean_cc, 'r', label='weighted SARIMAX')
ax2.plot(date, original_f, ':g', label='original SARIMAX')
ax2.plot(date, corrected_f, ':b', label='corrected SARIMAX')
ax2.plot(date, mean_f, 'r', label='weighted SARIMAX')

ax1.set_title(f'ConfirmedCases differences {region}')
ax1.set_xlabel('Date')
ax1.set_xticklabels(date, rotation=40)
ax1.set_ylabel('ConfirmedCases')
ax1.legend(loc='best')
ax1.grid()

ax2.set_title(f'Fatalities differences {region}')
ax2.set_xlabel('Date')
ax2.set_xticklabels(date, rotation=40)
ax2.set_ylabel('Fatalities')
ax2.legend(loc='best')
ax2.grid()

plt.show();
day_cc = conclusion2[conclusion2['Country_Region'] == region]['ConfirmedCases'].values

perday_cc = []
for i in range(1, len(day_cc)):
    perday_cc.append(day_cc[i] - day_cc[i-1])
    
perday_cc_df = pd.DataFrame(perday_cc, columns=['ConfirmedCases_per_day'])
perday_cc_df['Date'] = conclusion2[conclusion2['Country_Region'] == region]['Date'].values[1:]

perday_cc_df.plot(x='Date', y='ConfirmedCases_per_day', kind='bar', grid=True, figsize=(14, 6), title=region);
day_f = conclusion2[conclusion2['Country_Region'] == region]['Fatalities'].values

perday_f = []
for i in range(1, len(day_f)):
    perday_f.append(abs(day_f[i] - day_f[i-1]))
    
perday_f_df = pd.DataFrame(perday_f, columns=['Fatalities_per_day'])
perday_f_df['Date'] = conclusion2[conclusion2['Country_Region'] == region]['Date'].values[1:]

perday_f_df.plot(x='Date', y='Fatalities_per_day', kind='bar', grid=True, figsize=(14, 6), title=region);
submission2.to_csv('submission.csv', index=False)
submission2.head()