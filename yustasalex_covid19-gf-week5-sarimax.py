import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

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
train_df = pd.read_csv(sorted(paths)[2])

test_df = pd.read_csv(sorted(paths)[1])

submission = pd.read_csv(sorted(paths)[0])
train_df.head()
neg_val_count = 0

for val in train_df.TargetValue:

    if val < 0.0:

        neg_val_count += 1

        

neg_val_count
train_df['CPSCR'] = train_df[['County', 'Province_State', 'Country_Region']].apply(lambda x: str(x[0])+'_'+str(x[1])+'_'+str(x[2]), axis=1)
cpscr = list(train_df['CPSCR'].unique())

single_reg_len = len(train_df[train_df['CPSCR'] == cpscr[0]])
for idx in range(len(cpscr)):

    if len(train_df[train_df['CPSCR'] == cpscr[idx]]) != single_reg_len:

        print(cpscr[idx])
test_df.head()
single_reg_len_test = len(test_df[test_df['Country_Region'] == 'Afghanistan'])
test_df['CPSCR'] = test_df[['County', 'Province_State', 'Country_Region']].apply(lambda x: str(x[0])+'_'+str(x[1])+'_'+str(x[2]), axis=1)
for idx in range(len(cpscr)):

    if len(test_df[test_df['CPSCR'] == cpscr[idx]]) != single_reg_len_test:

        print(cpscr[idx])
submission.head(6)
int(len(submission)/3), int(len(test_df))
from statsmodels.tsa.statespace.sarimax import SARIMAX
train_ConfirmedCases = list(train_df[train_df['Target'] == 'ConfirmedCases']['TargetValue'].values)

train_Fatalities = list(train_df[train_df['Target'] == 'Fatalities']['TargetValue'].values)
count_len = int(len(train_df[train_df['CPSCR'] == 'nan_nan_Afghanistan']) / 2)



train_cc = []

train_f = []

count = 0

for i in range(int(int(len(train_df) / 2) / count_len)):

    train_cc.append(train_ConfirmedCases[count:count+count_len])

    train_f.append(train_Fatalities[count:count+count_len])

    count += count_len
from datetime import date

delta = (datetime.today().date() - date(2020, 4, 27)).days
test_count = int(len(test_df[test_df['CPSCR'] == 'nan_nan_Afghanistan']) / 2) - delta - 1

predicted_cc = []



for i in range(len(train_cc)):

    try:

        data1 = train_cc[i]

        #model1 =  SARIMAX(data1, order=(1,1,0), seasonal_order=(1,1,0,12), measurement_error=True)

        model1 =  SARIMAX(data1, order=(1,0,0), seasonal_order=(0,1,1,12), measurement_error=True)

        model1_fit = model1.fit(disp=False)

        predicted1 = model1_fit.predict(len(data1), len(data1)+test_count)

        predicted_cc.append(predicted1.tolist())

    except:

        data1 = train_cc[i]

        #model1 =  SARIMAX(data1,order=(1,1,0),seasonal_order=(1,1,0,12),measurement_error=True,enforce_stationarity=False)

        model1 =  SARIMAX(data1,order=(1,0,0),seasonal_order=(0,1,1,12),measurement_error=True,enforce_stationarity=False)

        model1_fit = model1.fit(disp=False)

        predicted1 = model1_fit.predict(len(data1), len(data1)+test_count)

        predicted_cc.append(predicted1.tolist())
predicted_f = []

for i in range(len(train_f)):

    try:

        data2 = train_f[i]

        #model2 =  SARIMAX(data2,order=(1,1,0), seasonal_order=(1,1,0,12), measurement_error=True)

        model2 =  SARIMAX(data2, order=(1,0,0), seasonal_order=(0,1,1,12), measurement_error=True)

        model2_fit = model2.fit(disp=False)

        predicted2 = model2_fit.predict(len(data2), len(data2)+test_count)

        predicted_f.append(predicted2.tolist())

    except:

        data2 = train_f[i]

        #model2 =  SARIMAX(data2,order=(1,1,0),seasonal_order=(1,1,0,12),measurement_error=True,enforce_stationarity=False)

        model2 =  SARIMAX(data2,order=(1,0,0),seasonal_order=(0,1,1,12),measurement_error=True,enforce_stationarity=False)

        model2_fit = model2.fit(disp=False)

        predicted2 = model2_fit.predict(len(data2), len(data2)+test_count)

        predicted_f.append(predicted2.tolist())
check_lenght = len(train_cc[0][-delta:]) + len(predicted_cc[0])

if check_lenght == 45:

    print('Check OK')

else:

    print('Check failed')
import itertools



predicted_ConfirmedCases = []

predicted_Fatalities = []

for i in range(int(int(len(train_df) / 2) / count_len)):

    predicted_ConfirmedCases.append(train_cc[i][-delta:])

    predicted_ConfirmedCases.append(predicted_cc[i])

    predicted_Fatalities.append(train_f[i][-delta:])

    predicted_Fatalities.append(predicted_f[i])

    

predicted_ConfirmedCases = list(itertools.chain.from_iterable(predicted_ConfirmedCases))

predicted_Fatalities = list(itertools.chain.from_iterable(predicted_Fatalities))
predicted_ConfirmedCases = np.clip(predicted_ConfirmedCases, 0.0, None)

predicted_Fatalities = np.clip(predicted_Fatalities, 0.0, None)
int(len(predicted_ConfirmedCases) * 3 * 2), int(len(submission))
def get_preds(cc_list, f_list):

    assert len(cc_list) == len(f_list), 'len cc != len f'

    TargetValue = []

    for i in range(len(cc_list)):

        TargetValue.append(np.quantile([0, cc_list[i]], 0.07))

        TargetValue.append(np.quantile([0, cc_list[i]], 0.52))

        TargetValue.append(np.quantile([0, cc_list[i]], 0.97))

        TargetValue.append(np.quantile([0, f_list[i]], 0.07))

        TargetValue.append(np.quantile([0, f_list[i]], 0.52))

        TargetValue.append(np.quantile([0, f_list[i]], 0.97))

    return TargetValue
TargetValue = get_preds(predicted_ConfirmedCases, predicted_Fatalities)
submission['TargetValue'] = TargetValue
submission.to_csv('submission.csv', index=False)
submission.head()
concl = test_df[test_df['Target'] == 'ConfirmedCases'][['CPSCR', 'Date']]

concl['ConfirmedCases'] = predicted_ConfirmedCases

concl['Fatalities'] = predicted_Fatalities

concl.head()
region = cpscr[2]



reg_concl = concl[concl['CPSCR'] == region]

reg_concl.plot(x='Date', y='ConfirmedCases', kind='bar', grid=True, figsize=(14, 6), title=region);
reg_concl.plot(x='Date', y='Fatalities', kind='bar', grid=True, figsize=(14, 6), title=region);