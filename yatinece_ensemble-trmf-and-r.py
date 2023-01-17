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

def RMSLE(pred,actual):

    return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))

# Any results you write to the current directory are saved as output.
sub1=pd.read_csv('/kaggle/input/rscript-week4/submission.csv')

sub4=pd.read_csv('/kaggle/input/trmfdata/submission.csv')

sub4.sort_values("ForecastId", inplace=True)

sub1.sort_values("ForecastId", inplace=True)

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

dacheckv='2020-04-01'

train['Province_State'].fillna('', inplace=True)

train['Date'] = pd.to_datetime(train['Date'])

train['day'] = train.Date.dt.dayofyear

#train = train[train.day <= 85]

train['geo'] = ['_'.join(x) for x in zip(train['Country_Region'], train['Province_State'])]

train=train[train.Date>dacheckv]



test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

test['Province_State'].fillna('', inplace=True)

test['Date'] = pd.to_datetime(test['Date'])

test['day'] = test.Date.dt.dayofyear

#train = train[train.day <= 85]

test['geo'] = ['_'.join(x) for x in zip(test['Country_Region'], test['Province_State'])]

test1=test.merge(train,on=['Country_Region','Province_State','Date'],how='inner')

print(RMSLE(sub1[sub1.ForecastId.isin(test1.ForecastId.values)]['ConfirmedCases'].values,test1['ConfirmedCases'].values))



print(RMSLE(sub1[sub1.ForecastId.isin(test1.ForecastId.values)]['Fatalities'].values,test1['Fatalities'].values))

print(RMSLE(sub4[sub4.ForecastId.isin(test1.ForecastId.values)]['ConfirmedCases'].values,test1['ConfirmedCases'].values))



print(RMSLE(sub4[sub4.ForecastId.isin(test1.ForecastId.values)]['Fatalities'].values,test1['Fatalities'].values))

sub1=sub1.reset_index(drop=True)

sub4=sub4.reset_index(drop=True)



sub_df = sub1.copy()

TARGETS = ["ConfirmedCases", "Fatalities"]

for t in TARGETS:

    sub_df[t] = sub1[t]*0.4 + sub4[t]*0.6

sub_df.to_csv("submission.csv", index=False)

print(RMSLE(sub_df[sub_df.ForecastId.isin(test1.ForecastId.values)]['ConfirmedCases'].values,test1['ConfirmedCases'].values))



print(RMSLE(sub_df[sub_df.ForecastId.isin(test1.ForecastId.values)]['Fatalities'].values,test1['Fatalities'].values))