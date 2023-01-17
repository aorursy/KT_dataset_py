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

train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv',sep=',')

test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv',sep=',')

submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv',sep=',')

print(train.head(2))

print(test.head(2))
print("Training data: \n", train.count(), "\n\nTest data: \n", test.count())

print("\n\nTraining Missing data: \n",  train.isnull().sum(), "\n\nTest Missing data: \n", test.isnull().sum())
train = train.drop(['Province/State'],axis=1)

test = test.drop(['Province/State'],axis=1)

train.dtypes
def create_time_features(df):

    """

    Creates time series features from datetime index

    """

    df['date'] = df.index

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear

    

    X = df[['dayofweek','quarter','month','year',

           'dayofyear','dayofmonth','weekofyear']]

    return X
train["iDate"] = train["Date"].apply(lambda x: x.replace("-",""))

train["iDate"]  = train["iDate"].astype(int)

test["iDate"] = test["Date"].apply(lambda x: x.replace("-",""))

test["iDate"]  = test["iDate"].astype(int)



train['Date'] = pd.to_datetime(train['Date'])

test['Date']= pd.to_datetime(test['Date'])

train = train.set_index(['Date'])

test = test.set_index(['Date'])
create_time_features(train)

create_time_features(test)


train.head(10)
train['ConfirmedCases_today'] = train.groupby(['Country/Region'])['ConfirmedCases'].diff(1)

train['Fatalities_today'] = train.groupby(['Country/Region'])['Fatalities'].diff(1)
train['ConfirmedCases_today'] = train['ConfirmedCases_today'].fillna(0)

train['Fatalities_today'] = train['Fatalities_today'].fillna(0)



print(train.groupby(['Country/Region'])['ConfirmedCases_today'].sum())

print(train.groupby(['Country/Region'])['Fatalities_today'].sum())

train.isnull().sum()
input_cols = ["Lat","Long", "iDate"] # 'dayofweek','quarter','month','year', 'dayofyear','dayofmonth','weekofyear'] #,

output_cols = ["ConfirmedCases","Fatalities"]

ids = submission["ForecastId"]

extra_feat = ["ConfirmedCases_today", "Fatalities_today"]
X = train[input_cols]

Y1 = train[output_cols[0]]

Y2 = train[output_cols[1]]

X_test = test[input_cols]

E1 = train[extra_feat[0]]

E2 = train[extra_feat[1]]

XE = train[input_cols+extra_feat]
#Modeling with Extra Features Prediction



import xgboost as xgb 

from xgboost import plot_importance, plot_tree 



model= xgb.XGBRegressor(n_estimators=1000)

#Training for cases



model.fit(X,E1) 

ep1 = model.predict(X_test)



# Training for deaths

model.fit(X,E2) 

ep2 = model.predict(X_test)

import xgboost as xgb

from xgboost import plot_importance, plot_tree

model= xgb.XGBRegressor(n_estimators=1000)

# Training for cases

model.fit(X,E1)

ep1 = model.predict(X_test)

preds = np.array(ep1)

preds[preds < 0] = 0

preds = np.round(preds, 0)



# Training for deaths

model.fit(X,E2)

ep2 = model.predict(X_test)

prds = np.array(ep2)

prds[prds < 0] = 0

prds = np.round(prds, 0)


test['ConfirmedCases_today'] = preds #.replace(np.inf, 0) # if ep1 > 0 else test['ConfirmedCases_today'] = 0 # ['ConfirmedCases_today_prediction']

test['Fatalities_today'] = prds #.replace(np.inf, 0) #['Fatalities_today_prediction']
test.dtypes
XE_test = test[input_cols+extra_feat]
tree_reg= xgb.XGBRegressor(n_estimators=1000)



# Training for cases

tree_reg.fit(XE,Y1)

prd = tree_reg.predict(XE_test)



# Training for deaths

tree_reg.fit(XE,Y2)

prd_2 = tree_reg.predict(XE_test)



output_tree_reg = pd.DataFrame({ 'ForecastId' : ids, 'ConfirmedCases': prd,'Fatalities':prd_2 })

output_tree_reg.to_csv('submission.csv', index=False)

""" scored: .46566 with ["Lat","Long", "iDate", 'dayofweek','quarter','month','year', 'dayofyear','dayofmonth','weekofyear']"""

""" scored:   with ["Lat","Long", "iDate"] """