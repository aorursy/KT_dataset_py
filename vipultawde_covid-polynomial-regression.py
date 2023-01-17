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
import datetime

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_log_error

from math import sqrt

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv');

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv');
df_train['Date'] = pd.to_datetime(df_train['Date'], format = '%Y-%m-%d');

df_test['Date'] = pd.to_datetime(df_test['Date'], format = '%Y-%m-%d');
df_train.describe()
df_train['Country_Region'].value_counts()
grp_obj = df_train.groupby(['Country_Region','Province_State']);

df = pd.DataFrame(grp_obj.agg({'Date':['min','max','count']}))

df
print(df[('Date','min')].value_counts())

print(df[('Date','max')].value_counts())

print(df[('Date','count')].value_counts())
grp_obj = df_test.groupby(['Country_Region','Province_State']);

df = pd.DataFrame(grp_obj.agg({'Date':['min','max','count']}))

print(df[('Date','count')].value_counts())
# Combine Country and Province

def combine_country_province(df):

    df.loc[:,'Province_State'] = df['Province_State'].fillna("")

    df.loc[:,'Region'] = df['Country_Region'] + " " + df['Province_State']

    df.loc[:,'Region'] = df.loc[:,'Region'].str.strip();

    df = df.drop(

        labels=['Province_State','Country_Region'],

        axis='columns',

        inplace=False

        )

    return df;

    

df_train = combine_country_province(df_train);

df_test = combine_country_province(df_test);

df_train
def days_since_dataset(df):

    df['Date2'] = df['Date'] - datetime.datetime.strptime('22012020', "%d%m%Y")

    df['Date2'] = df['Date2'].dt.days;

    return df['Date2'];

    

df_train['Date2'] = days_since_dataset(df_train)

df_test['Date2'] = days_since_dataset(df_test)

df_train
temp = df_train['Region'].unique();

for idx, val in enumerate(temp):

    print(idx,val);
regions = df_train['Region'].unique();

regions = regions[65:68]



for idx, region in enumerate(regions):

    plt.figure(idx);

    f, ax = plt.subplots(1, 2, figsize=(20,5*1));

    text = "*"*10+'INDEX='+str(idx)+"*"*10+"REGION <---->"+region+"*"*10

    plt.figtext(.5,.9,text, fontsize=20, color='red', ha='center')

    df = df_train.loc[df_train['Region'] == region,:]

    sns.regplot(data = df, x='Date2', y='ConfirmedCases', ax=ax[0],order=4)

    sns.regplot(data = df, x='Date2', y='Fatalities', ax=ax[1],order=4)
def err_func(y_true,y_pred):

    msle =  mean_squared_log_error(y_true, y_pred);

    return sqrt(msle)
# Train Test Split

# n = len(df_train['Date2'].unique());

# print(n);

# train_bool = df_train['Date2'] < 70;



# train = df_train.loc[train_bool,:];

# valid = df_train.loc[~train_bool,:];



# train.shape, valid.shape
# PARAMS

degree = 4

# MODEL

poly = PolynomialFeatures(degree = degree, include_bias=False)

model1 = LinearRegression()

model2 = LinearRegression()
# X_cols = ['Date2'];

# y1_col = ['ConfirmedCases']

# y2_col = ['Fatalities']



# all_pred_train = pd.DataFrame();

# all_pred_valid = pd.DataFrame();



# regions = df_train['Region'].unique();

# regions = ['Japan','Portugal']

# for idx, region in enumerate(regions):

#     this_region_train = train['Region'] == region;

#     this_region_valid = valid['Region'] == region;

    

#     X0_train_iter = train.loc[this_region_train,X_cols];

#     y1_train_iter = train.loc[this_region_train,y1_col];

#     y2_train_iter = train.loc[this_region_train,y2_col];

    

#     X0_valid_iter = valid.loc[this_region_valid,X_cols];

#     y1_valid_iter = valid.loc[this_region_valid,y1_col];

#     y2_valid_iter = valid.loc[this_region_valid,y2_col];

    

#     X0_train_iter = poly.fit_transform(X0_train_iter);

#     X0_valid_iter = poly.fit_transform(X0_valid_iter);



#     model1.fit(X0_train_iter, y1_train_iter);

#     y1_train_iter_pred = model1.predict(X0_train_iter);

#     y1_valid_iter_pred = model1.predict(X0_valid_iter);



#     model2.fit(X0_train_iter, y2_train_iter);

#     y2_train_iter_pred = model2.predict(X0_train_iter);

#     y2_valid_iter_pred = model2.predict(X0_valid_iter);

    

#     pred_iter_train = pd.DataFrame({

#         'Id': train.loc[this_region_train,'Id'],

#         'ConfirmedCases': y1_train_iter_pred.reshape(-1),

#         'Fatalities': y2_train_iter_pred.reshape(-1)

#     })

#     all_pred_train = pd.concat([all_pred_train, pred_iter_train], axis = 0);

    

    

#     pred_iter_valid = pd.DataFrame({

#         'Id': valid.loc[this_region_valid,'Id'],

#         'ConfirmedCases': y1_valid_iter_pred.reshape(-1),

#         'Fatalities': y2_valid_iter_pred.reshape(-1)

#     })

#     all_pred_valid = pd.concat([all_pred_valid, pred_iter_valid], axis = 0);

    

# print(all_pred_train)

# print(all_pred_valid)
X_cols = ['Date2'];

y1_col = ['ConfirmedCases']

y2_col = ['Fatalities']



all_pred_train = pd.DataFrame();

all_pred_test = pd.DataFrame();



regions = df_train['Region'].unique();



train = df_train.copy();

test = df_test.copy();





for idx, region in enumerate(regions):

    scaler = StandardScaler();

    

    this_region_train = train['Region'] == region;

    this_region_test = test['Region'] == region;

    

    X0_train_iter = train.loc[this_region_train,X_cols];

    y1_train_iter = train.loc[this_region_train,y1_col];

    y2_train_iter = train.loc[this_region_train,y2_col];

    

    X0_test_iter = test.loc[this_region_test,X_cols];



    X0_train_iter = poly.fit_transform(X0_train_iter);

    X0_test_iter = poly.fit_transform(X0_test_iter);

    

    X0_train_iter = scaler.fit_transform(X0_train_iter)

    X0_test_iter = scaler.transform(X0_test_iter)

    

#     scaler_y1 = StandardScaler();

#     scaler_y1.fit_transform(y1_train_iter);

    

    model1.fit(X0_train_iter, y1_train_iter);

    y1_train_iter_pred = model1.predict(X0_train_iter);

    y1_test_iter_pred = model1.predict(X0_test_iter);



#     scaler_y2 = StandardScaler();

#     scaler_y2.fit_transform(y2_train_iter);

    

    model2.fit(X0_train_iter, y2_train_iter);

    y2_train_iter_pred = model2.predict(X0_train_iter);

    y2_test_iter_pred = model2.predict(X0_test_iter);

    

    pred_iter_train = pd.DataFrame({

        'Id': train.loc[this_region_train,'Id'],

        'ConfirmedCases': y1_train_iter_pred.reshape(-1),

        'Fatalities': y2_train_iter_pred.reshape(-1)

    })

    all_pred_train = pd.concat([all_pred_train, pred_iter_train], axis = 0);

    

    

    pred_iter_test = pd.DataFrame({

        'ForecastId': test.loc[this_region_test,'ForecastId'],

        'ConfirmedCases': y1_test_iter_pred.reshape(-1),

        'Fatalities': y2_test_iter_pred.reshape(-1)

    })

    all_pred_test = pd.concat([all_pred_test, pred_iter_test], axis = 0);

    

print(all_pred_train)

print(all_pred_test)
all_pred_test = all_pred_test.astype('int')



# all_pred_test.to_csv("submission.csv", index = False);
# answer = pd.merge(df_test,all_pred_test, left_on = 'ForecastId',right_on = 'ForecastId');



# answer['true_cc'] = -1;

# answer['true_fat'] = -1;



# train_max_date = train['Date2'].max()

# test_min_date = test['Date2'].min()

# for idx, region in enumerate(regions):

#     temp = train.loc[((train['Date2'] >= test_min_date) & (train['Region'] == region)),['ConfirmedCases','Fatalities']]

#     answer.loc[((answer['Date2'] <= train_max_date) & (answer['Region'] == region)),['true_cc','true_fat']] = temp

#     print(temp.info())

    



# answer2 = answer.copy()
all_pred_test.describe()
answer = pd.merge(df_test,all_pred_test, left_on = 'ForecastId',right_on = 'ForecastId');



train_max_date = train['Date2'].max()

test_min_date = test['Date2'].min()

for idx, region in enumerate(regions):

    sel1 = ((train['Date2'] >= test_min_date) & (train['Region'] == region));

    to_paste = train.loc[sel1,['ConfirmedCases','Fatalities']].copy();

    sel2 = ((answer['Date2'] <= train_max_date) & (answer['Region'] == region))

    answer.loc[sel2,['ConfirmedCases','Fatalities']] = to_paste.loc[:,['ConfirmedCases','Fatalities']].values;
answer = answer.loc[:,['ForecastId','ConfirmedCases','Fatalities']]

answer = answer.astype('int');

answer.to_csv("submission.csv", index = False);

answer.dtypes
# import pandas as pd

# a = pd.DataFrame({'c1':[1,2,3],'c2':[1,2,3]})

# b = pd.DataFrame({'c3':[82,73,77],'c4':[9,9,9]})



# a.loc[:,['c1','c2']] = b.loc[:,['c3','c4']].values;

# a