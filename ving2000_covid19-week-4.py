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
## Load in data

## Imports



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv', index_col = 'Id')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv', index_col = 'ForecastId')



train.head()
test.head()
## Combine 2 province and country columns



train['Province_State'].fillna(" ", inplace = True)

train.rename(columns = {'Country_Region': 'Country'}, inplace = True)

train['Country_Region'] = train.apply(lambda x: (x['Country'] + "_" + str(x['Province_State'] )) 

                                              if x['Province_State'] != " "

                                              else x['Country'], axis = 1)



test['Province_State'].fillna(" ", inplace = True)

test.rename(columns = {'Country_Region': 'Country'}, inplace = True)

test['Country_Region'] = test.apply(lambda x: (x['Country'] + "_" + str(x['Province_State']) 

                                              if x['Province_State'] != " "

                                              else x['Country']), axis = 1)

train.head()
## Change data types



#train['Date'] = pd.to_datetime(train['Date'])

#test['Date'] = pd.to_datetime(test['Date'])



print('There are ' + str(len(train['Country_Region'].unique())) + ' countries reported in this dataset')

print('Date from ' + str(min(train['Date'])) + '. And ends on ' + str(max(train['Date'])))
## Plot world cases and fatalities



tot_cases = train.groupby(['Date'], as_index = False).sum()



plt.figure(figsize = (10,8))

ax = sns.lineplot(x = 'Date', y = 'ConfirmedCases', data = tot_cases, ci = False)

sns.lineplot(x = 'Date', y = 'Fatalities', data = tot_cases, ax = ax)
def show_values_on_bars(axs):

    def _show_on_single_plot(ax):        

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = '{:.2f}'.format(p.get_height())

            ax.text(_x, _y, value, ha="center") 



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)
gb_cntry = train.groupby(['Country', 'Date'], as_index = False).sum()

gb_cntry['Mortality_rate'] = gb_cntry['Fatalities'] / gb_cntry['ConfirmedCases'] 

gb_cntry['Mortality_rate'].fillna(0, inplace = True)



## Plot 5 countries with the most cases as of last day in training set



plt.figure(figsize = (10, 8))

ax = sns.barplot(x = 'Country', y = 'ConfirmedCases',

            data = gb_cntry[gb_cntry['Date'] == max(gb_cntry['Date'])].sort_values(['ConfirmedCases'], ascending = False)[:5])

plt.title('Number of COVID19 cases in top5 most infected countries')



show_values_on_bars(ax)
## Plot the mortality rates in top3 countries

## Notice: different y-scales



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20,8))



gb_cntry[gb_cntry['Country'] == 'US'].plot(x = 'Date', y = 'Mortality_rate', ax = ax1)

ax1.set_title('Mortality rate in the US')



gb_cntry[gb_cntry['Country'] == 'Spain'].plot(x = 'Date', y = 'Mortality_rate', ax = ax2, color = 'r')

ax2.set_title('Mortality rate in Spain')



gb_cntry[gb_cntry['Country'] == 'United Kingdom'].plot(x = 'Date', y = 'Mortality_rate', ax = ax3, color = 'g')

ax3.set_title('Mortality rate in UK')
## Plot the number of cases in top3 countries



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20,8), sharey = 'all')



gb_cntry[gb_cntry['Country'] == 'US'].plot(x = 'Date', y = 'ConfirmedCases', ax = ax1)

ax1.set_title('Number of cases in the US')



gb_cntry[gb_cntry['Country'] == 'Spain'].plot(x = 'Date', y = 'ConfirmedCases', ax = ax2, color = 'r')

ax2.set_title('Number of cases in Italy')



gb_cntry[gb_cntry['Country'] == 'United Kingdom'].plot(x = 'Date', y = 'ConfirmedCases', ax = ax3, color = 'g')

ax3.set_title('Number of cases in UK')
## EXTERNAL DATA



df_temp = pd.read_csv('/kaggle/input/weather-data-for-covid19-data-analysis/training_data_with_weather_info_week_4.csv')

df_temp.head()
df_temp.columns
## Match df_temp with train data



df_temp['country+province'] = df_temp['country+province'].apply(lambda x: x[:-1] if x[-1] == '-' else x.replace('-', '_'))

set(df_temp['country+province'].unique()) - set(train['Country_Region'].unique()) 
## Rename df_temp columns



df_temp.rename(columns = {'Country_Region': 'Country'}, inplace = True)

df_temp.rename(columns = {'country+province': 'Country_Region'}, inplace = True)



df_temp.columns
## Merge df_temp and train data



df_temp1 = df_temp[['Date', 'Country_Region', 'temp', 'rh', 'wdsp']]



train1 = train.copy().reset_index()



data = train1.merge(df_temp1, how = 'left', on = ['Country_Region', 'Date'], left_index = True)

data.head()
## Does not cover all dates in test set



print(df_temp['Date'].max())

print(test['Date'].min())
## Function to extract day, week, and num of day since first day recorded



import re



def add_datepart (df, field_name):

    fld = df[field_name]

    targ_pre = re.sub('[Dd]ate$', '', field_name)

    for n in ('Year', 'Month', 'Week', 'Day'):

        df[targ_pre + n] = getattr(fld.dt, n.lower())

    df[targ_pre + 'Elapsed'] = (fld - fld.min()).dt.days
## Take a subset of the data for testing model purposes



data['Date'] = pd.to_datetime(data['Date'])

us_va = data[data['Country_Region'] == 'US_Virginia']



print(f'Size of dataset {len(us_va)}')

add_datepart(us_va, 'Date')

us_va = us_va.set_index('Date')

us_va
# us_va['Prev_cases'] = [us_va['ConfirmedCases'][n-1] for n in range(us_va.shape[0])]

# us_va['Prev_cases'][0] = 0.0
## Transform X and fill in NAs





from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer





class ColumnChooser (BaseEstimator, TransformerMixin):

    

    def __init__ (self, columns):

        self.columns = columns

    def fit(self, X, y = None):

        return self

    def transform(self, X, y = None):

        return X[self.columns]

    





pre_pipeline = Pipeline([

    ('choose_cols', ColumnChooser(['temp', 'rh', 'wdsp', 'Elapsed'])),

    ('fill_na', SimpleImputer(strategy = 'median')),

    ('scaler', StandardScaler())

])

X_train = pre_pipeline.fit_transform(us_va)

y_cases = us_va['ConfirmedCases']

y_fatal = us_va['Fatalities']
from sklearn.linear_model import LinearRegression



lm = LinearRegression()

lm.fit(X_train, y_cases)

lm.score(X_train, y_cases)
def display_scores(scores):

    print('Scores: ', scores)

    print('Mean: ', scores.mean())

    print('Standard deviation: ', scores.std())
from sklearn.metrics import mean_squared_error



lm_pred = lm.predict(X_train)

print(np.sqrt(mean_squared_error(y_cases, lm_pred)))



from sklearn.model_selection import cross_val_score, cross_val_predict



lm_scores = cross_val_score(lm, X_train, y_cases, cv = 10, scoring = 'neg_mean_squared_error')

display_scores(np.sqrt(-lm_scores))
lm_fatal = LinearRegression()

lm_fatal.fit(X_train, y_fatal)

lm_fatal_pred = lm_fatal.predict(X_train)

print(np.sqrt(mean_squared_error(us_va['Fatalities'], lm_fatal_pred)))



lm_fatal_scores = cross_val_score(lm_fatal, X_train, y_fatal, cv = 10, scoring = 'neg_mean_squared_error')

display_scores(np.sqrt(-lm_fatal_scores))
from sklearn.linear_model import SGDRegressor



sgd = SGDRegressor(max_iter = 2000)

sgd.fit(X_train, y_cases)

sgd_pred = sgd.predict(X_train)

print(np.sqrt(mean_squared_error(y_cases, sgd_pred)))



sgd_scores = cross_val_score(sgd, X_train, y_cases, cv = 10, scoring = 'neg_mean_squared_error')

display_scores(np.sqrt(-sgd_scores))
sgd_fatal = SGDRegressor()

sgd_fatal.fit(X_train, y_fatal)

sgd_fatal_pred = sgd_fatal.predict(X_train)

print(np.sqrt(mean_squared_error(y_fatal, sgd_fatal_pred)))



sgd_fatal_scores = cross_val_score(sgd, X_train, y_fatal, cv = 10, scoring = 'neg_mean_squared_error')

display_scores(np.sqrt(-sgd_fatal_scores))
import math



for i in zip(y_fatal, lm_fatal_pred):

    print(f'Actual: {i[0]}, Predicted: {i[1]}, Off_by: {i[0] - i[1]}')
cases_pred = []

fatal_pred = []



for i in data['Country_Region'].unique():

    

    area = data[data['Country_Region'] == i]

    

    add_datepart(area, 'Date')

    area = area.set_index('Date')

    area['Prev_cases'] = [area['ConfirmedCases'][n-1] for n in range(area.shape[0])]

    area['Prev_cases'][0] = 0.0

    

    X_train = pre_pipeline.fit_transform(area[['Prev_cases', 'rh', 'temp', 'Elapsed']])

    y_cases = area['ConfirmedCases']

    y_fatal = area['Fatalities']

    

    lm_cases = LinearRegression()

    lm_cases.fit(X_train, y_cases)

    lm_cases_pred = lm_cases.predict(X_train).tolist()

    cases_pred += lm_cases_pred

    

    lm_fatal = LinearRegression()

    lm_fatal.fit(X_train, y_fatal)

    lm_fatal_pred = lm_fatal.predict(X_train).tolist()

    fatal_pred += lm_fatal_pred
actual = []

for i in data['Country_Region'].unique():

    new = data[data['Country_Region'] == i]['ConfirmedCases'].tolist()

    actual += new
cases_pred
cases_pred = [0 if i < 0 else math.floor(i) for i in cases_pred]

fatal_pred = [0 if i < 0 else math.floor(i) for i in fatal_pred]



data['Predicted cases'] = cases_pred

data['Predicted fatalities'] = fatal_pred

data
np.sqrt(mean_squared_error(data['ConfirmedCases'], data['Predicted cases']))
for i in zip(actual[:100], cases_pred[:100]):

    print(f'Actual: {i[0]}, Predicted: {i[1]}, Off_by: {i[0] - i[1]}')
np.sqrt(mean_squared_error(actual, cases_pred))
test1 = test.merge(df_temp1, how = 'left', on = ['Country_Region', 'Date'])

test1['Date'] = pd.to_datetime(test1['Date'])
test1
test1[['temp','rh','wdsp']].isnull().sum()
final_cases_pred = []

final_fatal_pred = []



for i in test1['Country_Region'].unique():

    

    area = test1[test1['Country_Region'] == i]

    

    add_datepart(area, 'Date')

    area = area.set_index('Date')

    area['Prev_cases'] = [area['ConfirmedCases'][n-1] for n in range(area.shape[0])]

    area['Prev_cases'][0] = 0.0

    

    X_train = pre_pipeline.fit_transform(area[['Prev_cases', 'rh', 'temp', 'Elapsed']])

    y_cases = area['ConfirmedCases']

    y_fatal = area['Fatalities']

    

    lm_cases = LinearRegression()

    lm_cases.fit(X_train, y_cases)

    lm_cases_pred = lm_cases.predict(X_train).tolist()

    final_cases_pred += lm_cases_pred

    

    lm_fatal = LinearRegression()

    lm_fatal.fit(X_train, y_fatal)

    lm_fatal_pred = lm_fatal.predict(X_train).tolist()

    final_fatal_pred += lm_fatal_pred