import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime

import pytz

import seaborn as sns

from sklearn import linear_model

from sklearn.tree import DecisionTreeRegressor

import os





%matplotlib inline
import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)
data = pd.read_csv("../input/SolarEnergy/SolarPrediction.csv")

data.head()
data.info()
data = data.sort_values(by='UNIXTime', ascending=True).reset_index(drop=True)

data.head()
HItz = pytz.timezone(zone='US/Hawaii')



datetimeHI = data['UNIXTime'].apply(lambda x: 

                                 datetime.datetime.utcfromtimestamp(x).replace(tzinfo=pytz.utc).astimezone(HItz))

data['DatetimeHI'] = datetimeHI

datetimeHI.head()
data['DatetimeHI'], data['Radiation'].iplot(kind='line',

    xTitle='Date',

    linecolor='black',

    yTitle='Radiation',

    title='Radiation between Sept 2016 and Dec 2016')
weekendmarker = datetime.datetime(2016,9, 8).replace(tzinfo=HItz)

weekonedata = data[data['DatetimeHI'] < weekendmarker]

# print(weekonedata)

weekonedata['DatetimeHI'], weekonedata['Radiation'].iplot(kind='line',

    xTitle='Date',

    linecolor='black',

    yTitle='Radiation',

    title='Radiation 9/01/2016 - 9/08/2016')
weekstartmarker = datetime.datetime(2016,9, 25).replace(tzinfo=HItz)

weekendspet=datetime.datetime(2016,9, 30).replace(tzinfo=HItz)

weeklastdata1 = data[data['DatetimeHI'] > weekstartmarker]

weeklastdata2 = weeklastdata1[weeklastdata1['DatetimeHI'] <= weekendspet]

weeklastdata2['DatetimeHI'], weeklastdata2['Radiation'].iplot(kind='line',

    xTitle='Date',

    linecolor='black',

    yTitle='Radiation',

    title='Radiation 9/26/2016 - 9/30/2016')
def week_plot(ax, dates, col, colname):



    if colname == 'Radiation':

        plt_color = 'Red'

    elif colname == 'Pressure':

        plt_color = 'Turquoise'

    elif colname == 'WindDirection(Degrees)':

        plt_color = 'DarkGray'

    elif colname == 'Temperature':

        plt_color = 'Gold'

    elif colname == 'Humidity':

        plt_color = 'SaddleBrown'   

    else:

        plt_color = 'Turquoise'

    

    ax.plot(dates, col, c=plt_color)

    

    ax.set_title(f'{colname} 9/1/2016 - 9/8/2016')

    ax.set_ylabel(f'{colname} Level')

    ax.set_xlabel('Date')

    

    for tick in ax.get_xticklabels():

        tick.set_rotation(45)

    

    return ax
plt.rcParams['figure.figsize'] = 16, 12



ts_cols = ['Radiation', 'Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed']

fig, axes = plt.subplots(len(ts_cols), sharex=True)

for i, ax in enumerate(axes):

    ax = week_plot(ax, weekonedata['DatetimeHI'], weekonedata[ts_cols[i]], ts_cols[i])



fig.tight_layout()
def date_plot(ax, data, colname, start_dt, end_dt):



    subset_data = data[(data['DatetimeHI'] > start_dt) & (data['DatetimeHI'] < end_dt)]

    dates = subset_data['DatetimeHI']

    col = subset_data[colname]

    

    srt = start_dt.strftime('%m/%d/%Y')

    end = end_dt.strftime('%m/%d/%Y')

    

    if colname == 'Radiation':

        plt_color = 'Red'

    elif colname == 'Pressure':

        plt_color = 'Turquoise'

    elif colname == 'WindDirection(Degrees)':

        plt_color = 'DarkGray'

    elif colname == 'Temperature':

        plt_color = 'Gold'

    elif colname == 'Humidity':

        plt_color = 'SaddleBrown'   

    else:

        plt_color = 'Turquoise'

    

    ax.plot(dates, col, c=plt_color)

    

    ax.set_title(f'{colname} {srt} - {end}')

    ax.set_ylabel(f'{colname} Level')

    ax.set_xlabel('Date')

    

    for tick in ax.get_xticklabels():

        tick.set_rotation(45)

    

    return ax
fig, axes = plt.subplots(len(ts_cols), sharex=True)

for i, ax in enumerate(axes):

    ax = date_plot(

        ax,

        data,

        ts_cols[i],

        datetime.datetime(2016,11, 27).replace(tzinfo=HItz),

        datetime.datetime(2016,12, 15).replace(tzinfo=HItz),

    )

fig.tight_layout()
fig, axes = plt.subplots(len(ts_cols), sharex=True)

for i, ax in enumerate(axes):

    ax = date_plot(

        ax,

        data,

        ts_cols[i],

        datetime.datetime(2016,9, 8).replace(tzinfo=HItz),

        datetime.datetime(2016,10, 3).replace(tzinfo=HItz),

    )

fig.tight_layout()
from datetime import datetime

from pytz import timezone

import pytz

hawaii= timezone('US/Hawaii')

data.index =  pd.to_datetime(data['UNIXTime'], unit='s')

data.index = data.index.tz_localize(pytz.utc).tz_convert(hawaii)

data['MonthOfYear'] = data.index.strftime('%m').astype(int)

data['DayOfYear'] = data.index.strftime('%j').astype(int)

data['WeekOfYear'] = data.index.strftime('%U').astype(int)

data['TimeOfDay(h)'] = data.index.hour

data['TimeOfDay(m)'] = data.index.hour*60 + data.index.minute

data['TimeOfDay(s)'] = data.index.hour*60*60 + data.index.minute*60 + data.index.second



data.drop(['Data','Time','TimeSunRise','TimeSunSet'], inplace=True, axis=1)

data.head()
print(data.index)
corrmat = data.drop(['TimeOfDay(h)', 'TimeOfDay(m)', 'TimeOfDay(s)', 'UNIXTime', 'MonthOfYear', 'WeekOfYear'], inplace=False, axis=1)

corrmat = corrmat.corr()

f, ax = plt.subplots(figsize=(7,7))

sns.heatmap(corrmat, vmin=-.8, vmax=.8, square=True, cmap = 'coolwarm')

plt.show()
X = data[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'DayOfYear', 'TimeOfDay(s)']]

y = data['Radiation']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

regressor = RandomForestRegressor(n_estimators = 100)

regressor.fit(X_train, y_train)

feature_importances = regressor.feature_importances_



X_train_opt = X_train.copy()

removed_columns = pd.DataFrame()

models = []

r2s_opt = []



for i in range(0,5):

    least_important = np.argmin(feature_importances)

    removed_columns = removed_columns.append(X_train_opt.pop(X_train_opt.columns[least_important]))

    regressor.fit(X_train_opt, y_train)

    feature_importances = regressor.feature_importances_

    accuracies = cross_val_score(estimator = regressor,

                                 X = X_train_opt,

                                 y = y_train, cv = 5,

                                 scoring = 'r2')

    r2s_opt = np.append(r2s_opt, accuracies.mean())

    models = np.append(models, ", ".join(list(X_train_opt)))

    

feature_selection = pd.DataFrame({'Features':models,'r2 Score':r2s_opt})

feature_selection.head()