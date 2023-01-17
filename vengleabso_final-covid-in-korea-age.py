# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime, timedelta

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
matplotlib.rc('figure', figsize=(12, 9))
TimeAge = pd.read_csv('/kaggle/input/coronavirusdataset/TimeAge.csv',parse_dates=['date'])

TimeAge.head(100)
# TimeAge['date'].diff().rolling(window=7).mean()
time_age_data = TimeAge.pivot(columns='age', index='date', values=['confirmed','deceased'] )

time_age_data.head()
confirmed_data = time_age_data['confirmed']

plt.plot(confirmed_data, linestyle='-')

plt.ylabel('Number of Confirmed Case per age group')

plt.legend(list(confirmed_data.columns),loc='upper left')

plt.show()
plt.stackplot(confirmed_data.index,confirmed_data.values.T )

plt.ylabel('Number of Confirmed Case')

plt.legend(list(confirmed_data.columns),loc='upper left')

plt.show()
decreased_data = time_age_data['deceased']

plt.plot(decreased_data)

plt.ylabel('Number of Deceased Case per age group')

plt.legend(list(decreased_data.columns),loc='upper left')

plt.show()
plt.stackplot(decreased_data.index,decreased_data.values.T )

plt.ylabel('Number of Deceased Case')

plt.legend(list(decreased_data.columns),loc='upper left')

plt.show()
group_date = TimeAge.groupby(['date'])[['confirmed', 'deceased']].sum()

group_date['mortality_rate_as_percent'] = (  group_date['deceased'] / group_date['confirmed'] * 100).round(3).astype(str) + '%'

group_date['mortality_rate'] = (  group_date['deceased'] / group_date['confirmed'] * 100).round(3)

group_date
mortality_rate = group_date['mortality_rate']

plt.plot(mortality_rate)

plt.show()
mortality_rate.describe()
group_age = TimeAge.groupby(['age'])[['confirmed', 'deceased']].sum()

group_age['mortality_rate'] = (  group_age['deceased'] / group_age['confirmed'] * 100).round(3).astype(str) + '%'

group_age
mortality_rate.hist()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

lin = LinearRegression()

ml_data = mortality_rate.tail(30);



ml_data.isnull().values.any()
x = ((ml_data.index - ml_data.index[0]).days.values).reshape(-1,1) 

y = ml_data.values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.80, random_state=42)

lin.fit(X_train, y_train)
predict = lin.predict(((mortality_rate.index - mortality_rate.index[0]).days.values).reshape(-1,1))
next_3_day = mortality_rate.tail(3).index + timedelta(days=3)

result = lin.predict((next_3_day - mortality_rate.index[0]).days.values.reshape(-1,1))
plt.plot(mortality_rate.index, predict,mortality_rate )

plt.plot(next_3_day, result,'brown',linestyle=':')

plt.legend(['regression','mortality_rate', 'predict' ],loc='upper left')

plt.show()