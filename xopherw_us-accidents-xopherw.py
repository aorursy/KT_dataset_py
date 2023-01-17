# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# pip install plotly==3.10.0

from collections import Counter
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt # datetime 
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly 
from scipy import stats


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
use_cols = ['Start_Time', 'Severity', 'County', 'State', 'Weather_Condition', 'Visibility(mi)', 'Humidity(%)']
raw = pd.read_csv(r'/kaggle/input/us-accidents/US_Accidents_Dec19.csv')
df =  pd.read_csv(r'/kaggle/input/us-accidents/US_Accidents_Dec19.csv', usecols = use_cols)
df.info()
df[['date','time']] = df['Start_Time'].str.split(' ', expand = True)
df['date'] = pd.to_datetime(df['date'])
date = df['date']
group1 = df.groupby(pd.DatetimeIndex(date).year).Severity.value_counts()
np.log10(group1).unstack(fill_value=0).plot.bar()

df['YYYY-MM'] = df.date.dt.strftime("%Y-%m")

group2 = df.groupby(df['YYYY-MM']).Severity.value_counts()
group2.unstack(fill_value=0).plot.line()

humidity = df['Humidity(%)'].value_counts().sort_index()
y = humidity
x = range(1,101)
slope, intercept, r_value, p_value, std_err = stats.linregress(range(1, 101), y)
r_value**2
plt.plot(x, intercept + slope*x, 'r', label='fitted line')
plt.plot(humidity)
plt.show()
df.time.value_counts().sort_index().plot.line()
