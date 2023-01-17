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
air_quality = pd.read_excel("/kaggle/input/airquality/AirQuality.xlsx")
type(air_quality)
air_quality.head()
air_quality.describe()
groups_state = air_quality.groupby('State')
groups_state.head()
groups_state.mean()
type(groups_state)
group_state_df = pd.DataFrame(groups_state)
type(group_state_df)
group_state_df.head()
air_quality_df = pd.DataFrame(air_quality)
air_quality_df.head()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
plt.hist(air_quality_df.Avg, histtype ='bar', rwidth=0.8)

plt.xlabel("AVERAGE")

plt.ylabel("COUNT")

plt.title("AVERAGE OF AIR QUALITY OF INDIA")

plt.show()
plt.figure(figsize=(17,7), dpi = 100)

sns.countplot(x='State',data=air_quality)

plt.xlabel('State')

plt.tight_layout()
plt.hist(air_quality_df.Max, histtype ='bar', rwidth=0.8)

plt.xlabel("MAX")

plt.ylabel("COUNT")

plt.title("MAX OF AIR QUALITY OF INDIA")

plt.show()
plt.hist(air_quality_df.Min, histtype ='bar', rwidth=0.8)

plt.xlabel("Min")

plt.ylabel("COUNT")

plt.title("Min OF AIR QUALITY OF INDIA")

plt.show()
air_quality['Pollutants'].value_counts().plot()

plt.xlabel("Pollutants")

plt.ylabel("COUNT")

plt.title("Pollutants OF AIR QUALITY OF INDIA")

plt.show()
air_quality['Pollutants'].value_counts().plot('bar')

plt.xlabel("Pollutants")

plt.ylabel("COUNT")

plt.title("Pollutants OF AIR QUALITY OF INDIA")

plt.show()
air_quality['lastupdate'].head()
air_quality.lastupdate.str.slice(-5, -3).astype(int).head()
air_quality['lastupdate'] = pd.to_datetime(air_quality.lastupdate)

air_quality.head()
air_quality.dtypes
## see the date in day of year 

air_quality.lastupdate.dt.dayofyear.head()
ts = pd.to_datetime('12-12-2018')
air_quality.loc[air_quality.lastupdate >= ts, :].head()
air_quality.head()
from matplotlib import pyplot

pyplot.plot(air_quality.State)

plt.xlabel("COUNT")

plt.ylabel("STATES")

plt.title("COUNTS OF STATES")

pyplot.show()
group_state = air_quality.groupby('State')
group_state.mean().head()
group_state.max().head()
list(air_quality['Pollutants'].unique())

pollutant = list(air_quality['Pollutants'].unique())

for poll in pollutant:

    plt.figure(figsize=(18,8), dpi = 100)

    sns.countplot(air_quality[air_quality['Pollutants'] == poll]['State'], data = air_quality)

    plt.tight_layout()

    plt.title(poll)
list(air_quality['State'].unique())