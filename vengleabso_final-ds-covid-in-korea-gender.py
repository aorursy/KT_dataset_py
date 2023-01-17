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
#         print(os.path.join(dirname, filename))
        print(f"{filename} = pd.read_csv('{os.path.join(dirname, filename)}')")
    

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
matplotlib.rc('figure', figsize=(10, 5))
# PatientInfo = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')
# Region = pd.read_csv('/kaggle/input/coronavirusdataset/Region.csv')
# SearchTrend = pd.read_csv('/kaggle/input/coronavirusdataset/SearchTrend.csv')
TimeGender = pd.read_csv('/kaggle/input/coronavirusdataset/TimeGender.csv')
# Weather = pd.read_csv('/kaggle/input/coronavirusdataset/Weather.csv')
# Case = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')
# Time = pd.read_csv('/kaggle/input/coronavirusdataset/Time.csv')
# TimeProvince = pd.read_csv('/kaggle/input/coronavirusdataset/TimeProvince.csv')
# TimeAge = pd.read_csv('/kaggle/input/coronavirusdataset/TimeAge.csv')
# Policy = pd.read_csv('/kaggle/input/coronavirusdataset/Policy.csv')
# PatientRoute = pd.read_csv('/kaggle/input/coronavirusdataset/PatientRoute.csv')
# SeoulFloating = pd.read_csv('/kaggle/input/coronavirusdataset/SeoulFloating.csv')
TimeGender = pd.read_csv('/kaggle/input/coronavirusdataset/TimeGender.csv', parse_dates=['date'])
TimeGender.head()
latest_data = TimeGender[TimeGender['date']==max(TimeGender['date'])]
latest_data
latest_data['Mortality Rate'] = ((latest_data['deceased'] / latest_data['confirmed']) * 100).round(3).astype(str) + '%'
latest_data[['sex','Mortality Rate']].head()
((latest_data['deceased'] / latest_data['confirmed']) * 100).round(3).mean()
TimeGender['yesterday'] = TimeGender['date'] - timedelta(days=1)
TimeGender.head()
TimeGender = TimeGender.merge(TimeGender, left_on=['yesterday','sex'], right_on=['date','sex'], suffixes=('_curr','_prev'))
TimeGender.head()
TimeGender['increase'] = TimeGender['confirmed_curr'] - TimeGender['confirmed_prev']
TimeGender.head()
male_data = TimeGender[TimeGender['sex'] == 'male']
female_data = TimeGender[TimeGender['sex'] == 'female']

plt.plot(female_data['date_curr'], female_data['increase'], label="Female")
plt.plot(male_data['date_curr'] , male_data['increase'], label="Male")
plt.xlabel('Date')
plt.ylabel('Number of Increase by each gender')
plt.legend(loc='best')
plt.show()
plt.xlabel('Date')
plt.ylabel('Number of Increase')
plt.legend(loc='best')
plt.stackplot(male_data['date_curr'], female_data['increase'], male_data['increase'], labels=['Male','Female'])
plt.legend(loc='upper left')

plt.show()
plt.plot(female_data['date_curr'], female_data['confirmed_curr'], label="Female")
plt.plot(male_data['date_curr'] , male_data['confirmed_curr'], label="Male")
plt.xlabel('Date')
plt.ylabel('Number of Confirmed by each gender')
plt.legend(loc='best')
plt.show()
plt.stackplot(male_data['date_curr'] , male_data['confirmed_curr'], female_data['confirmed_curr'], labels=["Male","Female"])
plt.xlabel('Date')
plt.ylabel('Number of Confirmed')
plt.legend(loc='upper left')
plt.show()
plt.plot(female_data['date_curr'], female_data['deceased_curr'], label="Female")
plt.plot(male_data['date_curr'] , male_data['deceased_curr'], label="Male")
plt.xlabel('Date')
plt.ylabel('Number of decease by each gender')
plt.legend(loc='best')
plt.show()
plt.stackplot(male_data['date_curr'] , male_data['deceased_curr'], female_data['deceased_curr'], labels=["Male",'Female'])
plt.xlabel('Date')
plt.ylabel('Number of decrease')
plt.legend(loc='best')
plt.show()
