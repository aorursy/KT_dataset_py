# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
file_clean_complete = "/kaggle/input/corona-virus-report/covid_19_clean_complete.csv"
file_day_wise = "/kaggle/input/corona-virus-report/day_wise.csv"
file_county_wise = "/kaggle/input/corona-virus-report/usa_county_wise.csv"
file_worldometer_data = "/kaggle/input/corona-virus-report/worldometer_data.csv"
file_full_grouped = "/kaggle/input/corona-virus-report/full_grouped.csv"
file_country_wise_latest = "/kaggle/input/corona-virus-report/country_wise_latest.csv"
csv_clean_complete = pd.read_csv(file_clean_complete)
csv_day_wise = pd.read_csv(file_day_wise)
csv_county_wise = pd.read_csv(file_county_wise)
csv_worldometer_data = pd.read_csv(file_worldometer_data)
csv_full_grouped = pd.read_csv(file_full_grouped)
csv_country_wise_latest = pd.read_csv(file_country_wise_latest)
csv_clean_complete.head()
csv_day_wise.head()
csv_county_wise.head()
csv_worldometer_data.head()
csv_full_grouped.head()
csv_country_wise_latest.head()
#pd.date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed=None)
pd.date_range('2017/07/07', '2018/07/07', freq='BMS')
csv_day_wise_copy = csv_day_wise[:]
csv_day_wise_copy.set_index(csv_day_wise_copy['Date'])
csv_day_wise_copy.set_index(csv_day_wise_copy['Date'])['2020-01-01':'2020-02-01']
csv_day_wise_copy.set_index(csv_day_wise_copy['Date'])['2020-01-01':'2020-02-01']
csv_day_wise_copy.set_index(csv_day_wise_copy['Date'])[:'2020-02']
csv_day_wise_copy.set_index(csv_day_wise_copy['Date'])['2020-01-01':'2020-02-01']
csv_full_grouped_copy = csv_full_grouped[:]
#csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date'])
csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date']).loc['2020-01-22'].count()
csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date']).loc['2020-01-22']
csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date']).loc['2020-07-27']
np.sum(csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date'])['2020-03':'2020-04']['Confirmed'])
csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date'])
csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date'])
csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date']).loc['2020-01-22']
csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date']).loc['2020-02-01']
pd.tseries.offsets.Week(1)+pd.tseries.offsets.Hour(8)
csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date']).loc[[str(d).split()[0] for d in pd.date_range('2020-01-22', periods=2, freq='W')]]
csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date']).loc[[str(v).split()[0] for v in pd.date_range('2020-01-22', periods=2, freq='W').to_list()]]
[str(v).split()[0] for v in pd.date_range('2020-01-22', periods=2, freq='W').to_list()]
select_range = [str(v).split()[0] for v in pd.date_range('2020-01-22', periods=2, freq='W').to_list()]
csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date']).loc[select_range[0]:select_range[1]]
s1=pd.DataFrame({'open':np.random.randn(8), 'close':np.random.randn(8)}, index=pd.date_range('2020-01-01', '2020-01-08', freq='D'))
s1
s1.shift(2)
s1.shift(-2, freq='1D')
sample = csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date']).loc['2020-02-01':'2020-02-10'][['WHO Region','Confirmed','Deaths','Recovered','Active','New cases','New deaths', 'New recovered']]
sample
s1=pd.DataFrame({'Confirmed':np.random.randn(8), 'Deaths':np.random.randn(8)}, index=pd.date_range('2020-01-01', '2020-01-08', freq='D'))
s1
s2=pd.DataFrame({'Confirmed':np.random.randn(4), 'Deaths':np.random.randn(4)}, index=pd.date_range('2020-01-02', '2020-01-05', freq='D'))
s2
s1+s2
s2.asfreq(freq='D')
s3=pd.DataFrame({'Confirmed':np.random.randn(4), 'Deaths':np.random.randn(4)}, index=['2020-01-02', '2020-01-03','2020-01-05','2020-01-06'])
s3
s3.asfreq(freq='D')
sample = csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date']).loc['2020-02-01':'2020-02-10'][['WHO Region','Confirmed','Deaths','Recovered','Active','New cases','New deaths', 'New recovered']]
sample.index=(sample['WHO Region'])
sample.drop('WHO Region', axis=1, inplace=True)
sample
sample = csv_full_grouped_copy.set_index(csv_full_grouped_copy['Date']).loc['2020-02-01'][['Country/Region','Confirmed','Deaths','Recovered','Active','New cases','New deaths', 'New recovered']]
sample.index=(sample['Country/Region'])
sample.drop('Country/Region', axis=1, inplace=True)
sample
csv_full_grouped_copy.groupby(csv_full_grouped_copy['Country/Region'])
csv_full_grouped_copy
{v:copy_samples.loc[v]['Confirmed'] for v in copy_samples.index}
copy_samples = csv_full_grouped_copy[:]
samples_group = copy_samples.groupby(['Country/Region'],as_index=False).agg(
    {'Confirmed': 'sum','Deaths':'sum','Recovered':'sum',
     'Active':'sum','New cases':'sum','New deaths':'sum','New recovered':'sum'})
samples_group
samples_group_with_total = samples_group.append([{"Country/Region":"World", 
        "Confirmed":np.sum(samples_group['Confirmed']), 
        "Deaths":np.sum(samples_group['Deaths']),
        "Recovered":np.sum(samples_group['Recovered']), 
        "Active":np.sum(samples_group['Active']),
        "New cases":np.sum(samples_group['New cases']), 
        "New deaths":np.sum(samples_group['New deaths']),
        "New recovered":np.sum(samples_group['New recovered'])
    }])

samples_group_with_total.index = samples_group_with_total['Country/Region']

samples_group_with_total
copy_samples = csv_full_grouped_copy[:]
samples_group = copy_samples.groupby(['Date'],as_index=False).agg(
    {'Confirmed': 'sum','Deaths':'sum','Recovered':'sum',
     'Active':'sum','New cases':'sum','New deaths':'sum','New recovered':'sum'})
samples_group.index = samples_group['Date']
samples_group
newtable = pd.DataFrame({'Confirmed Diff':{samples_group.index[i]:samples_group['Confirmed'].iloc[i]-samples_group['Confirmed'].iloc[i-1] if i>0 else samples_group['Confirmed'].iloc[i] for i in range(0,len(samples_group['Confirmed']))}})
samples_group_new = pd.concat([samples_group, newtable],axis=1)
samples_group_new
# let's start to plot datas
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

#plt.plot(samples_group_new['Confirmed'], label='Confirmed')
plt.plot(samples_group_new['Confirmed Diff'], label='Confirmed Difference')

plt.legend(loc='upper left', frameon=True)
plt.xticks([xtick for xtick in range(samples_group_new['Date'].count())],list(samples_group_new['Date']), rotation=40)

fig=plt.gcf()
fig.set_size_inches(18.5, 10.5)

plt.grid()
plt.show()

