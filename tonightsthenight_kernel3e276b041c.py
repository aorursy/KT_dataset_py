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
import seaborn as sns
df=pd.read_csv('/kaggle/input/covid19-17thmarch/newdata.csv')
df.head()
cntr = df[['Country','Confirmed','Deaths','Recovered']]
cntr= cntr.groupby(['Country'])['Confirmed','Deaths','Recovered'].sum()
cntr=cntr.sort_values(by='Confirmed',ascending=False)
cntrtop5 = cntr.head()

sns.barplot(cntrtop5['Confirmed'],cntrtop5.index,orient='h')
#Create calculations for active cases and closed cases
df['Active'] = df['Confirmed'] - (df['Deaths']+df['Recovered'])
df['Closed'] = df['Deaths'] + df['Recovered']
df.head()
#Top 5 Countries by Active Cases
cntra = df[['Country','Active','Closed','Deaths','Confirmed','Recovered']]
cntra = cntra.groupby(['Country'])['Active','Closed','Deaths','Confirmed','Recovered'].sum()
cntra = cntra.sort_values(by='Active',ascending=False)
cntratop5 = cntra.head()
cntratop5.index
sns.barplot(cntratop5['Active'],cntratop5.index,orient='h')
#Mortality Rate for Closed cases and Overall cases (closed+active)
cntra['Mortality Rate Closed'] = cntra['Deaths']/cntra['Closed']
cntra['Mortality Rate Overall'] = cntra['Deaths']/cntra['Confirmed']
#Top 5 countries with highest mortality rate for closed cases
# with at least 10k confirmed cases
cntratleast10k = cntra[(cntra['Confirmed']>=10000) & (cntra['Recovered']>=1)]
cntrtop5mortcl = cntratleast10k.sort_values(by='Mortality Rate Closed',ascending=False).head()
sns.barplot(cntrtop5mortcl['Mortality Rate Closed']*100,cntrtop5mortcl.index,orient='h')
