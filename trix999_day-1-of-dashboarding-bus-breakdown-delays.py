# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/bus-breakdown-and-delays.csv')

df.head()
# some info
df.info()
columns_of_interest = ['Occurred_On', 'How_Long_Delayed', 'Number_Of_Students_On_The_Bus', 'Has_Contractor_Notified_Schools', 'Has_Contractor_Notified_Parents','Reason']
last_incidents_df  = pd.DataFrame((df.sort_values(by=['Occurred_On'], ascending=False)[0:10][columns_of_interest]))

# let's remove the wrong first data
last_incidents_df[1:]
import seaborn as sns

df.Reason.unique()
addressable_problems = ['Mechanical Problem', 'Problem Run', 'Won`t Start', 'Flat Tire']

problematic_companies = pd.DataFrame(df.loc[df['Reason'].isin(addressable_problems)].groupby('Bus_Company_Name').Reason.count().sort_values(ascending=False), columns=['Reason'])
problematic_companies.reset_index(inplace=True)

ax = sns.catplot(x="Bus_Company_Name", y="Reason", kind="bar", data=problematic_companies[0:5], height=20, legend=False)
ax.set(xlabel='bus company', ylabel='number of addressable incidents')
