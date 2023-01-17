# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
day_ope=pd.read_csv("/kaggle/input/bus-operation/OperationPerformance_PerRoute-Day-SystemwideOps_8_2-15_2020.csv")

avg_ope=pd.read_csv("/kaggle/input/bus-operation/OperationPerformance_PerRoute-SystemwideOps_8_2-15_2020.csv")
day_ope.head()
avg_ope.head()
print(day_ope.shape)

print(avg_ope.shape)
print(day_ope.columns)

print(avg_ope.columns)
print(day_ope.info())

print(avg_ope.info())
day_ope.describe()
avg_ope.describe()
avg_ope['Line'].unique
avg_ope.sort_values(by='Sched. distance', ascending=False).loc[:,['Line','Sched. distance']]
avg_ope.sort_values(by='Act. distance', ascending=False).loc[:,['Line','Act. distance']]
avg_ope.sort_values(by='Sched.Trip(cnt)', ascending=False).loc[:,['Line','Sched.Trip(cnt)']]
avg_ope.sort_values(by='Act.Trip(cnt)', ascending=False).loc[:,['Line','Act.Trip(cnt)']]
day_ope[day_ope['Date'].apply(lambda x: x == '8/3/2020')].loc[:,['Line','Date','Act. distance','Dist. dev.']]
sns.barplot(day_ope['Date'][:10],day_ope['Dist. dev.'][:10],orient='v')