# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline  

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/database.csv', parse_dates=[['DATE','TIME']])

df.columns
injured = [x for x in df.columns if 'INJURED' in x]

killed = [x for x in df.columns if 'KILLED' in x]

df['severity'] = 'No injuries' 

df.loc[df[killed].sum(axis=1)>0,'severity'] = 'Fatal'

df.loc[(df[killed].sum(axis=1)==0)&(df[injured].sum(axis=1)>0),'severity'] = 'Injured'

df.severity.value_counts()
factors = [x for x in df.columns if 'FACTOR' in x]

fig, ax = plt.subplots(figsize=(4,8))

d = df.copy()

d = pd.DataFrame([d.loc[:,x].value_counts() for x in factors]).fillna(0).sum().sort_values(ascending=True)

d.plot(kind='barh',ax=ax)

ax.set_title('Causes of vehicle accidents in 2015, NYC')

ax.set_xlabel('# reported')
factors = [x for x in df.columns if 'FACTOR' in x]

d = df[factors].join(df.severity)

s0 = pd.DataFrame([d.loc[d.severity=='No injuries',x].value_counts() for x in factors]).fillna(0).sum().sort_values(ascending=False)

s1 = pd.DataFrame([d.loc[d.severity=='Injured',x].value_counts() for x in factors]).fillna(0).sum().sort_values(ascending=False)

s2 = pd.DataFrame([d.loc[d.severity=='Fatal',x].value_counts() for x in factors]).fillna(0).sum().sort_values(ascending=False)



fig = plt.figure(figsize=(9,3))

ax = fig.add_subplot(131 )

ax.set_title('No injuries')

s0[1:8].plot(kind='bar',ax=ax)

ax = fig.add_subplot(132 )

ax.set_title('Non fatal injuries')

s1[1:8].plot(kind='bar',ax=ax)

ax = fig.add_subplot(133 )

ax.set_title('Fatal Accidents')

s2[1:8].plot(kind='bar',ax=ax)
factors = [x for x in df.columns if 'FACTOR' in x]

d = (df[factors].join(df.severity).join(df.DATE_TIME)).set_index('DATE_TIME').copy()

d['HOURS'] = d.index.map(lambda t: t.hour)

hour_fatigue = d.loc[(d[factors]=='Fatigued/Drowsy').any(axis=1),:].groupby('HOURS').count().loc[:,factors].sum(axis=1)

# normalize by total accidents at that hour

hour_data = 100.0* hour_fatigue / d.groupby('HOURS').size()

fig, ax = plt.subplots(figsize=(6,4))

hour_data.plot(kind='bar',ax=ax)

ax.set_xlabel('Time of day')

ax.set_ylabel('% of tiredness related accidents')

ax.set_ylim(5)

ax.set_title('Fraction of Tiredness related accidents along the day')
vehicle_types = [x for x in df.columns if 'TYPE' in x]

fig, ax = plt.subplots(figsize=(4,8))

d = df.copy()

d = pd.DataFrame([d.loc[:,x].value_counts() for x in vehicle_types]).fillna(0).sum().sort_values(ascending=True)

d.plot(kind='barh',ax=ax)

ax.set_title('Involvement of vehicle types in accidents, 2015, NYC')

ax.set_xlabel('# reported')
vehicle_types = [x for x in df.columns if 'TYPE' in x]

d = df.copy()

d = pd.DataFrame([d.loc[:,x].value_counts() for x in vehicle_types]).fillna(0).sum().sort_values(ascending=True)

print("Ratio of taxis to normal cars involved in accidents: 1:{} ".format(int(d['PASSENGER VEHICLE']/d.TAXI)))