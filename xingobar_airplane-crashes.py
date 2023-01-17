# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter 

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv')
df.head()
operator_count = Counter(df['Operator'].dropna()).most_common(10)

operator_keys = [operator[0] for operator in operator_count]

operator_val = [operator[1] for operator in operator_count]



fig,ax = plt.subplots(figsize = (8,6))

sns.barplot(x = operator_keys, y = operator_val)

plt.title('Top 10 operator')

plt.ylabel('Count')

plt.xlabel('Operator')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
operator_fatal = df[['Operator','Fatalities']].groupby(['Operator']).sum()

operator_fatal = operator_fatal['Fatalities'].sort_values(ascending=False)[:10]

operator_fatal_keys = operator_fatal.index

operator_fatal_val = operator_fatal.values

fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x = operator_fatal_keys,y =operator_fatal_val)

plt.title('Operator vs Fatalities')

plt.xlabel('Operator')

plt.ylabel('Total Fatalities')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].map(lambda x:x.year)
table_count = df.groupby([df['Year']])['Fatalities'].size()



year = table_count.index

table_count_val = table_count.values

fig,ax = plt.subplots(figsize=(15,6))

sns.barplot(x = year , y = table_count_val)

plt.title('Fatalities per year')

plt.xlabel('Year')

plt.ylabel('Count')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
table_count = df.groupby([df['Year']])['Fatalities'].sum()



year = table_count.index

table_count_val = table_count.values

fig,ax = plt.subplots(figsize=(15,6))

sns.barplot(x = year , y = table_count_val)

plt.title('Fatalities per year')

plt.xlabel('Year')

plt.ylabel('Count')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
type_count = Counter(df['Type'].dropna().tolist()).most_common(10)

type_index = [type_[0] for type_ in type_count]

type_val = [type_[1] for type_ in type_count]



fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x = type_index , y = type_val)

plt.title('Top 10 type')

plt.xlabel('Type')

plt.ylabel('Count')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
operator_count = Counter(df['Operator'].dropna().tolist()).most_common(5)

operator_list = [operator[0] for operator in operator_count] 

operator_trend = df[['Operator','Year','Fatalities']].groupby(['Operator','Year']).agg(['sum','count'])

operator_trend = operator_trend['Fatalities'].reset_index()



fig,ax = plt.subplots(figsize=(8,6))

plt.title('Operator trend')

plt.ylabel('Total Fatalities')

plt.xlabel('Year')

for operator in operator_list:

    operator_trend[operator_trend['Operator'] == operator].plot(x = 'Year',

                                                                y = 'sum',

                                                                linewidth=2,

                                                                ax=ax,

                                                                label=operator)
operator_trend = df[['Operator','Year','Fatalities']].groupby(['Operator','Year']).agg(['sum','count'])

operator_trend = operator_trend['Fatalities'].reset_index()



fig,ax = plt.subplots(figsize=(8,6))

plt.title('Operator trend')

plt.ylabel('Total Fatalities')

plt.xlabel('Year')

for operator in operator_list:

    operator_trend[operator_trend['Operator'] == operator].plot(x = 'Year',

                                                                y = 'count',

                                                                linewidth=2,

                                                                ax=ax,

                                                                label=operator)
table_count = df[['Year','Fatalities']].dropna().groupby(['Year'])['Fatalities'].agg(['sum'])



table_count = table_count.dropna().reset_index()

table_count.columns = ['Year','Total Fatalities']

fig,ax = plt.subplots(figsize=(8,6))

table_count.plot(x = 'Year' , y = 'Total Fatalities',ax=ax)

plt.title('Fatalities per year')

plt.xlabel('Year')

plt.ylabel('Total Fatalities')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)