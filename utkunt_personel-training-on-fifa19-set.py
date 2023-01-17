# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/fifa19/data.csv')

data.info()
# stat info about data

data.describe()
# correlations between numerical data

data.corr()
# visualisation of data correlation by seaborn

sb.heatmap(data.corr(), linewidths=0.5)

plt.show()
# showing first 5 rows of table, 5 is default value, first 10 rows by data.head(10)

data.head()
# columns title

data.columns
# line plot

data.Potential.plot(kind='line', color='b', label='Potential',alpha=.5, grid=True, linestyle=':')

data.BallControl.plot(color='g',label='Overall',alpha=0.5)

plt.title('Potential - Overall')

plt.xlabel('x axis')

plt.ylabel('y label')

plt.show()



# p.s : i admit that it's a little bit meaningless :)
# scatter plot

data.plot(kind='scatter', x='GKDiving', y='GKHandling', alpha=.5, grid=True)

plt.xlabel('GK Diving')

plt.ylabel('GK Handling')

plt.title('Some GoalKeeping Skills Scatter Diagram')

plt.show()
# histogram

data.Age.plot(kind='hist', bins=30, figsize=(12,12))

plt.show()
### in this example country names are keys, city names are values

capital_city = {'turkey':'ankara', 'greece':'athens', 'germany':'berlin', 'england':'london'}

print('keys: ', capital_city.keys())

print('values: ', capital_city.values())
capital_city['spain'] = 'madrid' # adding new entry

del capital_city['england'] # remove existing one
# filtering datafram

high_potential = data.Potential>92

data[high_potential]
# multi-filter

data[(data.Potential>90) & (data.Overall>93)]
# loops in pandas

for index, value in capital_city.items():

    print(index,': ',value)
list01 = [10,20,30]

list02 = [i*2 for i in list01]

list02
# conditions on iterations

list03 = [i**2 if i > 20 else i*2 for i in list01]

list03
data['newie'] = ['high' if i > 92 else 'ordinary' if i > 80 else 'none' for i in data.Potential]

data.loc[:10,['Name','newie']]
# frequency of countries

print(data.head(100).Nationality.value_counts(dropna=False))
# box plot

data.head(50).boxplot(column='Overall', by='Nationality', figsize=(18,18))
# melting data

new_data = data.head(10)

melted_data = pd.melt(frame=new_data, id_vars='Name',value_vars=['Overall', 'Potential'])

melted_data
# pivoting data

melted_data.pivot(index='Name',columns='variable',values='value')
# by row

data01 = data.head()

data02 = data.tail()

newie01 = pd.concat([data01,data02],axis=0,ignore_index=True)

newie01
# by column

data03 = data.head().Overall

data04 = data.head().Potential

data05 = data.head().Name

newie02 = pd.concat([data05,data03,data04],axis=1)

newie02