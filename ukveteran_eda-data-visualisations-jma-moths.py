import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



sns.set_style('darkgrid')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/moths-data/moths.csv', encoding='latin1')



print('Number of features: %s' %data.shape[1])

print('Number of examples: %s' %data.shape[0])
data.head()
data.info()
data.isnull().sum()
for col in data[['meters', 'A', 'P']]:

    print('Unique values in column: %s' %col)

    print(data[col].unique())

    print('\n')



print('Number of unique values: ')

print(data[['meters', 'A', 'P']].nunique())
number = pd.DataFrame(data['meters'].describe())

number.columns = ['meters']

number['Stats'] = number.index

number.reset_index(inplace=True, drop=True)

number
data1 = pd.DataFrame(data.groupby('habitat')['meters'].sum().sort_values(ascending=False).reset_index())

plt.figure(figsize=(15,8))

sns.barplot(data=data1, x='habitat', y='meters', palette='autumn')

plt.xlabel('Habitat')

plt.ylabel('Meters')

plt.xticks(rotation=80)
data2 = pd.DataFrame(data.groupby('habitat')['A'].sum().sort_values(ascending=False).reset_index())

plt.figure(figsize=(15,8))

sns.barplot(data=data2, x='habitat', y='A', palette='autumn')

plt.xlabel('Habitat')

plt.ylabel('A')

plt.xticks(rotation=80)
data3 = pd.DataFrame(data.groupby('habitat')['P'].sum().sort_values(ascending=False).reset_index())

plt.figure(figsize=(15,8))

sns.barplot(data=data3, x='habitat', y='P', palette='autumn')

plt.xlabel('Habitat')

plt.ylabel('P')

plt.xticks(rotation=80)