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
fire_raw = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding = "latin1", thousands = '.')

print("First 5 rows of the dataset: ")

print(fire_raw.head())

print("\nData Description")

print(fire_raw.describe())

print("\nData info")

print(fire_raw.info())
month_list = fire_raw.groupby('month').size()

month_inspan = month_list.index.values.tolist()

month_innum = [4,8,12,2,1,7,6,5,3,11,10,9]

replace_dic = {}

for i in range(len(month_inspan)):

    replace_dic[month_inspan[i]] = month_innum[i]

fire_data = fire_raw.replace({'month': replace_dic})

fire_data.head()
state_series = fire_data.groupby('state').size()

print(state_series)
state_ls = state_series.index.values.tolist()

print("States in dataset:")

print(state_ls)

print('\n')

year_ls = fire_data.groupby('year').size().index.values.tolist()

print("Years in dataset:")

print(year_ls)
Acre_dataset = fire_data.query("state == 'Acre\'")

print(Acre_dataset.describe())

Acre_dataset.to_csv('Acre.csv')

# link to download the current output

from IPython.display import FileLink

print("Link to download Acre_dataset:")

FileLink(r'Acre.csv')
Acre_firecount = [0] * len(year_ls)

for i in range(len(Acre_dataset)):

    for j in range(len(year_ls)):

        if Acre_dataset.iloc[i]['year'] == year_ls[j]:

            Acre_firecount[j] += Acre_dataset.iloc[i]['number']

print(Acre_firecount)
import matplotlib.pyplot as plt



# One way to set the x axis shows year by year is to convert the years from int to string.

year_str = list(map(str, year_ls))

# Change the size of the plot

plt.figure(figsize=(20,10))

# Set title

plt.title('Forest Fire in State Acre(1998 - 2017)', fontsize = 20)

# Set labels

plt.xlabel('year', fontsize = 15)

plt.ylabel('count', fontsize = 15)

# Draw the bars

plt.bar(year_str, Acre_firecount)