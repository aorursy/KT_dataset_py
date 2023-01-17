# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
extended = pd.read_csv('../input/states_all_extended.csv')
extended.head()
fig, ax = plt.subplots()

ax.bar(extended.STATE, extended.STATE_REVENUE)

ax.set_xticklabels(extended.STATE, rotation=90)

ax.set_xlabel('States in USA')

ax.set_ylabel('State revenue ()')

fig.set_size_inches([10,4])

plt.show()
years=extended.YEAR.unique()
fig, ax = plt.subplots()

for year in years:

    year_df = extended[extended.YEAR == year]

    ax.bar(year, year_df['TOTAL_REVENUE'].mean())

ax.set_ylabel('TOTAL_REVENUE')

ax.set_xlabel('years')

plt.show()

fig, ax = plt.subplots()

for year in years:

    year_df = extended[extended.YEAR == year]

    ax.bar(year, year_df['FEDERAL_REVENUE'].mean())

    ax.bar(year, year_df['LOCAL_REVENUE'].mean(), bottom=year_df['STATE_REVENUE'].mean()+year_df['FEDERAL_REVENUE'].mean())

ax.set_ylabel('FEDERAL_REVENUE')

ax.set_xlabel('years')

plt.show()