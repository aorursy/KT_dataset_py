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



import datetime



import matplotlib.pyplot as plt



import seaborn as sns

psi = pd.read_csv("../input/singapore-psi-pm25-20162019/psi_df_2016_2019.csv")

psi.head()
psi.shape
psi['timestamp'] = pd.to_datetime(psi['timestamp'])
psi.info()
psi.isnull().sum()
psi.describe()
psi['year_month'] = pd.to_datetime(psi['timestamp']).dt.to_period('M')

psi['year'] = pd.DatetimeIndex(psi['timestamp']).year

psi['month'] = pd.DatetimeIndex(psi['timestamp']).month

psi.head()
psi.groupby(['year_month']).size()
all_area = ['national','south', 'north', 'east', 'central', 'west']

psi['overall_mean'] = psi[all_area].mean(axis=1, skipna=True)
by_month = psi.groupby(['year_month']).mean()

by_month
by_month['yearmonth'] = by_month.index.to_timestamp()


plt.figure(figsize=(30,15)).suptitle('Monthly Average PSI', fontsize=50)

plt.plot(by_month['yearmonth'], by_month['overall_mean'])



plt.xlabel('Month', fontsize=30)

plt.ylabel('PSI', fontsize=30)

plt.xticks(fontsize=20, rotation=0)

plt.yticks(fontsize=20, rotation=0)

plt.show()
# all_area = ['national','south', 'north', 'east', 'central', 'west']



plt.figure(figsize=(30,15)).suptitle('Monthly Average PSI', fontsize=50)

plt.plot(by_month['yearmonth'], by_month[all_area])



plt.legend(all_area, loc="upper left", fontsize=30)

plt.xlabel('Month', fontsize=30)

plt.ylabel('PSI', fontsize=30)

plt.xticks(fontsize=20, rotation=0)

plt.yticks(fontsize=20, rotation=0)

plt.show()
plt.figure(figsize=(30,15)).suptitle('Monthly Average PSI', fontsize=50)



clrs = ['red' if (x == max(by_month['overall_mean']) or x == min(by_month['overall_mean']))  else 'grey' for x in by_month['overall_mean'] ]



sns.barplot(x=by_month.index, y=by_month['overall_mean'], palette=clrs) # color=clrs)



plt.xlabel('Month', fontsize=30)

plt.ylabel('PSI', fontsize=30)

plt.xticks(np.arange(len(by_month.index)), by_month.index, fontsize=20, rotation=90)

plt.yticks(fontsize=20, rotation=0)
plt.figure(figsize=(30,15)).suptitle('Monthly Average PSI', fontsize=50)



clrs = ['blue' if (x > 55)  else 'grey' for x in by_month['overall_mean'] ]



sns.barplot(x=by_month.index, y=by_month['overall_mean'], palette=clrs) # color=clrs)



plt.xlabel('Month', fontsize=30)

plt.ylabel('PSI', fontsize=30)

plt.xticks(np.arange(len(by_month.index)), by_month.index, fontsize=20, rotation=90)

plt.yticks(fontsize=20, rotation=0)

ls = ['blue - normal (0 - 55)' , 'grey - elevated (56 - 150)']

plt.legend(ls, loc="upper left",fontsize=30)



plt.figure(figsize=(30,15)).suptitle('Monthly Average PSI', fontsize=50)



y_pos = np.arange(len(by_month.index))



plt.bar(y_pos, by_month['overall_mean'], align='center', alpha=0.5)



plt.xlabel('Month', fontsize=30)

plt.ylabel('PSI', fontsize=30)

plt.xticks(y_pos, by_month.index, fontsize=20, rotation=90)

plt.yticks(fontsize=20, rotation=0)



plt.show()




