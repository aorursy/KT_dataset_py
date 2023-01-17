# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import scipy

from scipy import stats



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/fifa19/data.csv')[['Nationality', 'Overall', 'Preferred Foot']]



data.head()
left_data = np.asarray((data[data['Preferred Foot'] == 'Left'])['Overall'])

right_data = np.asarray((data[data['Preferred Foot'] == 'Right'])['Overall'])

print(len(left_data), len(right_data))
print(left_data.mean(), right_data.mean())
fig = plt.figure(figsize=(24, 6))



ax1 = plt.subplot(121)

plt.hist(left_data, 47)



ax2 = plt.subplot(122)

plt.hist(right_data, 47)



plt.show()
scipy.stats.t.cdf(stats.ttest_ind(left_data,right_data)[0],len(left_data) -  1 + len(right_data) - 1)
argentina = np.asarray((data[data['Nationality'] == 'Argentina'])['Overall'])

england = np.asarray((data[data['Nationality'] == 'England'])['Overall'])

germany = np.asarray((data[data['Nationality'] == 'Germany'])['Overall'])

print(len(argentina), len(england), len(germany))
print(argentina.mean(), england.mean(), germany.mean())
plt.hist(argentina, 42)



plt.show()
plt.hist(england, 42)



plt.show()
plt.hist(germany, 42)



plt.show()
print(stats.ttest_ind(argentina, england))

print(stats.ttest_ind(england, germany))

print(stats.ttest_ind(germany,argentina))
print(scipy.stats.t.cdf(np.abs(stats.ttest_ind(argentina, england)[0]),len(argentina) -  1 + len(england) - 1))

print(scipy.stats.t.cdf(np.abs(stats.ttest_ind(england, germany)[0]),len(england) -  1 + len(germany) - 1))

print(scipy.stats.t.cdf(np.abs(stats.ttest_ind(germany, argentina)[0]),len(germany) -  1 + len(argentina) - 1))