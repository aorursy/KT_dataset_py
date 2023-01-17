# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # to plot different data on graphs
from scipy.stats import ttest_ind, ttest_ind_from_stats,chi2_contingency # t test to compare two coloumns to check if they are same
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/cereal.csv')
df.describe()
df['calories']
#plt.hist('calories')
plt.title('Different histogram')
df.describe()
#var1=df['sugars']
#var2=df['sodium']
#dropna ensures that 'NA' values are not included which other wise create an issue for t-test
ttest_ind(df.dropna()['sugars'], df.dropna()['sodium'])
#df['sugars']
#sugars=df.dropna()[float('sugars')]
#plt.hist('sugars')
df.sugars.describe()
plt.hist(df.sugars)
plt.hist(df.sodium)
#ttest_ind('calories', 'protein',equal_var=False)
df.dtypes
sns.barplot(x='name', y= 'sugars',data=df)
df.name.describe()
#chi2_contingency('name','mfr')

chi2_contingency(pd.crosstab(df['name'],df['mfr']))
#chi2_contingency()
#df.name
#df.describe()
#print (df.values)