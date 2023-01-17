# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
colspecs=[[8,12],[12,14],[74,76],[489,491]]
names=['birth_year','birth_month','mother_age','gest_weeks']
parsed_dat2016 = pd.read_fwf('../input/Nat2016PublicUS.c20170517.r20170913.txt',header=None,colspecs=colspecs,names=names)
parsed_dat2016.head()
parsed_dat2016.describe()
parsed_dat2016[parsed_dat2016.gest_weeks==99] = np.nan
parsed_dat2016['gest_weeks'].hist()
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

f, ax = plt.subplots(figsize=(20, 8))
# sns.distplot(data,bins=np.arange(data.min(), data.max() + 1),hist=True,kde=False);
sns.distplot(parsed_dat2016['gest_weeks'].dropna(),bins=np.arange(30, data.max() + 1),hist=True,kde=False);
plt.show()
f, ax = plt.subplots(figsize=(20, 8))
sns.distplot(parsed_dat2016['mother_age'].dropna(),bins=np.arange(data.min(), data.max() + 1),hist=True,kde=False);
plt.show()
f, ax = plt.subplots(figsize=(20, 8))
sns.regplot(x="mother_age", y="gest_weeks", data=parsed_dat2016)
plt.show()
parsed_dat2016['gest_weeks'].dropna().value_counts()
