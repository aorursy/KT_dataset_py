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
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
tdf = pd.read_csv('../input/train_V2.csv')
tdf.head()
tdf.info(verbose=False)
tdf.info()
tdf[["maxPlace","winPlacePerc"]].head()
ax1 = tdf[["maxPlace","winPlacePerc"]].plot.scatter(x='maxPlace', y='winPlacePerc', c='DarkBlue')
sns.jointplot(x="maxPlace", y="winPlacePerc", data=tdf, height=10, ratio=3, color="r")
plt.show()
tdf.corr()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(tdf.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
sns.jointplot(x="winPlacePerc", y="killPlace", data=tdf, height=10, ratio=3, color="r")
plt.show()
plt.figure(figsize=(15,10))
plt.title("Kill Place",fontsize=15)
sns.distplot(tdf['killPlace'])
plt.show()
sns.jointplot(x="winPlacePerc", y="walkDistance", data=tdf, height=10, ratio=3, color="r")
plt.show()
plt.figure(figsize=(15,10))
plt.title("Walk Distance",fontsize=15)
sns.distplot(tdf['walkDistance'])
plt.show()
tdf["weaponsAcquired"].describe()
plt.figure(figsize=(15,8))
sns.boxplot(x="weaponsAcquired", y="winPlacePerc", data=tdf)
plt.show()
data = tdf.copy()
plt.figure(figsize=(15,10))
plt.title("weaponsAcquired",fontsize=15)
sns.distplot(data['weaponsAcquired'])
plt.show()
data['weaponsAcquiredCategories'] = pd.cut(data['weaponsAcquired'], [-1, 0, 2, 4, 6,55,75, 236], labels=['0','1-2', '3-4', '5-6', '7-55','56-75','+'])

plt.figure(figsize=(15,8))
sns.boxplot(x="weaponsAcquiredCategories", y="winPlacePerc", data=data)
plt.show()
