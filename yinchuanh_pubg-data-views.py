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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train_V2.csv')
train.describe().drop('count').T
f,ax = plt.subplots(figsize=(15, 15))
sns.set(font_scale=1)
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data = train['matchType'].copy()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
data.value_counts().plot.bar(ax=ax[0])


mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
train['matchType'].apply(mapper).value_counts().plot.bar(ax=ax[1])
train['gameSize'] = train.groupby('matchId')['matchId'].transform('count')

data = train.copy()
data = data[data['gameSize']>60]
plt.figure(figsize=(15,10))
sns.countplot(data['gameSize'])
plt.title("Game Sizes",fontsize=15)
plt.show()
plt.figure(figsize=(9,7))
sns.countplot(data['kills'])
plt.figure(figsize=(9,7))
sns.countplot(data['longestKill'])
