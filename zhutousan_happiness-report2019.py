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
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/world-happiness-report-2019.csv')

df.head()
df.info()
df.describe()
df[df.isnull().any(1)]
df = df.fillna(df.mean().round())
plt.figure(figsize=(10,8))

corr = df.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)]= True

sns.heatmap(corr,mask=mask,annot=True,cmap=plt.cm.RdBu)
sns.lmplot(x='Ladder',y='SD of Ladder',data=df)
sns.lmplot(x='Ladder',y='Positive affect',data=df)
sns.lmplot(x='Ladder',y='Negative affect',data=df)
sns.lmplot(x='Ladder',y='Social support',data=df)
sns.lmplot(x='Ladder',y='Freedom',data=df)
sns.lmplot(x='Ladder',y='Corruption',data=df)
sns.lmplot(x='Ladder',y='Generosity',data=df)
sns.lmplot(x='Ladder',y='Log of GDP\nper capita',data=df)
sns.lmplot(x='Ladder',y='Healthy life\nexpectancy',data=df)