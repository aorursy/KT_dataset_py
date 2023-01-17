# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/StudentsPerformance.csv')
df.head()
df.describe()
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(24,8), sharey=True)

sns.distplot(df['math score'], kde=False, bins=20, ax=ax0)
sns.distplot(df['reading score'], kde=False, bins=20, ax=ax1)
sns.distplot(df['writing score'], kde=False, bins=20, ax=ax2)
sns.factorplot(data=df, x='math score', col='race/ethnicity', palette='bright')
sns.factorplot(data=df, x='reading score', col='race/ethnicity', palette='bright')
sns.factorplot(data=df, x='writing score', col='race/ethnicity', palette='bright')
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))

sns.swarmplot(data=df, x='math score', y='race/ethnicity', hue='parental level of education', ax=ax0)
sns.swarmplot(data=df, x= 'reading score', y='race/ethnicity', hue='parental level of education', ax=ax1)
sns.swarmplot(data=df, x='writing score',  y='race/ethnicity', hue='parental level of education', ax=ax2)
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))

sns.boxenplot(data=df, x='math score', y='race/ethnicity', hue='lunch', ax=ax0)
sns.boxenplot(data=df, x= 'reading score', y='race/ethnicity', hue='lunch', ax=ax1)
sns.boxenplot(data=df, x='writing score',  y='race/ethnicity', hue='lunch', ax=ax2)
df.shape
plt.figure(figsize=(25,10))
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(20, 10))

sns.countplot(data=df, x='math score', hue='gender', ax=ax0)
sns.countplot(data=df, x= 'reading score', hue='gender', ax=ax1)
sns.countplot(data=df, x='writing score',  hue='gender', ax=ax2)

sns.factorplot(data=df, x='math score', kind='box', row='test preparation course')
sns.factorplot(data=df, x='reading score', kind='box', row='test preparation course')
sns.factorplot(data=df, x='writing score', kind='box', row='test preparation course')
