# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
data.head()
data.info()
data.drop(columns='rankPoints', inplace=True)
data.head(1)
plt.subplots(figsize=(15,5))

sns.heatmap(data.corr(),linewidth=0.5,cmap='winter')

plt.title('Correlation among the columns')

plt.show()
won=data['winPlacePerc']==1

lost=data['winPlacePerc']==0

a=data[won|lost]

a['winPlacePerc']=a['winPlacePerc'].astype('str')

a

plt.subplots(figsize=(10,5))

sns.countplot(y='matchType',hue='winPlacePerc',data=a)

plt.subplots(figsize=(15,5))

sns.barplot(x='boosts',y='winPlacePerc', data=data)

plt.subplots(figsize=(15,5))

sns.barplot(x='headshotKills',y='winPlacePerc', data=data)





sns.jointplot(x='weaponsAcquired',y='winPlacePerc' ,data=data)

sns.boxplot(x='revives',y='winPlacePerc',data=data)

plt.subplots(figsize=(20,8))

a=data['matchType'].value_counts().to_frame().reset_index()

b=a['index'].to_list()

c=a['matchType'].to_list()

plt.pie(c,labels=b, autopct="%0.2f%%")

plt.show()
sns.scatterplot(x='longestKill',y='winPlacePerc', data=data)

sns.scatterplot(x='matchDuration',y='winPlacePerc', data=data)

plt.subplots(figsize=(15,5))

sns.violinplot(x='assists',y='winPlacePerc', data=data)
sns.scatterplot(x='swimDistance',y='winPlacePerc', data=data)
sns.scatterplot(x='rideDistance',y='winPlacePerc', data=data)
sns.scatterplot(x='walkDistance',y='winPlacePerc', data=data)
sns.barplot(x='vehicleDestroys',y='winPlacePerc', data=data)
sns.boxplot(x='teamKills', y='winPlacePerc', data=data)