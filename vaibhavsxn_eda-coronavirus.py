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
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
data.head()
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize = (40,20))

sns.countplot(x='Country',data=data)
cc= data.groupby('Country')['Confirmed'].sum().reset_index(drop = False).sort_values(by = 'Confirmed', ascending = False)
cc
plt.figure(figsize = (40,20))

sns.countplot(x='Confirmed',data=cc)
plt.figure(figsize=(35,17))

sns.heatmap(data.drop('Sno', axis = 1).corr(),annot=True,cmap='viridis')
data
state = data.groupby('Province/State')['Confirmed'].sum().reset_index(drop = False).sort_values(by = 'Confirmed', ascending = False)
state
plt.figure(figsize=(35,17))

sns.scatterplot(x='Province/State',y='Confirmed',data=data)
death = data.groupby('Country')['Deaths'].sum().reset_index(drop = False).sort_values(by = 'Deaths', ascending = False)
death
death_state = data.groupby('Province/State')['Deaths'].sum().reset_index(drop = False).sort_values(by = 'Deaths', ascending = False)
death_state
plt.figure(figsize=(35,17))

sns.countplot(x='Deaths',data=death_state,hue='Province/State')
data.Recovered.unique()
plt.figure(figsize=(35,17))

data.Recovered.plot()
rec_state = data.groupby('Province/State')['Recovered'].sum().reset_index(drop = False).sort_values(by = 'Recovered', ascending = False)
rec_state
plt.figure(figsize=(35,17))

sns.countplot(x='Recovered',data=rec_state,hue='Province/State')