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

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

%matplotlib inline
raw_data = pd.read_csv('../input/IPCC_emissions.csv', sep=';',index_col='ID')
raw_data.head()
raw_data.describe()
raw_data.info()
raw_data['Perioden'] = raw_data['Perioden'].apply(lambda x: int(x[:4]))
raw_data['Bronnen'].unique()
raw_data.head()
raw_data.info()
sns.pairplot(raw_data)
raw_data_bronnen = raw_data.set_index(['Bronnen','Perioden']).sort_index()
raw_data_bronnen.head()
raw_data_perioden = raw_data.set_index(['Perioden','Bronnen']).sort_index()
raw_data_perioden.head(30)
raw_data_co2 = raw_data.pivot(index='Perioden',columns='Bronnen',values='CO2_1')
raw_data_co2.head()
raw_data_ch4 = raw_data.pivot(index='Perioden',columns='Bronnen',values='CH4_2')
raw_data_ch4.head()
raw_data_no2 = raw_data.pivot(index='Perioden',columns='Bronnen',values='N2O_3')
raw_data_no2.head()
bronnen_list = raw_data['Bronnen'].unique()

raw_data_co2[bronnen_list].iplot(kind='line',size=10)
raw_data_ch4[bronnen_list].iplot(kind='line',size=10)
raw_data_no2[bronnen_list].iplot(kind='line',size=10)
plt.figure(figsize=(15,20))

sns.lineplot(data=raw_data,x='Perioden',y='CO2_1',estimator='sum')
plt.figure(figsize=(15,20))

sns.lineplot(data=raw_data,y='N2O_3',x='Perioden',estimator='sum')
plt.figure(figsize=(15,20))

sns.lineplot(data=raw_data,y='CH4_2',x='Perioden',estimator='sum')
plt.figure(figsize=(15,15))

pollutions = [x for x in raw_data_bronnen[raw_data_bronnen.columns].sum()]

labels = raw_data_bronnen.columns

sns.barplot(x=labels,y=pollutions)
plt.figure(figsize=(15,15))

pollutions = [x for x in raw_data_bronnen[raw_data_bronnen.columns[1:]].sum()]

labels = raw_data_bronnen.columns[1:]

sns.barplot(x=labels,y=pollutions)