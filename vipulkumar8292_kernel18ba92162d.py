import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
store = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
store.head()
store['Installs_new'] = store['Installs'].apply(lambda x : x.split('+')[0])
store['Installs_new'] = store['Installs_new'].str.replace(',','')
store['Size'] = store['Size'].str.replace('M','')
store.dropna(inplace=True)
store.isnull().sum()
store['Installs_new'] = store.Installs_new.astype('int32')
store['Rating'] = store.Rating.astype('float32')
store['Reviews'] = store.Reviews.astype('int32')
corr = store.corr()

corr
import seaborn as sns

import matplotlib.pyplot as plt
f, ax=plt.subplots(figsize=(6,6))

sns.heatmap(corr, ax=ax, cmap='YlGnBu',linewidths=0.1)