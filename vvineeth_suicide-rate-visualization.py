import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
df.head()
df.info()
print(df['country'])
plt.figure(figsize = (20,6))

sns.lineplot(x = 'country', y = 'suicides_no', data = df)
plt.figure (figsize = (20,6))

sns.barplot(x = 'country', y = 'suicides_no', data = df)
plt.figure(figsize = (20,6))

sns.lineplot(x = 'country', y = 'year', data = df)
plt.figure(figsize = (20,6))

sns.barplot(x = 'country', y = 'year', data = df)
plt.figure(figsize = (20,6))

sns.lineplot(x = 'sex', y = 'suicides_no', data = df)
plt.figure(figsize = (20,6))

sns.barplot(x = 'sex', y = 'suicides_no', data = df)
plt.figure(figsize = (20,6))

sns.barplot(x = 'country', y = 'gdp_per_capita ($)', data = df)
sns.pairplot(df)