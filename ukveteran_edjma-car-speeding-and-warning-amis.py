import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use('bmh')
df = pd.read_csv('../input/car-speeding-and-warning-signs/amis.csv')

df.head()
df.info()
print(df['speed'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['speed'], color='g', bins=100, hist_kws={'alpha': 0.4})
print(df['period'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['period'], color='g', bins=100, hist_kws={'alpha': 0.4})
list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64', 'int64'])

df_num.head()
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['speed'])