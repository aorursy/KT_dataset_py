import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use('bmh')
df = pd.read_csv('../input/leaf-and-air-temperature-data/leaftemp.csv')

df.head()
df.info()
print(df['tempDiff'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['tempDiff'], color='g', bins=100, hist_kws={'alpha': 0.4});
print(df['BtempDiff'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['BtempDiff'], color='g', bins=100, hist_kws={'alpha': 0.4});
print(df['vapPress'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['vapPress'], color='g', bins=100, hist_kws={'alpha': 0.4});
list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64', 'int64'])

df_num.head()
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['vapPress'])
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['tempDiff'])
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['BtempDiff'])
plt.figure(figsize = (10, 6))

ax = sns.boxplot(x='tempDiff', y='BtempDiff', data=df)

plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")

plt.xticks(rotation=45)
plt.figure(figsize = (10, 6))

ax = sns.boxplot(x='vapPress', y='tempDiff', data=df)

plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")

plt.xticks(rotation=45)
plt.figure(figsize = (10, 6))

ax = sns.boxplot(x='vapPress', y='BtempDiff', data=df)

plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")

plt.xticks(rotation=45)