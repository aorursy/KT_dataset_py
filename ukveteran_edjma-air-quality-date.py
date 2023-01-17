import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use('bmh')
df = pd.read_csv('../input/air-quality-data/airmay.csv')

df.head()
df.info()
print(df['X1'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['X1'], color='g', bins=100, hist_kws={'alpha': 0.4})
print(df['X2'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['X2'], color='g', bins=100, hist_kws={'alpha': 0.4})
print(df['X3'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['X3'], color='g', bins=100, hist_kws={'alpha': 0.4})
print(df['Y'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['Y'], color='g', bins=100, hist_kws={'alpha': 0.4})
list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64', 'int64'])

df_num.head()
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['X1'])
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['X2'])
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['X3'])
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['Y'])