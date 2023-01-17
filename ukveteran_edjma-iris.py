import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use('bmh')
df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')

df.head()
df.info()
print(df['sepal_length'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['sepal_length'], color='g', bins=100, hist_kws={'alpha': 0.4})
print(df['sepal_width'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['sepal_width'], color='g', bins=100, hist_kws={'alpha': 0.4})
print(df['petal_length'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['petal_length'], color='g', bins=100, hist_kws={'alpha': 0.4})
print(df['petal_width'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['petal_width'], color='g', bins=100, hist_kws={'alpha': 0.4})
list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64', 'int64'])

df_num.head()
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['sepal_length'])
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['sepal_width'])
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['petal_length'])
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['petal_width'])