import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



pd.set_option('max_columns', None)



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

data = pd.concat([train, test], axis=0, sort=False)
# === unique value

col_var = train.columns[2:]

df = pd.DataFrame(col_var, columns=['feature'])

df['n_train_unique'] = train[col_var].nunique(axis=0).values

df['n_test_unique'] = test[col_var].nunique(axis=0).values



for i in df.index:

    col = df.loc[i, 'feature']

    df.loc[i, 'n_overlap'] = int(np.isin(train[col].unique(), test[col]).sum())



df['value_range'] = data[col_var].max(axis=0).values - data[col_var].min(axis=0).values
df.T
# === plot

df = df.sort_values(by='n_train_unique').reset_index(drop=True)

df[['n_train_unique', 'n_test_unique', 'n_overlap']].plot(kind='barh' ,figsize=(22, 100), fontsize=20, width=0.8)

plt.yticks(df.index, df['feature'].values)

plt.xlabel('n_unique', fontsize=20)

plt.ylabel('feature', fontsize=20)

plt.legend(loc='center right', fontsize=20)