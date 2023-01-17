# Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv(open('../input/train.csv'))
df.describe()
df.info()
col_names = df.columns.tolist()

for col_name in col_names:

    missing = np.sum(df[col_name] == -1)

    print (col_name, missing)

df1 = df.replace(-1, np.NaN)
cat_cols = []

bin_cols = []

other_cols = []

ind_cols = []

reg_cols = []

car_cols = []

calc_cols = []

import re

for col_name in col_names:

    if re.search('bin', col_name):

        bin_cols.append(col_name)

    elif re.search('cat', col_name):

        cat_cols.append(col_name)

    else:

        other_cols.append(col_name)

    if re.search('ind', col_name):

        ind_cols.append(col_name)

    elif re.search('reg', col_name):

        reg_cols.append(col_name)

    elif re.search('car', col_name):

        car_cols.append(col_name)

    else:

        calc_cols.append(col_name)

other_cols.remove('id')

other_cols.remove('target')

calc_cols.remove('id')

calc_cols.remove('target')

print ("No of binary columns: ", len(bin_cols))

print ("No of categorical columns: ", len(cat_cols))

print ("No of other columns: ", len(other_cols))

print ("No of ind columns: ", len(ind_cols))

print ("No of reg columns: ", len(reg_cols))

print ("No of car columns: ", len(car_cols))

print ("No of calc columns: ", len(calc_cols))
corrmat = df1[ind_cols].dropna().corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True, cmap='RdBu');
corrmat = df1[reg_cols].dropna().corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True, cmap='RdBu');
corrmat = df1[car_cols].dropna().corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True, cmap='RdBu');
corrmat = df1[calc_cols].dropna().corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True, cmap='RdBu');
sums = (df1[['ps_ind_06_bin','ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin']].sum(axis=1))

len(sums[sums == 1])
sums = (df1[['ps_ind_16_bin','ps_ind_17_bin', 'ps_ind_18_bin']].sum(axis=1))

len(sums[sums == 1]) + len(sums[sums == 0])
df1['sum_ind_161718_bin'] = sums
target_bin = ['target'] + ['sum_ind_161718_bin']

corrmat = df1[target_bin].dropna().corr()

f, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(corrmat, vmax=.8, square=True, cmap='RdBu');
vars = ['ps_car_12', 'ps_car_14']

g = sns.pairplot(df1.dropna(), vars=vars, hue="target", size = 3.5)
for cat_col in cat_cols:

    print (cat_col, len(df1[df1['target'] == 0][cat_col].value_counts()), len(df1[df1['target'] == 1][cat_col].value_counts()))
cat_cols.remove('ps_car_11_cat')
types_sum = df1[cat_cols].apply(pd.Series.value_counts)

ax = types_sum.T.plot(kind='bar', figsize=(15, 7), fontsize=12)
plt.figure(figsize=(15, 8))

df["ps_car_11_cat"].value_counts().plot(kind='bar')
cat_cols += ['ps_car_11_cat']
for col in other_cols:

    plt.figure()

    sns.distplot(df1[col].dropna());
arr1 = ['ps_calc_01', 'ps_calc_02', 'ps_calc_03']

df1[arr1].head(10)