import matplotlib.pyplot as plt

import numpy as np # linear algebra

import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import seaborn as sns

import warnings



from lightgbm import LGBMClassifier

from pandas import DataFrame

from sklearn.feature_selection import mutual_info_classif

from sklearn.feature_selection import chi2

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import ShuffleSplit

from xgboost import XGBClassifier



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



warnings.filterwarnings('ignore')
data_dir = '../input'

train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

X = train_df.drop(['target', 'id'], axis=1)

y = train_df['target']

y_cnts = y.value_counts()

y_pct = y_cnts / y.shape[0]

print(y_cnts[0], y_cnts[1], y_pct[0], y_pct[1])

print(list(X.dtypes.loc[np.int64==X.dtypes].index))

print(list(X.dtypes.loc[np.float64==X.dtypes].index))

print('float_num(', X.dtypes.loc[np.float64==X.dtypes].shape[0],

      '), int_num(', X.dtypes.loc[np.int64==X.dtypes].shape[0], ')')

print(X.shape)



test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

test_x = test_df.drop('id', axis=1)

print(test_x.shape)
def round_float_str(info):

    def promote(matched):

        return str(float(matched.group()) + 9e-16)



    def trim1(matched):

        return matched.group(1) + matched.group(2)



    def trim2(matched):

        return matched.group(1)



    info = re.sub(r'[\d.]+?9{4,}[\de-]+', promote, info)

    info = re.sub(r'([\d.]*?)\.?0{4,}\d+(e-\d+)', trim1, info)

    info = re.sub(r'([\d.]+?)0{4,}\d+', trim2, info)



    return info





def prime_analysis(data):

    for col_name, col in data.iteritems():

        nuique = col.nunique()

        if np.object != col.dtype:

            if nuique > 40:

                print('--------【', col_name, '】, nuique(', nuique, '), null_cnt(', col.isnull().sum(),

                      '), not_null_cnt(', col.notnull().sum(), '), min(', col.min(), '), max(', col.max(), 

                      '), mean(', col.mean(), '), median(', col.median(), ')--------')

            else:

                print('--------【', col_name, '】, uique',

                      list(col.unique()) if col.dtype not in [np.float, np.float64, np.float32, np.float16] else [

                          round_float_str(str(ele)) for ele in col.unique()], ', null_cnt(', col.isnull().sum(),

                      '), not_null_cnt(', col.notnull().sum(), '), min(', col.min(), '), max(', col.max(), 

                      '), mean(', col.mean(), '), median(', col.median(), ')--------')

        else:

            if nuique > 40:

                print('--------【', col_name, '】, nuique(', nuique, '), null_cnt(', col.isnull().sum(),

                      '), not_null_cnt(', col.notnull().sum(), ')--------')

            else:

                print('--------【', col_name, '】, uique', list(col.unique()), ', null_cnt(', col.isnull().sum(),

                      '), not_null_cnt(', col.notnull().sum(), ')--------')



prime_analysis(X)
prime_analysis(test_x)
bin_cols = [col for col in X.columns if '_bin' in col]

print(bin_cols)



invariance_pct = y_pct[1] * 0.05

invar_cols = []

for col in bin_cols:

    print(col)

    tr_cnts = X[col].value_counts()

    tr_pct = tr_cnts/X.shape[0]

    print(tr_cnts[0],tr_cnts[1],tr_pct[0],tr_pct[1])

    ts_cnts = test_x[col].value_counts()

    ts_pct = ts_cnts/test_x.shape[0]

    print(ts_cnts[0],ts_cnts[1],ts_pct[0],ts_pct[1])

    print('-------------------------------')

    

    if tr_pct[0] < invariance_pct or tr_pct[1] < invariance_pct:

        invar_cols.append(col)        

print(invar_cols)



X['invar_combo_bin'] = X[invar_cols[0]]

test_x['invar_combo_bin'] = test_x[invar_cols[0]]

for i in range(1, len(invar_cols)):

    X['invar_combo_bin'] += 2 ** i * X[invar_cols[i]]

    test_x['invar_combo_bin'] += 2 ** i * test_x[invar_cols[i]]

X = X.drop(invar_cols, axis=1)

test_x = test_x.drop(invar_cols, axis=1)

print(X.invar_combo_bin.value_counts().to_dict())

print(test_x.invar_combo_bin.value_counts().to_dict())
# combine calc_bin infos

calc_bin_cols = ['ps_calc_' + str(i) + '_bin' for i in range(15, 21)]

X['combo_calc_bin'] = X[calc_bin_cols[0]]

test_x['combo_calc_bin'] = test_x[calc_bin_cols[0]]

for i in range(1, len(calc_bin_cols)):

    X['combo_calc_bin'] += 2 ** i * X[calc_bin_cols[i]]

    test_x['combo_calc_bin'] += 2 ** i * test_x[calc_bin_cols[i]]

X = X.drop(calc_bin_cols, axis=1)

test_x = test_x.drop(calc_bin_cols, axis=1)

print(X.combo_calc_bin.value_counts().to_dict())

print(test_x.combo_calc_bin.value_counts().to_dict())
cat_cols = [col for col in X.columns if '_cat' in col]

print(cat_cols)



# invariance_pct = y_pct[1] * 0.05

# invar_cols = []

for col in cat_cols:

    print(col)

    tr_cnts = X[col].value_counts()

    tr_pct = tr_cnts/X.shape[0]

    print(tr_cnts.to_dict())

    print(tr_pct.to_dict())

    ts_cnts = test_x[col].value_counts()

    ts_pct = ts_cnts/test_x.shape[0]

    print(ts_cnts.to_dict())

    print(ts_pct.to_dict())

    print('-------------------------------')
def fill_na(df):

    df.loc[df.ps_car_02_cat == -1, 'ps_car_02_cat'] = 1

    df.loc[df.ps_car_11 == -1, 'ps_car_11'] = 3

    df.loc[df.ps_car_12 == -1, 'ps_car_12'] = 0



fill_na(X)

fill_na(test_x)
# distribution

col = 'ps_ind_02_cat'

s = X[col]

s1 = test_x[col]

print(s.value_counts().to_dict())

print(s1.value_counts().to_dict())



df = DataFrame(columns=['col','tag'])

df['col'] = np.append(s.values, s1.values)

df['tag'] = 'train'

df.loc[s.shape[0]:, 'tag'] = 'test'

print(df.loc[df['tag']=='test'].shape)



plt.figure(figsize=(12, 8))

sns.countplot(x='col', hue='tag', data=df)





mis = []

chis = []

eles = s.unique()

print(eles)

for ele in eles:

    ts = DataFrame(s.copy())

    ts.loc[ts[col] == -1, col] = ele+1

    mis.append(mutual_info_classif(ts, y, random_state=0)[0])

    chis.append(chi2(ts,y))

print(sorted([(eles[i],mi) for i,mi in enumerate(mis)], key=lambda pair: pair[1]))

print(sorted([(eles[i],chi) for i,chi in enumerate(chis)], key=lambda pair: pair[1][0][0]))

X.loc[X[col] == -1, col] = 3

test_x.loc[test_x[col] == -1, col] = 3
# distribution

col = 'ps_ind_04_cat'

s = X[col]

s1 = test_x[col]

print(s.value_counts().to_dict())

print(s1.value_counts().to_dict())



df = DataFrame(columns=['col','tag'])

df['col'] = np.append(s.values, s1.values)

df['tag'] = 'train'

df.loc[s.shape[0]:, 'tag'] = 'test'

print(df.loc[df['tag']=='test'].shape)



plt.figure(figsize=(12, 8))

sns.countplot(x='col', hue='tag', data=df)





mis = []

chis = []

eles = s.unique()

print(eles)

for ele in eles:

    ts = DataFrame(s.copy())

    ts.loc[ts[col] == -1, col] = ele+1

    mis.append(mutual_info_classif(ts, y, random_state=0)[0])

    chis.append(chi2(ts,y))

print(sorted([(eles[i],mi) for i,mi in enumerate(mis)], key=lambda pair: pair[1]))

print(sorted([(eles[i],chi) for i,chi in enumerate(chis)], key=lambda pair: pair[1][0][0]))

X.loc[X[col] == -1, col] = 1

test_x.loc[test_x[col] == -1, col] = 1
# distribution

col = 'ps_ind_05_cat'

s = X[col]

s1 = test_x[col]

print(s.value_counts().to_dict())

print(s1.value_counts().to_dict())



df = DataFrame(columns=['col','tag'])

df['col'] = np.append(s.values, s1.values)

df['tag'] = 'train'

df.loc[s.shape[0]:, 'tag'] = 'test'

print(df.loc[df['tag']=='test'].shape)



plt.figure(figsize=(12, 8))

sns.countplot(x='col', hue='tag', data=df)





mis = []

chis = []

eles = s.unique()

print(eles)

for ele in eles:

    ts = DataFrame(s.copy())

    ts.loc[ts[col] == -1, col] = ele+1

    mis.append(mutual_info_classif(ts, y, random_state=0)[0])

    chis.append(chi2(ts,y))

print(sorted([(eles[i],mi) for i,mi in enumerate(mis)], key=lambda pair: pair[1]))

print(sorted([(eles[i],chi) for i,chi in enumerate(chis)], key=lambda pair: pair[1][0][0]))
# distribution

col = 'ps_car_01_cat'

s = X[col]

s1 = test_x[col]

print(s.value_counts().to_dict())

print(s1.value_counts().to_dict())



df = DataFrame(columns=['col','tag'])

df['col'] = np.append(s.values, s1.values)

df['tag'] = 'train'

df.loc[s.shape[0]:, 'tag'] = 'test'

print(df.loc[df['tag']=='test'].shape)



plt.figure(figsize=(12, 8))

sns.countplot(x='col', hue='tag', data=df)





mis = []

chis = []

eles = s.unique()

print(eles)

for ele in eles:

    ts = DataFrame(s.copy())

    ts.loc[ts[col] == -1, col] = ele+1

    mis.append(mutual_info_classif(ts, y, random_state=0)[0])

    chis.append(chi2(ts,y))

print(sorted([(eles[i],mi) for i,mi in enumerate(mis)], key=lambda pair: pair[1]))

print(sorted([(eles[i],chi) for i,chi in enumerate(chis)], key=lambda pair: pair[1][0][0]))

X.loc[X[col] == -1, col] = 5

test_x.loc[test_x[col] == -1, col] = 5
# distribution

col = 'ps_car_03_cat'

s = X[col]

s1 = test_x[col]

print(s.value_counts().to_dict())

print(s1.value_counts().to_dict())



df = DataFrame(columns=['col','tag'])

df['col'] = np.append(s.values, s1.values)

df['tag'] = 'train'

df.loc[s.shape[0]:, 'tag'] = 'test'

print(df.loc[df['tag']=='test'].shape)



plt.figure(figsize=(12, 8))

sns.countplot(x='col', hue='tag', data=df)





mis = []

chis = []

eles = s.unique()

print(eles)

for ele in eles:

    ts = DataFrame(s.copy())

    ts.loc[ts[col] == -1, col] = ele+1

    mis.append(mutual_info_classif(ts, y, random_state=0)[0])

    chis.append(chi2(ts,y))

print(sorted([(eles[i],mi) for i,mi in enumerate(mis)], key=lambda pair: pair[1]))

print(sorted([(eles[i],chi) for i,chi in enumerate(chis)], key=lambda pair: pair[1][0][0]))
# distribution

col = 'ps_car_05_cat'

s = X[col]

s1 = test_x[col]

print(s.value_counts().to_dict())

print(s1.value_counts().to_dict())



df = DataFrame(columns=['col','tag'])

df['col'] = np.append(s.values, s1.values)

df['tag'] = 'train'

df.loc[s.shape[0]:, 'tag'] = 'test'

print(df.loc[df['tag']=='test'].shape)



plt.figure(figsize=(12, 8))

sns.countplot(x='col', hue='tag', data=df)





mis = []

chis = []

eles = s.unique()

print(eles)

for ele in eles:

    ts = DataFrame(s.copy())

    ts.loc[ts[col] == -1, col] = ele+1

    mis.append(mutual_info_classif(ts, y, random_state=0)[0])

    chis.append(chi2(ts,y))

print(sorted([(eles[i],mi) for i,mi in enumerate(mis)], key=lambda pair: pair[1]))

print(sorted([(eles[i],chi) for i,chi in enumerate(chis)], key=lambda pair: pair[1][0][0]))
# distribution

col = 'ps_car_07_cat'

s = X[col]

s1 = test_x[col]

print(s.value_counts().to_dict())

print(s1.value_counts().to_dict())



df = DataFrame(columns=['col','tag'])

df['col'] = np.append(s.values, s1.values)

df['tag'] = 'train'

df.loc[s.shape[0]:, 'tag'] = 'test'

print(df.loc[df['tag']=='test'].shape)



plt.figure(figsize=(12, 8))

sns.countplot(x='col', hue='tag', data=df)





mis = []

chis = []

eles = s.unique()

print(eles)

for ele in eles:

    ts = DataFrame(s.copy())

    ts.loc[ts[col] == -1, col] = ele+1

    mis.append(mutual_info_classif(ts, y, random_state=0)[0])

    chis.append(chi2(ts,y))

print(sorted([(eles[i],mi) for i,mi in enumerate(mis)], key=lambda pair: pair[1]))

print(sorted([(eles[i],chi) for i,chi in enumerate(chis)], key=lambda pair: pair[1][0][0]))
# distribution

col = 'ps_car_09_cat'

s = X[col]

s1 = test_x[col]

print(s.value_counts().to_dict())

print(s1.value_counts().to_dict())



df = DataFrame(columns=['col','tag'])

df['col'] = np.append(s.values, s1.values)

df['tag'] = 'train'

df.loc[s.shape[0]:, 'tag'] = 'test'

print(df.loc[df['tag']=='test'].shape)



plt.figure(figsize=(12, 8))

sns.countplot(x='col', hue='tag', data=df)





mis = []

chis = []

eles = s.unique()

print(eles)

for ele in eles:

    ts = DataFrame(s.copy())

    ts.loc[ts[col] == -1, col] = ele+1

    mis.append(mutual_info_classif(ts, y, random_state=0)[0])

    chis.append(chi2(ts,y))

print(sorted([(eles[i],mi) for i,mi in enumerate(mis)], key=lambda pair: pair[1]))

print(sorted([(eles[i],chi) for i,chi in enumerate(chis)], key=lambda pair: pair[1][0][0]))

X.loc[X[col] == -1, col] = 3

test_x.loc[test_x[col] == -1, col] = 3
cols=['ps_calc_01', 'ps_calc_02', 'ps_calc_03']

for col in cols:

    df=DataFrame(X[col])

    df['target'] = y

    plt.figure(figsize=(12, 8))

    sns.countplot(x=col, hue='target', data=df)



    print(col)

    print(dict([(round_float_str(str(k)),v) for k,v in X[col].value_counts().to_dict().items()]))

    print(dict([(round_float_str(str(k)),v) for k,v in test_x[col].value_counts().to_dict().items()]))

    col_cnt = X[col].groupby(y).value_counts()

    print('---------------0--------------')

    print(dict([(round_float_str(str(k)),v) for k,v in col_cnt[0].to_dict().items()]))

    print('---------------1--------------')

    print(dict([(round_float_str(str(k)),v) for k,v in col_cnt[1].to_dict().items()]))

    print('-------------ratio-------------')

    print(dict([(round_float_str(str(k)),v) for k,v in (col_cnt[0]/col_cnt[1]).to_dict().items()]))

    print('-------------------------------------------------')



df = DataFrame(X[cols+['ps_ind_01']])

df['target'] = y

cnts = df.groupby(cols+['target']).count().reset_index()

s=cnts.ps_ind_01

s1=s.loc[s<100]

s2=s.loc[s>100]

print(s1.median(),s1.min(),s1.max(),s1.mean(),s1.std(),s1.std()/s1.mean(), ' ', 

      s2.median(),s2.min(),s2.max(),s2.mean(),s2.std(),s2.std()/s2.mean())



np.random.seed(0)

l1=np.random.poisson(21, 10*10*10)

l2=np.random.poisson(574, 10*10*10)

print(np.median(l1),np.min(l1),np.max(l1),np.mean(l1),np.std(l1), ' ', 

      np.median(l2),np.min(l2),np.max(l2),np.mean(l2),np.std(l2))
cols=['ps_calc_' + (str(i) if i>=10 else '0'+str(i)) for i in range(4, 15)]

for col in cols:

    df=DataFrame(X[col])

    df['target'] = y

    plt.figure(figsize=(12, 8))

    sns.countplot(x=col, hue='target', data=df)



    print(col)

    print(X[col].value_counts().to_dict())

    print(test_x[col].value_counts().to_dict())

    col_cnt = X[col].groupby(y).value_counts()

    print('---------------0--------------')

    print(col_cnt[0].to_dict())

    print('---------------1--------------')

    print(col_cnt[1].to_dict())

    print('-------------ratio-------------')

    print((col_cnt[0]/col_cnt[1]).to_dict())

    print('-------------------------------------------------')
cols=['ps_reg_01','ps_reg_02']

for col in cols:

    df=DataFrame(X[col])

    df['target'] = y

    plt.figure(figsize=(12, 8))

    sns.countplot(x=col, hue='target', data=df)



    print(col)

    print(dict([(round_float_str(str(k)),v) for k,v in X[col].value_counts().to_dict().items()]))

    print(dict([(round_float_str(str(k)),v) for k,v in test_x[col].value_counts().to_dict().items()]))

    col_cnt = X[col].groupby(y).value_counts()

    print('---------------0--------------')

    print(dict([(round_float_str(str(k)),v) for k,v in col_cnt[0].to_dict().items()]))

    print('---------------1--------------')

    print(dict([(round_float_str(str(k)),v) for k,v in col_cnt[1].to_dict().items()]))

    print('-------------ratio-------------')

    print(dict([(round_float_str(str(k)),v) for k,v in (col_cnt[0]/col_cnt[1]).to_dict().items()]))

    print('-------------------------------------------------')
X.loc[X.ps_car_11 == -1, 'ps_car_11'] = 3

test_x.loc[test_x.ps_car_11 == -1, 'ps_car_11'] = 3



cols=['ps_car_11','ps_car_15']

for col in cols:

    df=DataFrame(X[col])

    df['target'] = y

    plt.figure(figsize=(12, 8))

    sns.countplot(x=col, hue='target', data=df)



    print(col)

    print(dict([(round_float_str(str(k)),v) for k,v in X[col].value_counts().to_dict().items()]))

    print(dict([(round_float_str(str(k)),v) for k,v in test_x[col].value_counts().to_dict().items()]))

    col_cnt = X[col].groupby(y).value_counts()

    print('---------------0--------------')

    print(dict([(round_float_str(str(k)),v) for k,v in col_cnt[0].to_dict().items()]))

    print('---------------1--------------')

    print(dict([(round_float_str(str(k)),v) for k,v in col_cnt[1].to_dict().items()]))

    print('-------------ratio-------------')

    print(dict([(round_float_str(str(k)),v) for k,v in (col_cnt[0]/col_cnt[1]).to_dict().items()]))

    print('-------------------------------------------------')
cols=['ps_ind_01','ps_ind_03','ps_ind_14','ps_ind_15']

for col in cols:

    df=DataFrame(X[col])

    df['target'] = y

    plt.figure(figsize=(12, 8))

    sns.countplot(x=col, hue='target', data=df)



    print(col)

    print(dict([(round_float_str(str(k)),v) for k,v in X[col].value_counts().to_dict().items()]))

    print(dict([(round_float_str(str(k)),v) for k,v in test_x[col].value_counts().to_dict().items()]))

    col_cnt = X[col].groupby(y).value_counts()

    print('---------------0--------------')

    print(dict([(round_float_str(str(k)),v) for k,v in col_cnt[0].to_dict().items()]))

    print('---------------1--------------')

    print(dict([(round_float_str(str(k)),v) for k,v in col_cnt[1].to_dict().items()]))

    print('-------------ratio-------------')

    print(dict([(round_float_str(str(k)),v) for k,v in (col_cnt[0]/col_cnt[1]).to_dict().items()]))

    print('-------------------------------------------------')