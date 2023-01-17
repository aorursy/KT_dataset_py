# Import modules

import logging, os, pandas as pd, numpy as np

import matplotlib.pyplot as plt, seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve
%%time

# Load datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
# Explore features stats

# Tổng quan dataset

train.describe()
test.describe()
# Features histogram in train dataset

# Histogram của 100 biến đầu tiên trong train dataset

slot = 1

plt.figure(figsize=(30, 30))

for i in range(2,102):

    plt.subplot(10, 10, slot)

    train.iloc[:, i].hist()

    slot += 1
# Histogram của 100 biến tiếp theo trong train dataset

slot = 1

plt.figure(figsize=(30, 30))

for i in range(102, 202):

    plt.subplot(10, 10, slot)

    train.iloc[:, i].hist()

    slot += 1
# Target distribution

sns.countplot(train['target'])

print(train.target.value_counts(normalize=True))
# Distribution of target within each feature

# Phân bố target theo từng biến

def plot_feat_dist(df1, df2, label1, label2, feat):

    i = 0

    sns.set_style('whitegrid')

    fig, ax = plt.subplots(10, 10, figsize=(30, 30))

    

    for feat in feat:

        i += 1

        plt.subplot(10, 10, i)

        sns.distplot(df1[feat], hist=False, label=label1)

        sns.distplot(df2[feat], hist=False, label=label2)

        plt.xlabel(feat)

    plt.show()
# First 100 features

# Phân bố target trong 100 biến đầu

train0 = train.loc[train.target == 0]

train1 = train.loc[train.target == 1]

feat = train.columns.values[2:102]



plot_feat_dist(train0, train1, '0', '1', feat)
# Next 100 features

# Phân bố target trong 100 biến sau

feat = train.columns.values[102:202]

plot_feat_dist(train0, train1, '0', '1', feat)
# Distribution of statistical value per row/column by train and test dataset.

# Đồ thị phân bố giá trị của hàng/cột trong train và test dataset.

def train_test_dist(agg):

    features = train.columns.values[2:202]

    plt.figure(figsize=(30,8))

    sns.set_style('whitegrid')

    

    plt.subplot(1,2,1)

    sns.distplot(train[features].apply(func=agg, axis=1), kde=True, color='g', bins=120, label='train')

    sns.distplot(test[features].apply(func=agg, axis=1), kde=True, color='b', bins=120, label='test')

    plt.legend()

    plt.title('Distribution of {} values per row in the train and test set'.format(agg))



    plt.subplot(1,2,2)

    sns.distplot(train[features].apply(func=agg, axis=0), kde=True, color='g', bins=120, label='train')

    sns.distplot(test[features].apply(func=agg, axis=0), kde=True, color='b', bins=120, label='test')

    plt.legend()

    plt.title('Distribution of {} values per column in the train and test set'.format(agg))
train_test_dist('mean')
train_test_dist('std')
train_test_dist('min')
train_test_dist('max')
train_test_dist('skew')
train_test_dist('kurtosis')
# Distribution of statistical value per row/column in the train dataset, grouped by value of target.

# Đồ thị phân bố giá trị của hàng/cột nhóm theo target trong train dataset.

def train_dist(agg):

    t0 = train.loc[train['target'] == 0]

    t1 = train.loc[train['target'] == 1]

    features = train.columns.values[2:202]

    plt.figure(figsize=(30,12))

    sns.set_style('whitegrid')

    

    plt.subplot(1,2,1)

    sns.distplot(t0[features].apply(func=agg, axis=1), kde=True,bins=120, color='r',label='target = 0')

    sns.distplot(t1[features].apply(func=agg, axis=1), kde=True,bins=120, color='darkblue', label='target = 1')

    plt.legend()

    plt.title('Distribution of {} values per row in the train set'.format(agg))

    

    plt.subplot(1,2,2)

    sns.distplot(t0[features].apply(func=agg, axis=0), kde=True,bins=120, color='r', label='target = 0')

    sns.distplot(t1[features].apply(func=agg, axis=0), kde=True,bins=120, color='darkblue', label='target = 1')

    plt.title('Distribution of {} values per column in the train set'.format(agg))

    plt.legend()
train_dist('mean')
train_dist('std')
train_dist('min')
train_dist('max')
train_dist('skew')
train_dist('kurtosis')
# Features correlation -- Tương quan các biến

features = [c for c in train.columns.values if c not in ['ID_code', 'target']]

corr = train[features].corr().abs().unstack().sort_values().reset_index()

corr = corr[corr.level_0 != corr.level_1]

corr.tail()
# Duplicated value -- Kiểm tra xem trong các biến có xuất hiện nhiều giá trị trùng nhau không

def dupl_max_df(df):

    dupl_max = []

    

    for feature in features:

        values = df[feature].value_counts()

        dupl_max.append([feature, values.max(), values.idxmax()])



    dupl_max = pd.DataFrame(dupl_max, columns=['feature', '# dupl', 'value dupl']).sort_values(by='# dupl', ascending=False).T

    return dupl_max
dupl_max_df(train)
dupl_max_df(test)
%%time

# Feature Engineering

# Tạo thêm biến

for df in [train, test]:

    df['sum'] = df[features].sum(axis=1)  

    df['min'] = df[features].min(axis=1)

    df['max'] = df[features].max(axis=1)

    df['mean'] = df[features].mean(axis=1)

    df['std'] = df[features].std(axis=1)

    df['skew'] = df[features].skew(axis=1)

    df['kurt'] = df[features].kurtosis(axis=1)

    df['med'] = df[features].median(axis=1)
train.head()
test.head()
train.shape[1], test.shape[1]
features = [c for c in train.columns.values if c not in ['ID_code', 'target']]

target = train['target']



params = {

    'bagging_freq': 2,

    'bagging_fraction': 0.8,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.1403,

    'learning_rate': 0.07,

    'max_depth': 7,  

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 19,

    'num_leaves': 14,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': -1,

    'lambda_l1': 1.7916,

    'lambda_l2': 4.7454,

    'min_gain_to_split': 0.0319

}
%%time

# Quick training -- Train nhanh model trên train dataset với 100 iterations

val_size = 0.3

X_train, X_val, y_train, y_val = train_test_split(train[features], train['target'], test_size = val_size, random_state=42) # Chia file train dataset



iterations = 100

val_pred = np.zeros([int(len(train)*val_size), len(features)])

test_pred = np.zeros([200000, len(features)])



i = 0

for feature in features: # loop over all features -- Train model với từng biến để tăng tốc độ train mà vẫn đảm bảo hiệu quả

    print(feature)

    feat = [feature]

    train_data = lgb.Dataset(X_train[feat], y_train)

    gbm = lgb.train(params, train_data, iterations, verbose_eval=-1)

    val_pred[:, i] = gbm.predict(X_val[feat], num_iteration=gbm.best_iteration)

    test_pred[:, i] = gbm.predict(test[feat], num_iteration=gbm.best_iteration)

    i += 1



score = roc_auc_score(y_val, (val_pred).sum(axis=1))

print('CV score: ', score)
%%time

# Final training and predicting -- Training lần cuối dự đoán target cho test dataset

folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=42)



oof = np.zeros(len(train))

pred = np.zeros(len(test))

feature_important_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['target'])):

    print('Fold: {}'.format(fold_))

    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])

    

    num_round = 100000

    clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000, early_stopping_rounds=3000)

    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    

    fold_important_df = pd.DataFrame()

    fold_important_df['feature'] = features

    fold_important_df['importance'] = clf.feature_importance()

    fold_important_df['fold'] = fold_ + 1

    feature_important_df = pd.concat([feature_important_df, fold_important_df], axis=0)

    

    pred = clf.predict(test[features], num_iteration=clf.best_iteration)/folds.n_splits # Kết quả dự đoán target test dataset

    

print('CV score: {:.5f}'.format(roc_auc_score(target, oof)))
submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = pred

submission.to_csv('submission.csv', index=False)
submision.head()