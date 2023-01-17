import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train = pd.read_json("../input/train.json")
train['inc_angle'] = train['inc_angle'].replace('na', 0.).astype(np.float32)
train.head()
def band_mean(band):

    band = np.array(band)

    return band.mean()

train['band_1_mean'] = train['band_1'].apply(band_mean)

train['band_2_mean'] = train['band_2'].apply(band_mean)
train.groupby('is_iceberg')['band_1_mean'].plot.hist(bins=50, alpha=0.6)
train.groupby('is_iceberg')['band_2_mean'].plot.hist(bins=50, alpha=0.6)
def band_median(band):

    band = np.array(band)

    return np.median(band)

train['band_1_median'] = train['band_1'].apply(band_median)

train['band_2_median'] = train['band_2'].apply(band_median)
train.groupby('is_iceberg')['band_1_median'].plot.hist(bins=50, alpha=0.6)
train.groupby('is_iceberg')['band_2_median'].plot.hist(bins=50, alpha=0.6)
def band_max(band):

    band = np.array(band)

    return band.max()

train['band_1_max'] = train['band_1'].apply(band_max)

train['band_2_max'] = train['band_2'].apply(band_max)
train.groupby('is_iceberg')['band_1_max'].plot.hist(bins=50, alpha=0.6)
train.groupby('is_iceberg')['band_2_max'].plot.hist(bins=50, alpha=0.6)
def band_min(band):

    band = np.array(band)

    return band.min()

train['band_1_min'] = train['band_1'].apply(band_min)

train['band_2_min'] = train['band_2'].apply(band_min)
train.groupby('is_iceberg')['band_1_min'].plot.hist(bins=50, alpha=0.6)
train.groupby('is_iceberg')['band_2_min'].plot.hist(bins=50, alpha=0.6)
train[train['inc_angle'] > 0].groupby('is_iceberg')['inc_angle'].plot.hist(bins=50, alpha=0.6)
def band_variance(band):

    band = np.array(band)

    return band.var()

train['band_1_variance'] = train['band_1'].apply(band_variance)

train['band_2_variance'] = train['band_2'].apply(band_variance)
train.groupby('is_iceberg')['band_1_variance'].plot.hist(bins=50, alpha=0.6)
train.groupby('is_iceberg')['band_2_variance'].plot.hist(bins=50, alpha=0.6)
def band_size(band):

    band = np.array(band)

    return np.sum(band > np.mean(band) + np.std(band)) / float(len(band))

train['band_1_size'] = train['band_1'].apply(band_size)

train['band_2_size'] = train['band_2'].apply(band_size)
train.groupby('is_iceberg')['band_1_size'].plot.hist(bins=50, alpha=0.6)
train.groupby('is_iceberg')['band_2_size'].plot.hist(bins=50, alpha=0.6)
import lightgbm as lgb

from sklearn.model_selection import train_test_split
X = train[['inc_angle',

           'band_1_mean', 'band_2_mean',

           'band_1_median', 'band_2_median',

           'band_1_min', 'band_2_min',

           'band_1_max', 'band_2_max',

           'band_1_variance', 'band_2_variance',

           'band_1_size', 'band_2_size']]

y = train['is_iceberg'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_dataset = lgb.Dataset(X_train, y_train)

test_dataset = lgb.Dataset(X_test, y_test)
params = {

        'objective': 'binary',

        'metric': 'binary_logloss',

        'boosting': 'gbdt',

        'learning_rate': 0.1,

        'num_rounds': 200,

        'early_stopping_rounds': 10,

}

model = lgb.train(params, train_dataset, valid_sets=test_dataset, verbose_eval=5)
lgb.plot_importance(model)