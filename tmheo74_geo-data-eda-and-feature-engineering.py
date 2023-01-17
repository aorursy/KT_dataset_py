# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans



import xgboost as xgb

import lightgbm as lgb



# Set a few plotting defaults

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (12, 8)

plt.rcParams['font.size'] = 12



pd.options.display.max_rows = 10000

pd.options.display.max_columns = 10000

pd.options.display.max_colwidth = 1000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
def rmse_exp(y_true, y_pred):

    return np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))



def train_test_split(data, do_ohe=True):

    df = data.drop(['id','price','data'], axis=1).copy()

    cat_cols = df.select_dtypes('object').columns

    for col in cat_cols:

        if do_ohe:

            ohe_df = pd.get_dummies(df[[col]], prefix='ohe_'+col)

            df.drop(col, axis=1, inplace=True)

            df = pd.concat([df, ohe_df], axis=1)

        else:

            le = LabelEncoder()

            df[col] = le.fit_transform(df[col])



    train_len = data[data['data'] == 'train'].shape[0]

    X_train = df.iloc[:train_len]

    X_test = df.iloc[train_len:]

    y_train = data[data['data'] == 'train']['price']

    

    return X_train, X_test, y_train



def get_oof_lgb(X_train, y_train, X_test, lgb_param, verbose_eval=False, return_cv_score_only=False):



    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    oof = np.zeros(len(X_train))

    predictions = np.zeros(len(X_test))

    feature_importance_df = pd.DataFrame()



    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, y_train.values)):

        if verbose_eval > 0: print(f'Fold : {fold_ + 1}')

        trn_data = lgb.Dataset(X_train.iloc[trn_idx], label=y_train.iloc[trn_idx])

        val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx])



        num_round = 100000

        clf = lgb.train(lgb_param, trn_data, num_round, valid_sets=[trn_data, val_data],

                        verbose_eval=verbose_eval, early_stopping_rounds=200)

        oof[val_idx] = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)

        predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

        

        cv_fold_score = rmse_exp(y_train.iloc[val_idx], oof[val_idx])

        

        if verbose_eval > 0: print(f'Fold {fold_ + 1} / CV-Score: {cv_fold_score:.6f}')

        

        fold_importance_df = pd.DataFrame()

        fold_importance_df['feature'] = X_train.columns.tolist()

        fold_importance_df['importance'] = clf.feature_importance('gain')

        fold_importance_df['fold'] = fold_ + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    cv_score = rmse_exp(y_train, oof)

    print(f'CV-Score: {cv_score:.6f}')

    if return_cv_score_only: return cv_score

    else: return oof, predictions, cv_score, feature_importance_df

    

def plot_feature_importance(fi_df, num_feature=20):

    cols = (fi_df[['feature', 'importance']]

            .groupby('feature')

            .mean()

            .sort_values(by='importance', ascending=False)[:num_feature].index)

    best_features = fi_df.loc[fi_df.feature.isin(cols)]



    sns.barplot(x='importance', y='feature', data=best_features.sort_values(by='importance', ascending=False))

    plt.title('Feature Importances (averaged over folds)')

    plt.tight_layout()

    plt.show()

    

def plot_numeric_for_regression(df, field, target_field='price'):

    df = df[df[field].notnull()]



    fig = plt.figure(figsize = (16, 7))

    ax1 = plt.subplot(121)

    

    sns.distplot(df[df['data'] == 'train'][field], label='Train', hist_kws={'alpha': 0.5}, ax=ax1)

    sns.distplot(df[df['data'] == 'test'][field], label='Test', hist_kws={'alpha': 0.5}, ax=ax1)



    plt.xlabel(field)

    plt.ylabel('Density')

    plt.legend()

    

    ax2 = plt.subplot(122)

    

    df_copy = df[df['data'] == 'train'].copy()



    sns.scatterplot(x=field, y=target_field, data=df_copy, ax=ax2)

    

    plt.show()

    

def plot_categorical_for_regression(df, field, target_field='price', show_missing=True, missing_value='NA'):

    df_copy = df.copy()

    if show_missing: df_copy[field] = df_copy[field].fillna(missing_value)

    df_copy = df_copy[df_copy[field].notnull()]



    ax1_param = 121

    ax2_param = 122

    fig_size = (16, 7)

    if df_copy[field].nunique() > 30:

        ax1_param = 211

        ax2_param = 212

        fig_size = (16, 10)

    

    fig = plt.figure(figsize = fig_size)

    ax1 = plt.subplot(ax1_param)

    

    sns.countplot(x=field, hue='data', order=np.sort(df_copy[field].unique()), data=df_copy)

    plt.xticks(rotation=90, fontsize=11)

    

    ax2 = plt.subplot(ax2_param)

    

    df_copy = df_copy[df_copy['data'] == 'train']



    sns.boxplot(x=field, y=target_field, data=df_copy, order=np.sort(df_copy[field].unique()), ax=ax2)

    plt.xticks(rotation=90, fontsize=11)

    

    plt.show()

    

def load_original_data():

    train = pd.read_csv('../input/train.csv')

    test = pd.read_csv('../input/test.csv')



    train_copy = train.copy()

    train_copy['data'] = 'train'

    test_copy = test.copy()

    test_copy['data'] = 'test'

    test_copy['price'] = np.nan



    # remove outlier

    train_copy = train_copy[~((train_copy['sqft_living'] > 12000) & (train_copy['price'] < 3000000))].reset_index(drop=True)



    # concat train, test data to preprocess

    data = pd.concat([train_copy, test_copy], sort=False).reset_index(drop=True)

    data = data[train_copy.columns]



    data.drop('date', axis=1, inplace=True)

    data['zipcode'] = data['zipcode'].astype(str)



    # fix skew feature

    skew_columns = ['price']



    for c in skew_columns:

        data[c] = np.log1p(data[c])

        

    return data
data = load_original_data()



print(data.shape)

data.head()
X_train, X_test, y_train = train_test_split(data)

print(X_train.shape, X_test.shape)



lgb_param = {

    'objective': 'regression',

    'learning_rate': 0.05,

    'num_leaves': 15,

    'bagging_fraction': 0.7,

    'bagging_freq': 1,

    'feature_fraction': 0.7,

    'seed': RANDOM_SEED,

    'metric': ['rmse'],

}



oof, pred, cv_score, fi_df = get_oof_lgb(X_train, y_train, X_test, lgb_param)
plot_feature_importance(fi_df)
plot_categorical_for_regression(data, 'zipcode')
df = X_train

df['price'] = y_train



fig = plt.figure(figsize = (16, 12))



ax1 = plt.subplot(221)

sns.scatterplot(x='long', y='lat', hue='ohe_zipcode_98004', size='price', data=df, ax=ax1)



ax2 = plt.subplot(222)

sns.scatterplot(x='long', y='lat', hue='ohe_zipcode_98112', size='price', data=df, ax=ax2)



ax3 = plt.subplot(223)

sns.scatterplot(x='long', y='lat', hue='ohe_zipcode_98023', size='price', data=df, ax=ax3)



ax4 = plt.subplot(224)

sns.scatterplot(x='long', y='lat', hue='ohe_zipcode_98108', size='price', data=df, ax=ax4)



plt.show()
data = load_original_data()



data['zipcode-3'] = 'z_' + data['zipcode'].str[2:3]

data['zipcode-4'] = 'z_' + data['zipcode'].str[3:4]

data['zipcode-5'] = 'z_' + data['zipcode'].str[4:5]

data['zipcode-34'] = 'z_' + data['zipcode'].str[2:4]

data['zipcode-45'] = 'z_' + data['zipcode'].str[3:5]

data['zipcode-35'] = 'z_' + data['zipcode'].str[2:3] + data['zipcode'].str[4:5]



print(data.shape)

data.head()
data['zipcode'] = 'z_' + data['zipcode']

sns.scatterplot(x='long', y='lat', hue='zipcode', hue_order=np.sort(data['zipcode'].unique()), data=data);
sns.scatterplot(x='long', y='lat', hue='zipcode-3', hue_order=np.sort(data['zipcode-3'].unique()), data=data);
sns.scatterplot(x='long', y='lat', hue='zipcode-4', hue_order=np.sort(data['zipcode-4'].unique()), data=data);
sns.scatterplot(x='long', y='lat', hue='zipcode-5', hue_order=np.sort(data['zipcode-5'].unique()), data=data);
sns.scatterplot(x='long', y='lat', hue='zipcode-34', hue_order=np.sort(data['zipcode-34'].unique()), data=data);
sns.scatterplot(x='long', y='lat', hue='zipcode-45', hue_order=np.sort(data['zipcode-45'].unique()), data=data);
sns.scatterplot(x='long', y='lat', hue='zipcode-35', hue_order=np.sort(data['zipcode-35'].unique()), data=data);
X_train, X_test, y_train = train_test_split(data)

print(X_train.shape, X_test.shape)



lgb_param = {

    'objective': 'regression',

    'learning_rate': 0.05,

    'num_leaves': 15,

    'bagging_fraction': 0.7,

    'bagging_freq': 1,

    'feature_fraction': 0.7,

    'seed': RANDOM_SEED,

    'metric': ['rmse'],

}



oof, pred, cv_score, fi_df = get_oof_lgb(X_train, y_train, X_test, lgb_param)
plot_feature_importance(fi_df)
plot_categorical_for_regression(data, 'zipcode-35')
plot_categorical_for_regression(data, 'zipcode-5')
# pca for lat, long

data = load_original_data()



coord = data[['lat','long']]

pca = PCA(n_components=2)

pca.fit(coord)



coord_pca = pca.transform(coord)



data['coord_pca1'] = coord_pca[:, 0]

data['coord_pca2'] = coord_pca[:, 1]
sns.scatterplot(x='coord_pca2', y='coord_pca1', hue='price', data=data);
X_train, X_test, y_train = train_test_split(data)

print(X_train.shape, X_test.shape)



lgb_param = {

    'objective': 'regression',

    'learning_rate': 0.05,

    'num_leaves': 15,

    'bagging_fraction': 0.7,

    'bagging_freq': 1,

    'feature_fraction': 0.7,

    'seed': RANDOM_SEED,

    'metric': ['rmse'],

}



oof, pred, cv_score, fi_df = get_oof_lgb(X_train, y_train, X_test, lgb_param)
plot_feature_importance(fi_df)
plot_numeric_for_regression(data, 'coord_pca2')

plot_numeric_for_regression(data, 'coord_pca1')
inertia_arr = []



k_range = range(2, 16)



for k in k_range:

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED).fit(coord)

 

    # Sum of distances of samples to their closest cluster center

    interia = kmeans.inertia_

    print ("k:",k, " cost:", interia)

    inertia_arr.append(interia)

    

inertia_arr = np.array(inertia_arr)



plt.plot(k_range, inertia_arr)

plt.vlines(5, ymin=inertia_arr.min()*0.9999, ymax=inertia_arr.max()*1.0003, linestyles='--', colors='b')

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia');
data = load_original_data()



# kmeans for lat, long

kmeans = KMeans(n_clusters=5, random_state=RANDOM_SEED).fit(coord)

coord_cluster = kmeans.predict(coord)

data['coord_cluster'] = coord_cluster

data['coord_cluster'] = data['coord_cluster'].map(lambda x: 'c_' + str(x).rjust(2, '0'))



X_train, X_test, y_train = train_test_split(data)

print(X_train.shape, X_test.shape)



lgb_param = {

    'objective': 'regression',

    'learning_rate': 0.05,

    'num_leaves': 15,

    'bagging_fraction': 0.7,

    'bagging_freq': 1,

    'feature_fraction': 0.7,

    'seed': RANDOM_SEED,

    'metric': ['rmse'],

}



oof, pred, cv_score, fi_df = get_oof_lgb(X_train, y_train, X_test, lgb_param)
k_range = range(2, 80, 5)



for k in k_range:

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED).fit(coord)

    coord_cluster = kmeans.predict(coord)

    data['coord_cluster'] = coord_cluster

    data['coord_cluster'] = data['coord_cluster'].map(lambda x: str(x).rjust(2, '0'))

    

    X_train, X_test, y_train = train_test_split(data)



    lgb_param = {

        'objective': 'regression',

        'learning_rate': 0.05,

        'num_leaves': 15,

        'bagging_fraction': 0.7,

        'bagging_freq': 1,

        'feature_fraction': 0.7,

        'seed': RANDOM_SEED,

        'metric': ['rmse'],

    }



    print('K :', k)

    get_oof_lgb(X_train, y_train, X_test, lgb_param)

    print()
k_range = range(28, 37)



for k in k_range:

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED).fit(coord)

    coord_cluster = kmeans.predict(coord)

    data['coord_cluster'] = coord_cluster

    data['coord_cluster'] = data['coord_cluster'].map(lambda x: str(x).rjust(2, '0'))

    

    X_train, X_test, y_train = train_test_split(data)



    lgb_param = {

        'objective': 'regression',

        'learning_rate': 0.05,

        'num_leaves': 15,

        'bagging_fraction': 0.7,

        'bagging_freq': 1,

        'feature_fraction': 0.7,

        'seed': RANDOM_SEED,

        'metric': ['rmse'],

    }



    print('K :', k)

    get_oof_lgb(X_train, y_train, X_test, lgb_param)

    print()
# kmeans for lat, long

kmeans = KMeans(n_clusters=32, random_state=RANDOM_SEED).fit(coord)

coord_cluster = kmeans.predict(coord)

data['coord_cluster'] = coord_cluster

data['coord_cluster'] = data['coord_cluster'].map(lambda x: 'c_' + str(x).rjust(2, '0'))
sns.scatterplot(x='long', y='lat', hue='coord_cluster', hue_order=np.sort(data['coord_cluster'].unique()), data=data);
X_train, X_test, y_train = train_test_split(data)

print(X_train.shape, X_test.shape)



lgb_param = {

    'objective': 'regression',

    'learning_rate': 0.05,

    'num_leaves': 15,

    'bagging_fraction': 0.7,

    'bagging_freq': 1,

    'feature_fraction': 0.7,

    'seed': RANDOM_SEED,

    'metric': ['rmse'],

}



oof, pred, cv_score, fi_df = get_oof_lgb(X_train, y_train, X_test, lgb_param)
plot_feature_importance(fi_df)
df = X_train

df['price'] = y_train

sns.scatterplot(x='long', y='lat', hue='ohe_coord_cluster_c_11', data=df);
plot_categorical_for_regression(data, 'coord_cluster')
def haversine_array(lat1, lng1, lat2, lng2): 

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2)) 

    AVG_EARTH_RADIUS = 6371 # in km 

    lat = lat2 - lat1 

    lng = lng2 - lng1 

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2 

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d)) 

    return h
print(data['lat'].min(), data['lat'].max(), data['long'].min(), data['long'].max())



haversine_dist = haversine_array(data['lat'].min(), data['long'].min(), data['lat'].max(), data['long'].max())

print(f'max distance: {haversine_dist:.2f}km')
neighbor_df = pd.DataFrame()

lat2 = data['lat'].values

long2 = data['long'].values



lat1 = data.loc[0, 'lat'] # id = 0 house lat

long1 = data.loc[0, 'long'] # id = 0 house long

dist_arr = haversine_array(lat1, long1, lat2, long2)

neighbor_df = pd.DataFrame({

    'id': np.tile(np.array([data.loc[0, 'id']]), data.shape[0]),

    'neighbor_id': data['id'],

    'neighbor_lat': lat2,

    'neighbor_long': long2,

    'distance': dist_arr,

})

    

print(neighbor_df.shape)

neighbor_df.head()
neighbor_df['neighbor_10km'] = neighbor_df['distance'] <= 5

sns.scatterplot(x='neighbor_long', y='neighbor_lat', hue='neighbor_10km', data=neighbor_df);