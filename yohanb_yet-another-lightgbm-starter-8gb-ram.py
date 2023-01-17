# Linear algebra:

import numpy as np

import pandas as pd

# Graphics:

import matplotlib.pyplot as plt

import seaborn as sns  

# Frameworks:

import lightgbm as lgb # LightGBM

# Utils:

import gc # garbage collector

%matplotlib inline

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
IDIR = '../input/' # main path

members = pd.read_csv(IDIR + 'members.csv')

songs = pd.read_csv(IDIR + 'songs.csv')

song_extra_info = pd.read_csv(IDIR + 'song_extra_info.csv')

train = pd.read_csv(IDIR + 'train.csv')

test = pd.read_csv(IDIR + 'test.csv')
# Adding songs' info:

train_aug1 = pd.merge(left=train, right=songs, on='song_id', how='left')

test_aug1 = pd.merge(left=test, right=songs, on='song_id', how='left')

# Adding extra info about songs:

train_aug2 = pd.merge(left=train_aug1, right=song_extra_info, on='song_id', how='left')

test_aug2 = pd.merge(left=test_aug1, right=song_extra_info, on='song_id', how='left')

del train_aug1, test_aug1

# Addind users' info:

train_aug3 = pd.merge(left=train_aug2, right=members, on='msno', how='left')

test_aug3 = pd.merge(left=test_aug2, right=members, on='msno', how='left')

del train_aug2, test_aug2

# Merging train and test data:

train_aug3.drop(['song_id'], axis=1, inplace=True)

train_aug3['set'] = 0

test_aug3.drop(['song_id'], axis=1, inplace=True)

test_aug3['set'] = 1

test_aug3['target'] = -1

all_aug = pd.concat([train_aug3, test_aug3], axis=0)

del train_aug3, test_aug3

gc.collect();
# source_system_tab encoding:

all_aug['source_system_tab'] = all_aug.source_system_tab.fillna('NA')

all_aug['source_system_tab'] = all_aug.source_system_tab.astype(np.str)

source_system_tab_le = LabelEncoder()

source_system_tab_le.fit(all_aug.source_system_tab)

all_aug['source_system_tab'] = source_system_tab_le.transform(all_aug.source_system_tab).astype(np.int8)

# source_screen_name encoding:

all_aug['source_screen_name'] = all_aug.source_screen_name.fillna('NA')

all_aug['source_screen_name'] = all_aug.source_screen_name.astype(np.str)

source_screen_name_le = LabelEncoder()

source_screen_name_le.fit(all_aug.source_screen_name)

all_aug['source_screen_name'] = source_screen_name_le.transform(all_aug.source_screen_name).astype(np.int8)

# source_type encoding:

all_aug['source_type'] = all_aug.source_type.fillna('NA')

all_aug['source_type'] = all_aug.source_type.astype(np.str)

source_type_le = LabelEncoder()

source_type_le.fit(all_aug.source_type)

all_aug['source_type'] = source_type_le.transform(all_aug.source_type).astype(np.int8)

# target encoding:

all_aug['target'] = all_aug.target.astype(np.int8)

# song_length encoding:

all_aug['song_length'] = all_aug.song_length.fillna(-1)

all_aug['song_length'] = all_aug.song_length.astype(np.int32)

# genre_ids encoding:

all_aug['genre_ids'] = all_aug.genre_ids.fillna('NA')

all_aug['genre_ids'] = all_aug.genre_ids.astype(np.str)

genre_ids_le = LabelEncoder()

genre_ids_le.fit(all_aug.genre_ids)

all_aug['genre_ids'] = genre_ids_le.transform(all_aug.genre_ids).astype(np.int16)

# artist_name encoding:

all_aug['artist_name'] = all_aug.artist_name.fillna('NA')

all_aug['artist_name'] = all_aug.artist_name.astype(np.str)

artist_name_le = LabelEncoder()

artist_name_le.fit(all_aug.artist_name)

all_aug['artist_name'] = artist_name_le.transform(all_aug.artist_name).astype(np.int32)

# composer encoding:

all_aug['composer'] = all_aug.composer.fillna('NA')

all_aug['composer'] = all_aug.composer.astype(np.str)

composer_le = LabelEncoder()

composer_le.fit(all_aug.composer)

all_aug['composer'] = composer_le.transform(all_aug.composer).astype(np.int32)

# lyricist encoding:

all_aug['lyricist'] = all_aug.lyricist.fillna('NA')

all_aug['lyricist'] = all_aug.lyricist.astype(np.str)

lyricist_le = LabelEncoder()

lyricist_le.fit(all_aug.lyricist)

all_aug['lyricist'] = lyricist_le.transform(all_aug.lyricist).astype(np.int32)

# language encoding:

all_aug['language'] = all_aug.language.fillna(-2)

all_aug['language'] = all_aug.language.astype(np.int8)

# name encoding:

all_aug['name'] = all_aug.name.fillna('NA')

all_aug['name'] = all_aug.name.astype(np.str)

name_le = LabelEncoder()

name_le.fit(all_aug.name)

all_aug['name'] = name_le.transform(all_aug.name).astype(np.int32)

# isrc encoding:

all_aug['isrc'] = all_aug.isrc.fillna('NA')

all_aug['isrc'] = all_aug.isrc.astype(np.str)

isrc_le = LabelEncoder()

isrc_le.fit(all_aug.isrc)

all_aug['isrc'] = isrc_le.transform(all_aug.isrc).astype(np.int32)

# city encoding:

all_aug['city'] = all_aug.city.astype(np.int8)

# bd encoding:

all_aug['bd'] = all_aug.bd.astype(np.int16)

# gender encoding:

all_aug['gender'] = all_aug.gender.fillna('NA')

all_aug['gender'] = all_aug.gender.astype(np.str)

gender_le = LabelEncoder()

gender_le.fit(all_aug.gender)

all_aug['gender'] = gender_le.transform(all_aug.gender).astype(np.int8)

# registered_via encoding:

all_aug['registered_via'] = all_aug.registered_via.astype(np.int8)

# registration_init_time encoding:

all_aug['registration_init_time'] = all_aug.registration_init_time.astype(np.int32)

# expiration_date encoding:

all_aug['expiration_date'] = all_aug.expiration_date.astype(np.int32)

# Info:

all_aug.info(max_cols=0)

all_aug.head(2)
all_aug['exp_reg_time'] = all_aug.expiration_date - all_aug.registration_init_time
gc.collect();

d_train = lgb.Dataset(all_aug[all_aug.set == 0].drop(['target', 'msno', 'id', 'set'], axis=1), 

                      label=all_aug[all_aug.set == 0].pop('target'))

ids_train = all_aug[all_aug.set == 0].pop('msno')



lgb_params = {

    'learning_rate': 1.0,

    'max_depth': 15,

    'num_leaves': 250, 

    'objective': 'binary',

    'metric': {'auc'},

    'feature_fraction': 0.8,

    'bagging_fraction': 0.75,

    'bagging_freq': 5,

    'max_bin': 100}

cv_result_lgb = lgb.cv(lgb_params, 

                       d_train, 

                       num_boost_round=5000, 

                       nfold=3, 

                       stratified=True, 

                       early_stopping_rounds=50, 

                       verbose_eval=100, 

                       show_stdv=True)



num_boost_rounds_lgb = len(cv_result_lgb['auc-mean'])

print('num_boost_rounds_lgb=' + str(num_boost_rounds_lgb))
%%time

ROUNDS = num_boost_rounds_lgb

print('light GBM train :-)')

bst = lgb.train(lgb_params, d_train, ROUNDS)

# lgb.plot_importance(bst, figsize=(9,20))

# del d_train

gc.collect()
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

feature_imp = pd.Series(dict(zip(d_train.feature_name, 

                                 bst.feature_importance()))).sort_values(ascending=False)

sns.barplot(x=feature_imp.values, y=feature_imp.index.values, orient='h', color='g')

plt.subplot(1,2,2)

train_scores = np.array(cv_result_lgb['auc-mean'])

train_stds = np.array(cv_result_lgb['auc-stdv'])

plt.plot(train_scores, color='green')

plt.fill_between(range(len(cv_result_lgb['auc-mean'])), 

                 train_scores - train_stds, train_scores + train_stds, 

                 alpha=0.1, color='green')

plt.title('LightGMB CV-results')

plt.show()