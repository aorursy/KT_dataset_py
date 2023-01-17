# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import lightgbm as lgb

import gc; gc.enable()

from sklearn.model_selection import ShuffleSplit

from sklearn import *



df_members = pd.read_csv('../input/members.csv',dtype = {'msno':'category','city':np.uint8,'bd':np.uint8, 'gender':'category'})

df_members.head()

#df_members.dtypes

df_songs = pd.read_csv('../input/songs.csv', dtype = {'song_id':'category','genre_ids':'category',

                                                      'artist_name':'category','composer':'category',

                                                     'lyricist':'category', 'language':'category'})

df_songs.head()

#df_songs.dtypes
df_song_extra_info = pd.read_csv('../input/song_extra_info.csv',dtype = {'song_id':'category', 'name':'category',

                                                                        'isrc':'category'})

df_song_extra_info.head()

#df_song_extra_info.dtypes
df_train = pd.read_csv('../input/train.csv',dtype = {'msno':'category', 'song_id':'category',

                                                    'source_system_tab':'category',

                                                     'source_screen_name':'category',

                                                    'source_type':'category'})

df_train.head()

#df_train.dtypes
df_test = pd.read_csv('../input/test.csv', dtype = {'msno':'category', 'song_id':'category',

                                                   'source_system_tab':'category',

                                                     'source_screen_name':'category',

                                                    'source_type':'category'})

df_test.head()

#df_test.dtypes
# merging data sets

df_train_members = pd.merge(left=df_train, right = df_members, how = 'left', on = ['msno'])

df_train_members.head()

df_train_members.dtypes
del df_train, df_members; gc.collect();
# merging the data sets

df_songs_songs_extra_info = pd.merge(left = df_songs, right = df_song_extra_info, how = 'left', on = 'song_id')

df_songs_songs_extra_info.head()
del df_songs, df_song_extra_info; gc.collect();
df_train_final = pd.merge(left=df_train_members, right = df_songs_songs_extra_info, how = 'left', on = ['song_id'])

df_train_final.head()
#df_train_final.dtypes

for col in ['msno', 'song_id']:

    df_train_final[col]= df_train_final[col].astype('category')

df_train_final.dtypes

#df_train_final = (df_train_final,dtypes = {'msno':'category','song_id':'category'})

#df_train_final.dtypes
del df_train_members, df_songs_songs_extra_info; gc.collect();
# training the dataset

model = None 

for train_indices,val_indices in ShuffleSplit(n_splits=1,test_size = 0.2,train_size=0.8).split(df_train_final): 

    train_data = lgb.Dataset(df_train_final.drop(['song_id','target'],axis=1).loc[train_indices,:],label=df_train_final.loc[train_indices,'target'])

    val_data = lgb.Dataset(df_train_final.drop(['song_id','target'],axis=1).loc[val_indices,:],label=df_train_final.loc[val_indices,'target'])

    

    params = {

        'objective': 'binary',

        'metric': 'auc',

        'boosting': 'gbdt',

        'learning_rate': 0.05 , 

        'verbose': 0,

        'num_leaves': 108,

        'bagging_fraction': 0.95,

        'bagging_freq': 1,

        'bagging_seed': 1,

        'feature_fraction': 0.9,

        'feature_fraction_seed': 1,

        'max_bin': 128,

        'max_depth': 10,

        'num_rounds': 200,

        } 

    

    model = lgb.train(params, train_data, 200, valid_sets=[val_data])

# for testing set :-

df_members = pd.read_csv('../input/members.csv',dtype = {'msno':'category','city':np.uint8,'bd':np.uint8, 'gender':'category'})

#df_members.head()



df_songs = pd.read_csv('../input/songs.csv', dtype = {'song_id':'category','genre_ids':'category',

                                                      'artist_name':'category','composer':'category',

                                                     'lyricist':'category', 'language':'category'})

#df_songs.head()

df_song_extra_info = pd.read_csv('../input/song_extra_info.csv',dtype = {'song_id':'category', 'name':'category',

                                                                        'isrc':'category'})

#df_song_extra_info.head()

df_test = pd.read_csv('../input/test.csv', dtype = {'msno':'category', 'song_id':'category',

                                                   'source_system_tab':'category',

                                                     'source_screen_name':'category',

                                                    'source_type':'category'})

#df_test.head()

# merging data sets

df_test_members = pd.merge(left=df_test, right = df_members, how = 'left', on = ['msno'])

df_test_members.head()

df_test_members.dtypes



del df_test, df_members; gc.collect();



# merging the data sets

df_songs_songs_extra_info = pd.merge(left = df_songs, right = df_song_extra_info, how = 'left', on = 'song_id')

df_songs_songs_extra_info.head()



del df_songs, df_song_extra_info; gc.collect();



df_test_final = pd.merge(left=df_test_members, right = df_songs_songs_extra_info, how = 'left', on = ['song_id'])

df_test_final.head()



#converting object data types into categorical 

for col in ['msno', 'song_id']:

    df_test_final[col]= df_test_final[col].astype('category')

df_test_final.dtypes



del df_test_members, df_songs_songs_extra_info; gc.collect();



# using test dataset



predictions = model.predict(df_test_final.drop(['song_id'],axis=1))

df_test_final['target'] = predictions

df_test_final.drop(['msno','song_id','source_system_tab','source_screen_name', 'source_type', 'city','bd','gender',

                    'registered_via','registration_init_time','expiration_date','song_length','genre_ids',

                    'artist_name','composer','lyricist', 'language', 'name', 'isrc'],axis=1,inplace=True)

df_test_final.to_csv('submissions.csv',index=False)
output = pd.read_csv('submissions.csv')

output.head()