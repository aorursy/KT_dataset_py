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
df_train = pd.read_csv('../input/train.csv')

df_train.head()
revenue = df_train.revenue

df_train = df_train.drop(columns=['imdb_id','homepage','poster_path','status','revenue'])

df_train.fillna("[]",inplace=True)

df_train.head()
import ast



converted = df_train.genres.apply( lambda x: ast.literal_eval(str(x)))

genres_kinds = set()

for val in converted.values:

    for genre_names in val:

        genres_kinds.add((genre_names['id'],genre_names['name']))

  

genre_kinds_map = dict([ (k,v) for (v,k) in genres_kinds])

print(genre_kinds_map)



df_genres = pd.DataFrame(columns=['id']+list(genre_kinds_map.keys()))

df_genres = df_genres[list(genre_kinds_map.keys())].astype('uint8')

print(df_genres.columns)



for index,row in df_train.iterrows():

    genres_list = ast.literal_eval(str(row.genres))

    genres_dict = dict([ (genre['name'],1) for genre in genres_list ])

    genres_dict['id'] = row.id

    df_genres = df_genres.append(genres_dict,ignore_index=True)



df_genres.fillna(0,inplace=True)

df_genres.head()
df_train.drop(columns=['genres'],inplace=True)
df_train = df_train.merge(df_genres, on='id')

df_train.head()
df_subm = pd.read_csv('../input/sample_submission.csv')

df_subm.to_csv('submission.csv', index=False)

df_subm.head()