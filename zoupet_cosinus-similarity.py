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
df = pd.read_csv('../input/movie_metadata.csv')
df.columns
df.head()
df_select = df[['director_name','actor_1_name','actor_2_name','actor_3_name','genres']]
unique_genre_labels = set()

for genre_flags in df_select.genres.str.split('|').values:

    unique_genre_labels = unique_genre_labels.union(set(genre_flags))

for label in unique_genre_labels:

    df_select['Genre='+label] = df_select.genres.str.contains(label).astype(int)

df_select = df_select.drop('genres', axis=1)

#df_select['name'] = df_select['director_name'] + '|' + df_select['actor_1_name'] + '|' + df_select['actor_2_name'] + '|' + df_select['actor_3_name']



#df_select.name.unique()

df_select.fillna('UNKNOWN',inplace=True)
#unique_name_labels = set()

#for name_flags in df_select.name.str.split('|').values:

    #print(name_flags)

#    unique_name_labels = unique_name_labels.union(set(name_flags))

#for label in unique_name_labels:

#    df_select['name='+label] = df_select.name.str.contains(label).astype(int)

    

df_select = df_select.drop('director_name', axis=1)

df_select = df_select.drop('actor_1_name', axis=1)

df_select = df_select.drop('actor_2_name', axis=1)

df_select = df_select.drop('actor_3_name', axis=1)
#df_select = df_select.drop('name', axis=1)
#df[df['movie_title'] == 'Hesher\xa0']

#df['movie_title'].unique()
from sklearn.metrics.pairwise import cosine_similarity



index_input = 5

df_score = pd.DataFrame()

for i in range(0,df_select.shape[0]):

    score = cosine_similarity(df_select.iloc[index_input].reshape(1, -1) ,df_select.iloc[i].reshape(1, -1))[0]

    #print(df['movie_title'].iloc[index_input],df['movie_title'].iloc[i],score[0])

    df_score = pd.concat([df_score,pd.DataFrame(score)])
df_score.reset_index(drop=True,inplace=True)

df_score.columns = ['Similarity']
index_simi_top = df_score.sort('Similarity',ascending=False)[0:5].index.values
df_score.sort('Similarity',ascending=False)[0:5]
df['movie_title'].iloc[index_simi_top]