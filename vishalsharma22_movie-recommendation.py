# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np 
credits=pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv")

credits.head()
movies_df=pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")

movies_df.head()
credits.shape
movies_df.shape
credits_column_renamed=credits.rename(index=str,columns={"movie_id":"id"})

movies_df_merges=movies_df.merge(credits_column_renamed,on='id')

movies_df_merges.head()
movies_clean_df=movies_df_merges.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])

movies_clean_df.head()
movies_clean_df.shape
movies_clean_df.head(2)['overview']
from sklearn.feature_extraction.text import TfidfVectorizer



tvf=TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3),

            stop_words = 'english')



movies_clean_df['overview'] = movies_clean_df['overview'].fillna('')
tvf_matrix=tvf.fit_transform(movies_clean_df['overview'])
tvf_matrix
tvf_matrix.shape
from sklearn.metrics.pairwise import sigmoid_kernel
sig=sigmoid_kernel(tvf_matrix,tvf_matrix)
sig[1]
indices = pd.Series(movies_clean_df.index, index=movies_clean_df['original_title']).drop_duplicates()
indices.head()
indices['Avatar']
sig[0]
sorted(list(enumerate(sig[indices['Newlyweds']])), key=lambda x: x[1], reverse=True)
def give_rec(title, sig=sig):

    idx = indices[title] 

    sig_scores = list(enumerate(sig[idx]))

    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    sig_scores = sig_scores[1:11]

    movie_indices = [i[0] for i in sig_scores]

    return movies_clean_df['original_title'].iloc[movie_indices]
give_rec('Iron Man')