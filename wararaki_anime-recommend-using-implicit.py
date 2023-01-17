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
# !pip install implicit
rate_df = pd.read_csv('/kaggle/input/anime-recommendations-database/rating.csv')
anime_df = pd.read_csv('/kaggle/input/anime-recommendations-database/anime.csv')
print(rate_df.shape)
print(anime_df.shape)
tv_df = anime_df[anime_df.type == 'TV']
merge_df = pd.merge(rate_df, tv_df.drop('rating', axis=1), on='anime_id')
merge_df.shape
from implicit.als import AlternatingLeastSquares
import scipy
rate_df.user_id.unique().shape
merge_df.head(3)
data_df = merge_df[~(merge_df.rating == -1)][['user_id', 'anime_id', 'rating']]
data_df.shape
ratings = scipy.sparse.coo_matrix((
    data_df.rating.values,
    (data_df.anime_id.values, data_df.user_id.values)
)).tocsr()
ratings.shape
model = AlternatingLeastSquares(factors=150, iterations=30, regularization=0.01)
model.fit(ratings)
def show(similarities):
    result_df = pd.DataFrame()
    scores = []
    for similarity in similarities:
        result_df = pd.concat([result_df, tv_df[tv_df.anime_id == similarity[0]]])
        scores.append(similarity[1])
    result_df['similarity'] = scores
    display(result_df[['anime_id', 'name', 'similarity']].reset_index(drop=True))
anime_id = 20 # naruto
anime_id = 1535 # bleach
anime_id = 28977 # gintama
anime_id = 9253 # syutage
anime_id = 22 # tennis
anime_id = 170 # slam dunk

anime_ids = [20, 1535, 28977, 9253, 22, 170]

for anime_id in anime_ids:
    similarities = model.similar_items(anime_id, 21)
    display(f'anime_id={anime_id}')
    show(similarities)
# similarities
tv_df[tv_df.name.str.contains('Slam')]
