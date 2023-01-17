#https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import umap
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.max_columns', 500)
df_ratings=pd.read_csv('/kaggle/input/top-personality-dataset/2018_ratings.csv')
df_ratings['userid']=df_ratings['useri']
df_ratings.columns=df_ratings.columns.str.lstrip()#columns name contains empty spaces
df_ratings.head()
df_ratings.info()
df_per=pd.read_csv('/kaggle/input/top-personality-dataset/2018-personality-data.csv')
df_per.columns=df_per.columns.str.lstrip()#column names contains empty space
df_per.head()
df_per_melted=df_per.melt(id_vars=['userid', 'openness', 'agreeableness', 'emotional_stability',
       'conscientiousness', 'extraversion', 'assigned metric',
       'assigned condition', 'predicted_rating_1',
       'predicted_rating_2', 'predicted_rating_3',
       'predicted_rating_4', 'predicted_rating_5',
       'predicted_rating_6',  'predicted_rating_7',
       'predicted_rating_8', 'predicted_rating_9',
       'predicted_rating_10', 'predicted_rating_11',
       'predicted_rating_12', 'is_personalized',
       'enjoy_watching '], 
        value_vars=['movie_1',  'movie_2', 'movie_3', 'movie_4', 'movie_5','movie_6',
                 'movie_6', 'movie_7', 'movie_8', 'movie_9', 'movie_10', 'movie_11', 'movie_12'],
        var_name='movie_id', value_name="rates")
df_per_melted['movie_id']=df_per_melted['movie_id'].str.replace('movie_', '')
df_per_melted['movie_id']=pd.to_numeric(df_per_melted['movie_id'])
df_per_melted.head()
df=pd.merge(df_ratings, df_per_melted, on=['userid', 'movie_id'])
df.head()
df_num = df.select_dtypes(include=[np.float, np.int])
df_num
#https://github.com/lmcinnes/umap/blob/master/notebooks/UMAP%20usage%20and%20parameters.ipynb
def draw_umap(n_neighbors=3, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(df_num);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=u[:,0])
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=u[:,0])
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=u[:,0])
    plt.title(title, fontsize=18)
for n in (2, 5, 10, 20, 50, 100, 200):
    draw_umap(n_neighbors=n, title='n_neighbors = {}'.format(n))
for d in (0.0, 0.1, 0.25, 0.5, 0.8, 0.99):
    draw_umap(min_dist=d, title='min_dist = {}'.format(d))