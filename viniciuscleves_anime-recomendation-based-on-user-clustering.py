#imports

import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
rating = pd.read_csv('/kaggle/input/anime-recommendations-database/rating.csv')

rating.head(1)
anime0 = pd.read_csv('/kaggle/input/anime-recommendations-database/anime.csv')

anime0.head(1)
anime = anime0.drop(columns=['name', 'members', 'rating'])
# spand genre in columns, one for each genre

def func(x):

    if x['genre'] is np.nan:

        return x

    else:

        genres = list(map(lambda y: y.strip(), x['genre'].split(',')))

        for g in genres:

            x[g] = 1

        return x





anime2 = anime.apply(func, axis=1)

anime2.head(1)
# expand type in columns, one for each type

one_hot = pd.get_dummies(anime2['type'])

one_hot[one_hot == 0] = np.nan

anime3 = (anime2

          .drop(columns=['type', 'episodes', 'genre'])

          .join(one_hot, rsuffix='-type'))

anime3.head(1)
rating_anime = rating.join(anime3.set_index('anime_id'), on='anime_id')

rating_anime.head(1)
rating_anime.loc[rating_anime['rating'] == -1, 'rating'] = 5

rating_anime.head()
# anime3 is the dataframe joined before.

# All columns are anime properties, except anime_id.

attr = anime3.columns.tolist()

attr.remove('anime_id')



rating_anime[attr] = rating_anime[attr].mul(rating_anime['rating'], axis=0)

rating_anime.head(10)
users = (rating_anime

         .drop(columns=['anime_id', 'rating'])

         .groupby(by='user_id')

         .mean())

users.head()
users = users.fillna(value=0)

users.head()
%matplotlib inline



pca = PCA()

pca.fit(users)

acc_var = np.cumsum(pca.explained_variance_ratio_) 



plt.style.use('seaborn')

plt.plot(range(1, len(acc_var)+1), acc_var)

plt.title('PCA explained variance')

plt.xlabel('Number of Variables')

_ = plt.ylabel('Variance Explained')
number_of_components = 20

pca.set_params(n_components=number_of_components)

pca.fit(users)

users_pca = pca.transform(users)

users_pos_pca = pd.DataFrame(users_pca)

users_pos_pca['user_id'] = users.index

users_pos_pca = users_pos_pca.set_index('user_id')

users_pos_pca.head(1)
inertia = []

scores = []

for n_clusters in range(2, 12):

    kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)

    kmeans.fit(users_pos_pca)

    inertia.append(kmeans.inertia_)

plt.plot(range(2, 12), inertia)

plt.xlabel('Number of Clusters')

plt.ylabel('Quadratic Error')

_ = plt.title('K-means error vs number of Clusters')
#project the users feature vector in 3 dimensions

users_with_label = pd.DataFrame(PCA(n_components=3).fit_transform(users))

users_with_label['user_id'] = users.index

users_with_label = users_with_label.set_index('user_id')



#find each user's cluster

kmeans = KMeans(n_clusters=6, n_init=30, n_jobs=-1)

users_with_label['label'] = kmeans.fit_predict(users_pos_pca)

users_with_label.head()
fig = plt.figure()

ax = Axes3D(fig)



ax.scatter(users_with_label[0], users_with_label[1], users_with_label[2], c=users_with_label['label'].to_numpy(), cmap='viridis', s=10)

_ = plt.title('Clusters')
print('Cluster ID     Number of users in cluster')

for idx, val in (pd.get_dummies(users_with_label['label'])).sum().iteritems():

    print(f'{idx}              {val}')
rating_user = rating.join(users_with_label[['label']], on='user_id')

rating_user.loc[rating_user['rating'] == -1, 'rating'] = np.nan

rating_user.head(1)
groups = (rating_user[['anime_id', 'rating', 'label']]

          .groupby(by=['label', 'anime_id'])

          .rating.agg(['mean', 'count']))

groups.head(2)
groups['obj'] = groups['mean']*groups['count']

groups.head()
groups_obj = groups[['obj']].dropna()

groups_obj.head(2)
cats = groups_obj.index.get_level_values(0).unique().tolist()

rec = []

for cat in cats:

    rec.append(

        groups_obj

        .loc[cat]

        .sort_values(by='obj', ascending=False)

        .reset_index()

        .join(

            anime0[['name', 'anime_id']].set_index('anime_id'),

            on='anime_id')

        ['name']

        .rename(cat)

    )

rec = pd.concat(rec, axis=1)

rec.head(10)
for i in range(2, 20, 2):

    print('First {} recomendations: {} animes in total'

          .format(

              i,

              np.unique(

                  rec

                  .head(i)

                  .to_numpy())

              .shape[0]))
