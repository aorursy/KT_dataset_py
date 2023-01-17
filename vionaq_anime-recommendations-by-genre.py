import numpy as np

import pandas as pd

import itertools

import collections

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import jaccard_similarity_score 

from mpl_toolkits.mplot3d import Axes3D
rating = pd.read_csv('../input/rating.csv')

print (rating.head())
rating_clean = rating.drop(rating[rating.rating == -1].index)

print (rating_clean.head())



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

#ax.scatter(xs=rating_clean['rating'], ys=rating_clean['anime_id'],

#           zs=rating_clean['user_id'], s=20, c=None, depthshade=True)#, *args, **kwargs)

ax.scatter(xs=rating_clean['rating'], ys=rating_clean['anime_id'])#, *args, **kwargs)

plt.show()

    

import numpy as np

import pandas as pd

import itertools

import collections

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import jaccard_similarity_score 

from mpl_toolkits.mplot3d import Axes3D
rating = pd.read_csv('../input/rating.csv')

print (rating.head())
rating_clean = rating.drop(rating[rating.rating == -1].index)

print (rating_clean.head())



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs=rating_clean['user_id'], ys=rating_clean['anime_id']

           , s=20, c=None, depthshade=True)#, *args, **kwargs)

#ax.scatter(xs=rating_clean['rating'], ys=rating_clean['anime_id'])#, *args, **kwargs)

plt.show()

    
