# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/spotifyclassification/data.csv')

data.head()
data.columns
data = data[['acousticness', 'danceability', 'duration_ms', 'energy',

       'instrumentalness', 'key', 'liveness', 'loudness', 'mode',

       'speechiness', 'tempo', 'time_signature', 'valence',

       'song_title', 'artist']]

X = data.drop(['song_title','artist'],axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



scores = []



for cluster in range(5,20):

    kmeans = KMeans(n_clusters=cluster)

    kmeans.fit(X)

    scores.append(silhouette_score(X,kmeans.labels_))
import seaborn as sns

sns.set_style('whitegrid')

sns.lineplot(range(5,20),scores)
kmeans = KMeans(n_clusters=7)

kmeans.fit(X)
kmeans.cluster_centers_
kmeans.cluster_centers_.shape
data['song_title'].tolist()
from sklearn.utils import shuffle



print("Enter Song Title:")

song = input(':: ')



#Make sure the song is in the data

if song not in data['song_title'].tolist():

    print('\nSong not found.')

else:

    #access the label of the song

    label = int(data[data['song_title']==song]['label'])

    

    #get songs that have the same label

    results = data[data['label']==label]

    

    #shuffle the results, because we will be getting the 

    #top 5 songs in each cluster

    results = shuffle(results)

    

    #reset the index because the indices have been shuffled

    results = results.reset_index()

    #resetting the index pops the old one out,let's delete it

    results = results.drop('index',axis=1)

    

    print('\nOther songs you might like:')

    for i in range(5): #for each index in top

        print(results.loc[i,'song_title']+' by '+results.loc[i,'artist'])

        

#try 'Hotline Bling'