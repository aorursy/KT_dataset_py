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
import json

from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator

from sklearn.metrics import f1_score

import scipy
train = pd.read_csv('/kaggle/input/finec-1941-hw6/train.csv', encoding='utf-8')

test = pd.read_csv('/kaggle/input/finec-1941-hw6/test.csv', encoding='utf-8')



for col in (['production_companies', 'production_countries', 'spoken_languages', 'cast']):

    train[col] = train[col].apply(json.loads)

    test[col] = test[col].apply(json.loads)



train.head().T
x_train = train.drop(['index', 'genre_id'], axis=1)

y_train = train['genre_id']

x_test = test.drop('index', axis=1)



np.random.seed(514229)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33)
class ClusteringClassifier(BaseEstimator):

    def __init__(self, clustering):

        """Принимает объект, который имеет fit/predict и может каждому фильму определить какой-то номер кластера. 

        Затем для этого кластера выбирается наиболее популярный жанр, и все фильмы кластера приписываются этому жанру"""

        self.clustering = clustering

    

    def fit(self, x, y):

        self.clustering.fit(x)

        clustered = self.clustering.predict(x)

        self.cluster_labels = {}

        for c in np.unique(clustered):

            ys = y[clustered == c]

            most_popular_label = scipy.stats.mode(ys).mode[0]

            self.cluster_labels[c] = most_popular_label

        return self

    

    def predict(self, x):

        clustered = self.clustering.predict(x)

        prediction = [self.cluster_labels[c] for c in clustered]

        return np.array(prediction)

            

            
class ConstClustering(BaseEstimator):

    """относит все объекты к кластеру 0"""

        

    def fit(self, x):

        pass

    

    def predict(self, x):

        return np.zeros(len(x), dtype=int)
model = ClusteringClassifier(clustering=ConstClustering())



model.fit(x_train, y_train)

    

pred_train = model.predict(x_train)

pred_val = model.predict(x_val)



print('f1 train:', f1_score(y_train, pred_train, average='macro'))

print('f1 val:', f1_score(y_val, pred_val, average='macro'))
pred = model.predict(x_test)

pd.Series(pred, index=test['index'], name='genre_id').to_frame().to_csv(f"default_const_submit.csv")