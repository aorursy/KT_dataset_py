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
! ls ../input/heart-disease-uci/heart.csv
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/heart-disease-uci/heart.csv')

data.head()
data.shape
data.info()
data.describe()
#checking if threre is any null values present in dataset or not 

data.isnull().sum()
corr = data.corr()

plt.plot(figsize= (20,20))

sns.heatmap(corr, annot=True)
sns.countplot(x='target',data=data ,palette='winter_r')

plt.xlabel('heart disease outcome')

plt.ylabel('count of patient')
# checking counts of true and false heartprediction ,we have balance dataset so no need to upsamping and down sampling. 

target_true= len(data.loc[data['target']==1])

target_false= len(data.loc[data['target']==0])

print(target_true,target_false)

from sklearn.model_selection import  train_test_split

from sklearn.preprocessing import StandardScaler

sc_x= StandardScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

data[columns_to_scale] = sc_x.fit_transform(data[columns_to_scale])

data.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

y = data['target']

X = data.drop(['target'], axis = 1)
#y is series hence convering into 2d array using reshape 

y1 =y.ravel().reshape(-1,1)

y1.shape
# checking best nearest naighbours values

from sklearn.model_selection import cross_val_score

knn_score= []

for i in range(1,20):

    knn_classifier = KNeighborsClassifier(n_neighbors=i)

    score = cross_val_score(knn_classifier,X,y1,cv=10)

    knn_score.append(score.mean())

    
plt.plot([i for i in range(1,20)],knn_score,'b*--')

plt.xticks([i for i in range(1, 20)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')


knn_classifier = KNeighborsClassifier(n_neighbors = 5)

score=cross_val_score(knn_classifier,X,y1,cv=10)



score.mean()
##random forest

from sklearn.ensemble import RandomForestClassifier





randomforest_classifier= RandomForestClassifier(n_estimators=10)

score=cross_val_score(randomforest_classifier,X,y,cv=10)

score.mean() 