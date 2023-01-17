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
import matplotlib.pyplot as plt
df = pd.read_csv('../input/heart-disease-uci/heart.csv')

df.head()
df.describe()
df.isnull().sum()
import seaborn as sns

corr = df.corr()

plt.figure(figsize=(15,15))

sns.heatmap(corr, cmap='RdYlBu',annot = True)
sns.countplot(df['target'],palette='rainbow')
data = pd.get_dummies(df, columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

data[columns_to_scale] = sc.fit_transform(data[columns_to_scale])
data.head()
y=data['target']

X= data.drop(['target'],axis = 1)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

scores =[]

for k in range(1,21):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    score=cross_val_score(knn_classifier,X,y,cv=10)

    scores.append(score.mean())
plt.figure(figsize=(30,30))

plt.plot([k for k in range(1, 21)], scores, color = 'red')

for i in range(1,21):

    plt.text(i, scores[i-1], (i, scores[i-1]))

plt.xticks([i for i in range(1, 21)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')
model = KNeighborsClassifier(n_neighbors = 12)

score=cross_val_score(knn_classifier,X,y,cv=10)
score.mean()