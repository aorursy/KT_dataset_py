import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import rcParams

from matplotlib.cm import rainbow

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('../input/heart-disease-dataset/Heart Disease Dataset.csv')

df.head(5)
df.info()
df.describe()
import seaborn as sns

#get correlations of each features in dataset

corrmat = df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#plt.figure(figsize=(10,10))

df.hist(figsize=(10,10))

sns.set_style('whitegrid')

sns.countplot(x='target',data=df,palette='RdBu_r')
dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()



columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
dataset.head()
y = dataset['target']

X = dataset.drop(['target'], axis = 1)
from sklearn.model_selection import cross_val_score

knn_scores = []



for k in range(1,21):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    score=cross_val_score(knn_classifier,X,y,cv=10)

    knn_scores.append(score.mean())
plt.figure(figsize=(15,15))

plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')



for i in range(1,21):

    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))





plt.xticks([i for i in range(1, 21)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')



plt.title('K Neighbors Classifier scores for different K values')

knn_classifier = KNeighborsClassifier(n_neighbors = 12)

score=cross_val_score(knn_classifier,X,y,cv=10)

score.mean()
from sklearn.ensemble import RandomForestClassifier
randomforest_scores=[]

#estimators= (i for i in range(1,20))

for n_est in range(1,20):

    randomforest_classifier= RandomForestClassifier(n_estimators= n_est)

    

    score=cross_val_score(randomforest_classifier,X,y,cv=10)
score.mean()
randomforest_scores.append(score.mean())