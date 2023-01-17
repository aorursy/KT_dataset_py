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
import matplotlib.pyplot as plt

from matplotlib import rcParams

from matplotlib.cm import rainbow

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.info()
df.head()

df.describe()
import seaborn as sns

#get correlations of each features in dataset

corrmat = df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

df.hist(figsize=(20,20))

#plot heat map

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")



"""

Generate HEAT Map.

sns.set_style('whitegrid')

plt.figure(figsize=(14,8))

sns.heatmap(df.corr(), annot = True, cmap='coolwarm',linewidths=.1)

plt.show()

"""

sns.set_style('whitegrid')

sns.countplot(x='target',data=df,palette='RdBu_r')
df = pd.get_dummies(df, drop_first=True)
df.head()
dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

dataset.head()
y = dataset['target']

X = dataset.drop(['target'], axis = 1)
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(df.drop('target', 1), df['target'], test_size = .2, random_state=10)



from sklearn.model_selection import cross_val_score

knn_scores = []

for k in range(1,21):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    score=cross_val_score(knn_classifier,X,y,cv=10)

    knn_scores.append(score.mean())
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
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

knn =KNeighborsClassifier()

params = {'n_neighbors':[i for i in range(1,33,2)]}
model = GridSearchCV(knn,params,cv=10)



model.fit(X_train,y_train)

model.best_params_





predict = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix

print('Accuracy Score: ',accuracy_score(y_test,predict))

print('Using k-NN we get an accuracy score of: ',

      round(accuracy_score(y_test,predict),5)*100,'%')
random_forest_model = RandomForestClassifier(max_depth=5)

random_forest_model.fit(X_train,y_train)

score=cross_val_score(random_forest_model,X_train,y_train,cv=10)



score.mean()


randomforest_classifier= RandomForestClassifier(n_estimators=10)

score=cross_val_score(randomforest_classifier,X,y,cv=10)
score.mean()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



import xgboost as xg

from xgboost import XGBClassifier



from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=0)

gb.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gb.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(gb.score(X_test, y_test)))
#GB after pruning

gb1 = GradientBoostingClassifier(random_state=0, max_depth=1)

gb1.fit(X_train, y_train)

print("****Gradient Boosting after Pruning using Max_depth****")

print("Accuracy on training set: {:.3f}".format(gb1.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(gb1.score(X_test, y_test)))
# Support Vector Machine
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))

print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=42)

mlp.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))

print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))