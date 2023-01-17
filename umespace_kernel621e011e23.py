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
import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.model_selection import KFold
train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')

PassengerId=test['PassengerId']



train.info()
full_data = [train, test]



train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



for dataset in full_data:

    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1

    

for dataset in full_data:

    dataset['IsAlone']=0

    dataset.loc[dataset['FamilySize']==1,'IsAlone']==1

    

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    

train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train.Name.str.extract(' ([A-Za-z]+)\.', expand=False).value_counts()
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')  

        

for dataset in full_data:

    

    dataset['Pclass'] = dataset['Pclass'].map( {1: 'F', 2: 'S',3:'T'} )



for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

    
from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(categorical_features=[0],sparse=False)

train=pd.get_dummies(train,columns=['Pclass','Sex','Embarked','Title'])

test=pd.get_dummies(test,columns=['Pclass','Sex','Embarked','Title'])

drop_elements=['PassengerId','Name','SibSp','Ticket','Cabin']

train = train.drop(drop_elements, axis = 1)

test =test.drop(drop_elements,axis=1)



train.head()
import scipy.stats



age_std=scipy.stats.zscore(list(train['Age']))

train['Age']=pd.Series(age_std)

fare_std=scipy.stats.zscore(list(train['Fare']))

train['Fare']=pd.Series(fare_std)

age_std_test=scipy.stats.zscore(list(test['Age']))

test['Age']=pd.Series(age_std_test)

fare_std_test=scipy.stats.zscore(list(test['Fare']))

test['Fare']=pd.Series(fare_std_test)



test.head()
X_train = train.drop(['Survived'], axis=1)  # X_trainはtrainのSurvived列以外

Y_train = train['Survived']

from sklearn.model_selection import train_test_split



train_x, valid_x, train_y, valid_y = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)

type(train_x)
Results = pd.DataFrame({'Model': [],'Accuracy Score': []})
ele=['Fare','Parch','Pclass_T','Pclass_S','Sex_male','Embarked_Q','Has_Cabin','Embarked_C','IsAlone','Title_Rare']

train_x=train_x.drop(ele, axis=1) 

valid_x=valid_x.drop(ele, axis=1) 

test=test.drop(ele, axis=1) 
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

dt=DecisionTreeClassifier(max_depth=4,random_state=0)

dt.fit(train_x,train_y)

y_pred=dt.predict(valid_x)

res = pd.DataFrame({"Model":['DecisionTreeClassifier'],

                    "Accuracy Score": [accuracy_score(y_pred,valid_y)]})

Results = Results.append(res)
from sklearn.base import clone

from itertools import combinations

import numpy

from sklearn.metrics import accuracy_score





class SBS():

    """ Sequential backward selection """



    def __init__(self, estimator, k_features, scoring=accuracy_score,

                 test_size=0.25, random_state=1):

        self.scoring = scoring

        self.estimator = clone(estimator)

        self.k_features = k_features

        self.test_size = test_size

        self.random_state = random_state

        self.indices_ = None

        self.subsets_ = None



    def fit(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(

                X, y,

                test_size=self.test_size,

                random_state=self.random_state,

        )

        dim = X_train.shape[1]

        self.indices_ = tuple(range(dim))

        self.subsets_ = [self.indices_]

        score = self._calc_score(X_train, y_train, X_test, y_test, list(self.indices_))

        self.scores_ = [score]

        while dim > self.k_features:

            scores = []

            subsets = []

            for p in combinations(self.indices_, r=dim - 1):

                score = self._calc_score(X_train, y_train, X_test, y_test, p)

                scores.append(score)

                subsets.append(p)

            best = numpy.argmax(scores)

            self.indices_ = subsets[best]

            self.subsets_.append(self.indices_)

            dim -= 1

            self.scores_.append(scores[best])



        self.k_score_ = self.scores_[-1]

        return self



    def transform(self, X):

        return X.iloc(axis=0)[:, self.indices_]



    def _calc_score(self, X_train, y_train, X_test, y_test, indices):

        self.estimator.fit(X_train.iloc(axis=0)[:, indices], y_train)

        y_pred = self.estimator.predict(X_test.iloc(axis=0)[:, indices])

        score = self.scoring(y_test, y_pred)

        return score

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=2500, max_depth=4,random_state=0)

rf.fit(train_x,train_y)

y_pred=rf.predict(valid_x)

res = pd.DataFrame({"Model":['RandomForestClassifier'],

                    "Accuracy Score": [accuracy_score(y_pred,valid_y)]})

Results = Results.append(res)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(train_x,train_y)

y_pred=knn.predict(valid_x)

res = pd.DataFrame({"Model":['KNeighborsClassifier'],

                    "Accuracy Score": [accuracy_score(y_pred,valid_y)]})

Results = Results.append(res)
from sklearn.svm import SVC

svc = SVC(kernel='rbf',random_state=0 ,gamma=0.2,C=1.0)

svc.fit(train_x, train_y)

y_pred = svc.predict(valid_x)

res = pd.DataFrame({"Model":['SVC'],

                    "Accuracy Score": [accuracy_score(y_pred,valid_y)]})

Results = Results.append(res)
from matplotlib.colors import ListedColormap



def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):





    # setup marker generator and color map

    markers = ('s', 'x', 'o', '^', 'v')

    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])



    # plot the decision surface

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),

                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.ylim(xx2.min(), xx2.max())



    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],

                    alpha=0.8, c=cmap(idx),

                    marker=markers[idx], label=cl)



    # highlight test samples

    if test_idx:

        # plot all samples

        X_test, y_test = X[test_idx, :], y[test_idx]



        plt.scatter(X_test[:, 0],

                    X_test[:, 1],

                    c='black',

                    alpha=0.5,

                    linewidths=1,

                    marker='o',

                    s=55, label='test set')

          
from sklearn.linear_model import LogisticRegression

LogisticRegression(penalty='l1')

lr = LogisticRegression(penalty='l1', C=0.1)

lr.fit(train_x, train_y)

y_pred = lr.predict(valid_x)

res = pd.DataFrame({"Model":['LogisticRegression'],

                    "Accuracy Score": [accuracy_score(y_pred,valid_y)]})

Results = Results.append(res)

from matplotlib import pyplot

fig = pyplot.figure()

ax = pyplot.subplot(111)

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']

weights, params = [], []

for c in range(-10, 10):

    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)

    lr.fit(train_x, train_y)

    weights.append(lr.coef_[0])

    params.append(10**c)



weights = numpy.array(weights)



for column, color in zip(range(weights.shape[1]), colors):

    pyplot.plot(params, weights[:, column], label=train.columns[column+1],

                color=color)



pyplot.axhline(0, color='black', linestyle='--', linewidth=3)

pyplot.xlim([10**(-5), 10**5])

pyplot.ylabel('weight coefficient')

pyplot.xlabel('C')

pyplot.xscale('log')

pyplot.legend(loc='upper left')

ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)

pyplot.show()
train_x
from xgboost.sklearn import XGBClassifier

xgb = XGBClassifier(learning_rate=0.001,n_estimators=2500,

                                max_depth=4, min_child_weight=0,

                                gamma=0, subsample=0.7,

                                colsample_bytree=0.7,

                                scale_pos_weight=1, seed=27,

                                reg_alpha=0.00006)

xgb.fit(train_x, train_y)

y_pred = xgb.predict(valid_x)

res = pd.DataFrame({"Model":['XGBClassifier'],

                    "Accuracy Score": [accuracy_score(y_pred,valid_y)]})

Results = Results.append(res)
Results 
pred=xgb.predict(test)



StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': pred })

StackingSubmission.to_csv("StackingSubmission2.csv", index=False)
