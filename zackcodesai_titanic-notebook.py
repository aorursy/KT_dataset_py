import os
import tqdm
import sklearn
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.svm import SVC
# from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.ensemble import (
        RandomForestClassifier, AdaBoostClassifier,
        GradientBoostingClassifier, ExtraTreesClassifier,
        BaggingClassifier, VotingClassifier
    )

%matplotlib inline
warnings.filterwarnings('ignore')

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
# for directory, _, files in os.walk('data'):
#     for f in files:
#         print(os.path.join(directory, f))
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
display(train_df.head())
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
display(test_df.head())
display(train_df.describe())
print('=' * 90)
display(test_df.describe())
display(train_df.info())
print('=' * 65)
display(test_df.info())
colors = plt.cm.PRGn
correlation = train_df.corr()
plt.figure(figsize = (16,12))
plt.title("Correlation Matrix: Stock Data", y = 1.02, size = 16)
sns.heatmap(
    correlation, linewidths = 0.12, 
    vmax = 1.0, square = True, 
    cmap = colors, linecolor = 'white',
    annot = True
    )
plt.show()
test_id = test_df['PassengerId']
test_id[:5]
dataList = [train_df, test_df]
for df in dataList:
    df['hasCabin'] = df['Cabin'].apply(lambda x : 0 if type(x) == float else 1)
    df.drop('Cabin', axis = 1, inplace = True)
ageVals = []

for df in dataList:
    ageVals.append([
        df['Age'].mean(), 
        df['Age'].std()
        ])

ageVals.append(
    list(
        np.random.randint(
            ageVals[0][0] - ageVals[0][1], 
            ageVals[0][0] + ageVals[0][1], 
            size = train_df['Age'].isnull().sum()
            )))

ageVals.append(
    list(
        np.random.randint(
            ageVals[1][0] - ageVals[1][1], 
            ageVals[1][0] + ageVals[1][1], 
            size = test_df['Age'].isnull().sum()
            )))
def get_title(name):
    for word in name.split():
        if word.endswith('.'):
            return(word[:-1])
for ds in dataList:
    ds['Title'] = ds['Name'].apply(get_title)
    ds['nameLen'] = ds['Name'].apply(len)
    ds['Title'].replace((['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'), inplace = True)
    ds['Title'].replace('Mlle', 'Miss', inplace = True)
    ds['Title'].replace('Ms', 'Miss', inplace = True)
    ds['Title'].replace('Mme', 'Mrs', inplace = True)
    display(ds['Title'])
title_maps = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for ds in dataList:
    ds['Title'] = ds['Title'].map(title_maps)
    ds['Title'] = ds['Title'].fillna(0)
    ds['Title'] = ds['Title'].astype(int)
train_df.info()
train_df['Age'][np.isnan(train_df['Age'])] = ageVals[2]
test_df['Age'][test_df['Age'].isnull()] = ageVals[3]

for df in dataList:
    display(pd.cut(df['Age'], 8))
    df['Age'] = pd.cut(df['Age'], 8, labels=[0,1,2,3,4,5,6,7])
    print('=' * 45)
train_df.dropna(inplace = True)

for df in dataList:
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    df['familySize'] = df['SibSp'] + df['Parch'] + 1
test_df['Fare'] = test_df['Fare'].fillna(32.2)

for df in dataList:
    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[
        (df['Fare'] > 7.91)
        & (df['Fare'] <= 14.45),
        'Fare'] = 1
    df.loc[
        (df['Fare'] > 14.45)
        & (df['Fare'] <= 31),
        'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3

    df['Fare'] = df['Fare'].astype(int)
dropVals = ['PassengerId', 'Name', 'Ticket', 'SibSp']

for df in dataList:
    df.drop(dropVals, axis = 1, inplace = True)
colors = plt.cm.PRGn
correlation = train_df.corr()
plt.figure(
    figsize = (16,12)
    )
plt.title(
    "Correlation Matrix: After Feature Extraction",
    y = 1.02,
    size = 16
    )
sns.heatmap(
    correlation, linewidths = 0.12, 
    vmax = 1.0, square = True, 
    cmap = colors, linecolor = 'white',
    annot = True
    )
plt.show()
# pair_plot = sns.pairplot(train_df, hue='Survived', palette= 'deep', size = 1.4, diag_kind= 'kde', diag_kws= dict(shade=True), plot_kws= dict(s=10))
# pair_plot.set(xticklabels=[])
x_train = train_df.drop(['Survived'], axis = 1).values
y_train = train_df['Survived'].ravel()

x_test = test_df.values
display(x_train)
print("=" * 50)
display(y_train)
print("=" * 75)
display(x_test)
for df in dataList:
    display(df.head())
    print('=' * 70)
cart = DecisionTreeClassifier()
seed = 14

params = [{
    'n_estimators': 100,
    'random_state': seed
}, {
    'base_estimator': cart,
    'n_estimators': 100,
    'random_state': seed
}, {
    'n_estimators': 100,
    'random_state': seed
}]

models = [RandomForestClassifier, BaggingClassifier, AdaBoostClassifier]
kfold = KFold(n_splits=10, random_state=seed)

results = []

for mod, param in zip(models, params):
   model = mod(**param)
   res = cross_val_score(model, x_train, y_train, cv=kfold)
   results.append([str(mod).split(sep='.')[-1][:-2], res.mean()])
display(results)
adaB = AdaBoostClassifier()
adaB.fit(x_train, y_train)
y_pred = adaB.predict(x_test)
y_pred
dfOut = pd.DataFrame({'PassengerId': test_id,
    'Survived': y_pred
})
dfOut.to_csv("gender_submission.csv", index=False)
