# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn import preprocessing 

from sklearn.model_selection import GridSearchCV 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.ensemble import RandomForestRegressor



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pd.options.mode.chained_assignment = None



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

submit = pd.read_csv('../input/gender_submission.csv')
train.head(5)
train.info()
test.info()
train.describe()
train.describe()
data = train.append(test)

data
data.reset_index(inplace=True, drop=True)
data
sns.countplot(data['Survived'])
sns.countplot(data['Pclass'], hue=data['Survived'])
sns.countplot(data['Sex'], hue=data['Survived'])
sns.countplot(data['Embarked'], hue=data['Survived'])
g = sns.FacetGrid(data, col='Survived')

g.map(sns.distplot, 'Age', kde=False)
g = sns.FacetGrid(data, col='Survived')

g.map(sns.distplot, 'Fare', kde=False)
g = sns.FacetGrid(data, col='Survived')

g.map(sns.distplot, 'Parch', kde=False)
g = sns.FacetGrid(data, col='Survived')

g.map(sns.distplot, 'SibSp', kde=False)
data['Family_Size'] = data['Parch'] + data['SibSp']
g = sns.FacetGrid(data, col='Survived')

g.map(sns.distplot, 'Family_Size', kde=False)
data['Title1'] = data['Name'].str.split(", ", expand=True)[1]
data['Name'].str.split(", ", expand=True).head(3)
data['Title1'].head(3)
data['Title1'] = data['Title1'].str.split(".", expand=True)[0]
data['Title1'].head(3)
data['Title1'].unique()
pd.crosstab(data['Title1'],data['Sex']).T.style.background_gradient(cmap='summer_r')
pd.crosstab(data['Title1'],data['Survived']).T.style.background_gradient(cmap='summer_r')
data.groupby(['Title1'])['Age'].mean()
data.groupby(['Title1','Pclass'])['Age'].mean()
data['Title2'] = data['Title1'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],

         ['Miss','Mrs','Miss','Mr','Mr','Mrs','Mrs','Mr','Mr','Mr','Mr','Mr','Mr','Mrs'])
data['Title2'].unique()
data.groupby('Title2')['Age'].mean()
data.groupby(['Title2'])['Age'].mean()
data.groupby(['Title2','Pclass'])['Age'].mean()
pd.crosstab(data['Title2'],data['Sex']).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex
pd.crosstab(data['Title2'],data['Survived']).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex
list(data.groupby(['Title2','Pclass'])['Age'].mean().iteritems())[:3]
data.info()
data['Ticket_info'] = data['Ticket'].apply(lambda x : x.replace(".","").replace("/","").strip().split(' ')[0] if not x.isdigit() else 'X')
data['Ticket_info'].unique()
sns.countplot(data['Ticket_info'], hue=data['Survived'])
data['Embarked'] = data['Embarked'].fillna('S')
data.info()
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
data.info()
data['Cabin'].head(10)
data["Cabin"] = data['Cabin'].apply(lambda x : str(x)[0] if not pd.isnull(x) else 'NoCabin')
data["Cabin"].unique()
sns.countplot(data['Cabin'], hue=data['Survived'])
data['Sex'] = data['Sex'].astype('category').cat.codes

data['Embarked'] = data['Embarked'].astype('category').cat.codes

data['Pclass'] = data['Pclass'].astype('category').cat.codes

data['Title1'] = data['Title1'].astype('category').cat.codes

data['Title2'] = data['Title2'].astype('category').cat.codes

data['Cabin'] = data['Cabin'].astype('category').cat.codes

data['Ticket_info'] = data['Ticket_info'].astype('category').cat.codes
dataAgeNull = data[data["Age"].isnull()]

dataAgeNotNull = data[data["Age"].notnull()]

remove_outlier = dataAgeNotNull[(np.abs(dataAgeNotNull["Fare"]-dataAgeNotNull["Fare"].mean())>(4*dataAgeNotNull["Fare"].std()))|

                      (np.abs(dataAgeNotNull["Family_Size"]-dataAgeNotNull["Family_Size"].mean())>(4*dataAgeNotNull["Family_Size"].std()))                     

                     ]

rfModel_age = RandomForestRegressor(n_estimators=2000,random_state=42)

ageColumns = ['Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title1', 'Title2','Cabin','Ticket_info']

rfModel_age.fit(remove_outlier[ageColumns], remove_outlier["Age"])



ageNullValues = rfModel_age.predict(X= dataAgeNull[ageColumns])

dataAgeNull.loc[:,"Age"] = ageNullValues

data = dataAgeNull.append(dataAgeNotNull)

data.reset_index(inplace=True, drop=True)
dataTrain = data[pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])

dataTest = data[~pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])
dataTrain.columns
dataTrain = dataTrain[['Survived', 'Age', 'Embarked', 'Fare',  'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]

dataTest = dataTest[['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]
dataTrain
# rf = RandomForestClassifier(oob_score=True, random_state=1, n_jobs=-1)

# param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16, 20], "n_estimators": [50, 100, 400, 700, 1000]}

# gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)



# gs = gs.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])



# print(gs.best_score_)

# print(gs.best_params_)
from sklearn.ensemble import RandomForestClassifier

 

rf = RandomForestClassifier(criterion='gini', 

                             n_estimators=1000,

                             min_samples_split=12,

                             min_samples_leaf=1,

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1) 



rf.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])

print("%.4f" % rf.oob_score_)
pd.concat((pd.DataFrame(dataTrain.iloc[:, 1:].columns, columns = ['variable']), 

           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 

          axis = 1).sort_values(by='importance', ascending = False)[:20]
rf_res =  rf.predict(dataTest)

submit['Survived'] = rf_res

submit['Survived'] = submit['Survived'].astype(int)

submit.to_csv('submit.csv', index= False)
submit