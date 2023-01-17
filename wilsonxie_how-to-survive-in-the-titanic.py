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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
import seaborn as sns

import os

import matplotlib.pyplot as plt





from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin

import re

from sklearn.tree import DecisionTreeClassifier

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
titanic = train.copy()

titanic.head()
plt.figure(figsize=(4, 5)) #Counts for survived and non-survived passengers

sns.countplot(x='Survived', data=titanic)

plt.show()
cat_cols = ['Pclass', 'Sex', 'Embarked']#Counts for survived and non-survived passengers distinguished by Pclass, sex, and embark location



fig, ax = plt.subplots(1, 3, figsize=(20, 4))

for ind, val in enumerate(cat_cols):

    sns.countplot(x=val, hue='Survived', data=titanic, ax=ax[ind])

    ax[ind].legend(['Died', 'Survived'])
titanic[titanic['Survived'] == 1].Age.plot.kde() and titanic[titanic['Survived'] == 0].Age.plot.kde()
titanic[titanic['Survived'] == 1].Fare.plot.kde() and titanic[titanic['Survived'] == 0].Fare.plot.kde()
cat_cols = ['Parch', 'SibSp']#Counts for survived and non-survived passengers distinguished by the number of parents and siblings on board



fig, ax = plt.subplots(1, 2, figsize=(20, 4))

for ind, val in enumerate(cat_cols):

    sns.countplot(x=val, hue='Survived', data=titanic, ax=ax[ind])

    ax[ind].legend(['Died', 'Survived'])
titanic.describe() #Overview of the statistics
test.describe()
class StdScalerByGroup(BaseEstimator, TransformerMixin):

    

    def __init__(self):

        pass



    def fit(self, X, y=None):

        df = pd.DataFrame(X)

        

        self.grps_ = [df.groupby(df.columns[0]).mean().to_dict('index'), df.groupby(df.columns[0]).std().to_dict('index')]



        return self



    def transform(self, X, y=None):

        try:

            getattr(self, "grps_")

        except AttributeError:

            raise RuntimeError("You must fit the transformer before tranforming the data!")

        



        df = pd.DataFrame(X)



        for i in df.columns[1:]:

        	mean = df[df.columns[0]].map(self.grps_[0]).map(lambda x: x[i])

        	std = df[df.columns[0]].map(self.grps_[1]).map(lambda x: x[i])

        	df[i] = (df[i] - mean) / std

        

        return df.drop(df.columns[0], axis = 1)
def final(dataset):

    #Drop unnecessary columns

    titanic = dataset.copy()

    titanic = titanic.drop(['PassengerId', 'Ticket', 'Cabin'], axis = 1)

    

    #Impute missingness. Median for numerical and mode for categorical

    titanic['Age'] = titanic[['Age']].fillna(titanic.Age.median())

    

    freq = titanic.Embarked.mode().iloc[0]

    titanic['Embarked'] = titanic[['Embarked']].fillna(freq)

    

    titanic['Fare'] = titanic[['Fare']].fillna(titanic.Fare.median())

    

    #Feature engineering

    

    #name

    #Regex to filter the name of the passengers to only their title. We believe that the name of a passenger has no meaning but the title of he/she does.

    l = []

    for i in titanic.Name.values:

        l.append(re.findall('[A-Z]{1}[a-z]+\.', i)[0])

    titanic.Name = l

    

    #We also see there are many rare titles in the names. We want to combine those uncommon ones so then it would be easier to do the one-hot encoding.

    titanic['Name'] = titanic['Name'].replace(['Lady.', 'Countess.','Capt.', 'Col.', 'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Rare')



    titanic['Name'] = titanic['Name'].replace('Mlle.', 'Miss.')

    titanic['Name'] = titanic['Name'].replace('Ms.', 'Miss.')

    titanic['Name'] = titanic['Name'].replace('Mme.', 'Mrs.')

    

    title_mapping = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Rare.": 5}

    titanic['Name'] = titanic['Name'].map(title_mapping)

    titanic['Name'] = titanic['Name'].fillna(0)

    



    

    #family

    #We want to combine the siblings and parents together in a new column, called "Family".

    titanic['Family'] = titanic['SibSp'] + titanic['Parch']

    

    

    #sex

    #Converting categorical to numerical

    def sex(n):

        if n == 'male':

            return 1

        else:

            return 0

    titanic['Sex'] = titanic.Sex.apply(sex)

    

    #embarked

    places = titanic.Embarked.unique()

    pl = titanic['Embarked'].apply(lambda x: pd.Series(x == places, index=places, dtype=float))

    titanic = pd.concat([titanic, pl], axis = 1)

    

    #Drop unnecessary columns again

    titanic = titanic.drop([ 'Embarked'], axis = 1)



    #age

    #Z-values for Age column

    g = titanic[['Pclass', 'Age']]

    std = StdScalerByGroup().fit(g)

    titanic['Age'] = std.transform(g)

    

    return titanic
copy = final(titanic)





X = copy.drop('Survived', axis = 1)

y = copy.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1)

ml = Pipeline([('r', RandomForestClassifier(n_estimators=500,max_depth=6,min_samples_leaf=2,max_features='sqrt'))])

ml.fit(X_train, y_train)
ml.predict(X_test)
ml.score(X_test, y_test)
preds = ml.predict(final(test))
ml.score(X,y)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": preds

    })
submission.to_csv('submission.csv', index=False)