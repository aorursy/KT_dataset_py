# Import modules

import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import tree

# from sklearn.metrics import accuracy_score

# import statsmodels as sm



# Figures inline and set visualization style

%matplotlib inline

sns.set()
# Import test and train datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# View first lines of training data

train.head(4)
# View first lines of test data

test.head()
# Let's now examine the datatypes in our training set

train.info()
# And let's take a loook at the summary statistics for our numeric variables:

train.describe()
sns.countplot(x='Survived', data=train)
# Here we create a new column 'Survived' which contains zeros for every row, i.e. everyone is predicted to have perished. 

test['Survived'] = 0

# We then simply combine that column with the relevant PassengerID to create a csv file to submit to Kaggle.

# Note: uncomment the next line to create your .csv!

# test[['PassengerId', 'Survived']].to_csv('data/predictions/no_survivors.csv', index = False)

# And now let's just drop the new Survived column as we won't be using it in that form again.

test.drop(['Survived'], axis=1, inplace=True)
sns.countplot(x='Sex', data=train)
sns.factorplot(x='Survived', col='Sex', kind='count', data=train)
train.groupby(['Sex']).Survived.sum()
print(train[train.Sex == 'female'].Survived.sum()/train[train.Sex == 'female'].Survived.count())

print(train[train.Sex == 'male'].Survived.sum()/train[train.Sex == 'male'].Survived.count())
sns.factorplot(x='Survived', col='Pclass', kind='count', data=train)
sns.factorplot(x='Survived', col='Embarked', kind='count', data=train)
train_drop = train.dropna()

sns.distplot(train_drop['Fare'], bins=20)
sns.boxplot(x='Survived', y='Fare', data=train)
train_drop = train.dropna()

sns.distplot(train_drop['Age'], bins=20)
sns.lmplot(x='Age', y='Fare', hue='Survived', data=train)
sns.pairplot(train_drop, hue='Survived', vars=['Fare', 'Age'])
# Store target variable of training data in a safe place

survived_train = train.Survived



# Concatenate training and test sets

data = pd.concat([train.drop(['Survived'], axis=1), test]) 
data.info()
# Impute missing numerical variables using the 

data['Age'] = data.Age.fillna(data.Age.median())

data['Fare'] = data.Fare.fillna(data.Fare.median())

data['Embarked'] = data.Embarked.fillna('S')

data.info()
# Here the tilda (~) simply reverses the True/False so that they are the right way around 

data['Has_Cabin'] = ~data.Cabin.isnull()
data['Fam_Size'] = data['SibSp'] + data['Parch'] + 1
# View head of 'Name' column

data.Name.head()
# Extract Title from Name, store in column and plot barplot

data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

sns.countplot(x='Title', data=data)

plt.xticks(rotation=45)
data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})

data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',

                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')

sns.countplot(x='Title', data=data)

plt.xticks(rotation=45)
# Drop columns that we aren't using and view head

data.drop(['Cabin', 'Name', 'SibSp', 'Parch', 'PassengerId', 'Ticket'], axis=1, inplace=True)

data.head()
data = pd.get_dummies(data, columns=['Embarked', 'Sex', 'Has_Cabin', 'Title'], drop_first=True)

data.head()
data_train = data.iloc[:891]

data_test = data.iloc[891:]
X = np.array(data_train)

array_test = np.array(data_test)

y = np.array(survived_train)
clf = tree.DecisionTreeClassifier(max_depth=2)

clf.fit(X, y)
import graphviz
# Some quick tidying to set things up for the plot:

feature_names = list(data)
# And now the graphviz plot

dot_data = tree.export_graphviz(clf, out_file=None, 

                         feature_names=feature_names, 

                         class_names = ['Died', 'Survived'],

                         filled=True, rounded=True,  

                         special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
# Make model predictions and store in 'Survived' column of test dataset

Y_pred = clf.predict(array_test)

test['Survived'] = Y_pred
# Note: uncomment the next line to create your .csv!

# test[['PassengerId', 'Survived']].to_csv('data/predictions/1st_dec_tree.csv', index=False)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)