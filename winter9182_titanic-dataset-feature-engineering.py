import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

print(train_df.shape)

print(test_df.shape)

train_df.head()
#check the survival with respect to Pclass

train_df[['Pclass','Survived']].groupby('Pclass').mean().sort_values(by = 'Survived', ascending = False)
train_df[['Sex','Survived']].groupby('Sex').mean().sort_values(by = 'Survived', ascending = False)
train_df[['SibSp','Survived']].groupby('SibSp').mean().sort_values(by = 'Survived', ascending = False)
train_df[['Parch','Survived']].groupby('Parch').mean().sort_values(by = 'Survived', ascending = False)
#converting categorial variable sex into encoding

labelencoder = LabelEncoder()

train_df['Sex'] = labelencoder.fit_transform(train_df['Sex'])

test_df['Sex'] = labelencoder.fit_transform(test_df['Sex'])

test_df.head()
bins = [0,16,32,48,64,200]

labels = [0,1,2,3,4]

train_df['Age Bin'] = pd.cut(train_df['Age'], bins=bins, labels=labels)

test_df['Age Bin'] = pd.cut(test_df['Age'], bins=bins, labels=labels)

train_df.head()
train_df['Family size'] = train_df['SibSp'] + train_df['Parch'] + 1

test_df['Family size'] = test_df['SibSp'] + test_df['Parch'] + 1

train_df[['Family size','Survived']].groupby('Family size').mean().sort_values(by = 'Survived', ascending = False)
train_df['Fam_type'] = pd.cut(train_df['Family size'], [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])

test_df['Fam_type'] = pd.cut(test_df['Family size'], [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])
combine = [train_df, test_df]

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Don', 'Sir', 'Jonkheer', 'Dona'],'Royalty')

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Dr','Major','Rev'],'Special')



train_df[['Title','Survived']].groupby(['Title'], as_index=False).mean()
train_df.head()
#first features extraction

y = train_df['Survived']

features = ['Pclass','Sex','Fam_type','Fare','Age Bin','Embarked']

X = train_df[features]

X.head()
numerical_col = ['Fare']

categorical_col = ['Pclass','Sex','Fam_type','Age Bin','Embarked']

num_trans = SimpleImputer(strategy = 'median')

cat_trans = Pipeline(steps = [

    ('imputer',SimpleImputer(strategy = 'most_frequent')),

    ('onehot',OneHotEncoder())

])

preprocessor = ColumnTransformer(

    transformers = [

        ('num',num_trans,numerical_col),

        ('cat',cat_trans,categorical_col)

])

titanic_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', RandomForestClassifier(random_state=0

                                                               ))

                                  ])

#titanic_pipeline.fit(X,y)

param_grid = {

    'model__max_depth': [2, 3, 4, 5],

    'model__min_samples_leaf': [3, 4, 5],

    'model__min_samples_split': [6, 8, 10, 12],

    'model__n_estimators': [100, 200, 300, 500]

}

search = GridSearchCV(titanic_pipeline, param_grid, n_jobs=-1)

search.fit(X, y)

print(search.best_params_)

#print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline, X, y, cv=10).mean()))
numerical_col = ['Fare']

categorical_col = ['Pclass','Sex','Fam_type','Age Bin','Embarked']

num_trans = SimpleImputer(strategy = 'median')

cat_trans = Pipeline(steps = [

    ('imputer',SimpleImputer(strategy = 'most_frequent')),

    ('onehot',OneHotEncoder())

])

preprocessor = ColumnTransformer(

    transformers = [

        ('num',num_trans,numerical_col),

        ('cat',cat_trans,categorical_col)

])

titanic_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', RandomForestClassifier(random_state=0,

                                                               max_depth = 5,

                                                               #min_samples_leaf = 5,

                                                               #min_samples_split = 12,

                                                               n_estimators = 500

                                                               ))

                                  ])

titanic_pipeline.fit(X,y)

print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline, X, y, cv=10).mean()))
X_test = test_df[features]

X_test.head()
predictions = titanic_pipeline.predict(X_test)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})

output.to_csv('my_submission2.csv', index=False)

print("Your submission was successfully saved!")