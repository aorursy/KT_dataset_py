import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# Vamos iniciar o notebook importanto o Dataset

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# Podemos observar as primeiras linhas dele.

titanic_df.head()
titanic_df[pd.isnull(titanic_df['Age'])]
age_median = titanic_df['Age'].median()

print(age_median)
titanic_df['Age'] = titanic_df['Age'].fillna(age_median)

test_df['Age'] = test_df['Age'].fillna(age_median)
import seaborn as sns

sns.countplot(titanic_df['Sex']);
from sklearn.preprocessing import LabelEncoder

sex_encoder = LabelEncoder()



sex_encoder.fit(list(titanic_df['Sex'].values) + list(test_df['Sex'].values))
sex_encoder.classes_
titanic_df['Sex'] = sex_encoder.transform(titanic_df['Sex'].values)

test_df['Sex'] = sex_encoder.transform(test_df['Sex'].values)
sns.countplot(titanic_df['Sex']);
titanic_df.head()
feature_names = ['Pclass', 'SibSp', 'Parch', 'Fare']
seed=45

from sklearn.model_selection import train_test_split

train_X, valid_X, train_y, valid_y = train_test_split(titanic_df[feature_names].as_matrix(), 

                                                      titanic_df['Survived'].as_matrix(),

                                                      test_size=0.2,

                                                      random_state=seed)

                                                      

                                                      

print(train_X.shape)

print(valid_X.shape)                                           

print(train_y.shape)

print(valid_y.shape)
from sklearn.ensemble import RandomForestClassifier



#Hiperparametros

rf_clf = RandomForestClassifier(random_state=seed, n_estimators=200, max_depth=5)



#Treino

rf_clf.fit(train_X, train_y)



print(rf_clf.score(train_X, train_y))

print(rf_clf.score(valid_X, valid_y))
import seaborn as sns

sns.barplot(rf_clf.feature_importances_, feature_names);
feature_names = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Sex']



from sklearn.model_selection import train_test_split

train_X, valid_X, train_y, valid_y = train_test_split(titanic_df[feature_names].as_matrix(), 

                                                      titanic_df['Survived'].as_matrix(),

                                                      test_size=0.2,

                                                      random_state=seed)

                                                      

                                                      

print(train_X.shape)

print(valid_X.shape)                                           

print(train_y.shape)

print(valid_y.shape)



rf_clf = RandomForestClassifier(random_state=seed, n_estimators=200, max_depth=3)

rf_clf.fit(train_X, train_y)

print(rf_clf.score(train_X, train_y))

print(rf_clf.score(valid_X, valid_y))



sns.barplot(rf_clf.feature_importances_, feature_names);
titanic_df.head()['Name']
import re

def extract_title(name):

    x = re.search(', (.+?)\.', name)

    if x:

        return x.group(1)

    else:

        return ''
titanic_df['Name'].apply(extract_title).unique()
titanic_df['Title'] = titanic_df['Name'].apply(extract_title)

test_df['Title'] = test_df['Name'].apply(extract_title)
titanic_df.head()
#from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction import DictVectorizer



feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'Embarked']

dv = DictVectorizer()

dv.fit(titanic_df[feature_names].append(test_df[feature_names]).to_dict(orient='records'))

dv.feature_names_
train_X, valid_X, test_y, valid_y = train_test_split(dv.transform(titanic_df[feature_names].to_dict(orient='records')),

                                                     titanic_df['Survived'],

                                                     test_size=0.2,

                                                     random_state=seed)
train_X.shape

titanic_df['Embarked'] = titanic_df['Embarked'].fillna('Z')
from sklearn.model_selection import train_test_split

rf_clf = RandomForestClassifier(random_state=seed, n_estimators=200, max_depth=3)

rf_clf.fit(train_X, train_y)

print(rf_clf.score(train_X, train_y))

print(rf_clf.score(valid_X, valid_y))



sns.barplot(rf_clf.feature_importances_, dv.feature_names_);
titanic_df[pd.isnull(titanic_df['Embarked'])]
test_df['Fare'] = test_df['Fare'].fillna(0)

#test_df['Embarked'] = titanic_df['Embarked'].fillna('Z')
valid_X = dv.transform(titanic_df[feature_names].to_dict(orient='records'))

valid_y = titanic_df['Survived']



#rf_clf = RandomForestClassifier(random_state=seed, max_depth=3, n_estimators=200)

#rf_clf.fit(train_X, train_y)
test_df['Fare'] = test_df['Fare'].fillna(0)

test_X = dv.transform(test_df[feature_names].to_dict(orient='records'))

print(test_X.shape)
y_pred = rf_clf.predict(test_X)
y_pred
submission_df = pd.DataFrame()
submission_df['PassengerId'] = test_df['PassengerId']

submission_df['Survived'] = y_pred

submission_df
submission_df.to_csv('feature_engineering_rf.csv', index=False)