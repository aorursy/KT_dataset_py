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
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

train=pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")

gender=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
print("keys of train:\n{}".format(train.keys()))

print("keys of test:\n{}".format(test.keys()))

print("keys of gender:\n{}".format(gender.keys()))
print(train['Survived'])

print(train['PassengerId'])

print(test['PassengerId'])
train_data=train.drop(columns=['Survived'])

train_data.head()
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train)
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train,

              palette={"male": "yellow", "female": "brown"},

              markers=["^", "o"], linestyles=["-", "--"]);
def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df



def format_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df    

    

def drop_features(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)



def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_name(df)

    df = drop_features(df)

    return df



train = transform_features(train)

test = transform_features(test)

train.head()
from sklearn import preprocessing

def encode_features(train,test):

    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']

    df_combined = pd.concat([train[features],test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        train[feature] = le.transform(train[feature])

        test[feature] = le.transform(test[feature])

    return train,test

    

train,test = encode_features(train,test)

train.head()
X_all = train.drop(['Survived', 'PassengerId'], axis=1)

y_all = train['Survived']

num_test = 0.20

X_train,X_test,y_train,y_test=train_test_split(X_all,y_all,test_size=num_test, random_state=23)

forest=RandomForestClassifier(n_estimators=5,random_state=2)

forest.fit(X_train,y_train)
print("Accuracy on training set:{:.3f}".format(forest.score(X_train,y_train)))

print("Accuracy on test set:{:.3f}".format(forest.score(X_test,y_test)))
ids = test['PassengerId']

predictions = forest.predict(test.drop('PassengerId', axis=1))

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

display(output)