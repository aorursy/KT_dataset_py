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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/titanic/train.csv')

train.head(3)
print('\n ### 훈련 데이터셋 정보 ### \n')

print(train.info())
train['Age'].fillna(train['Age'].mean(), inplace = True)
train['Cabin'].fillna('N', inplace=True)

train['Embarked'].fillna('N', inplace=True)

print('데이터 Null 값 개수: ', train.isnull().sum().sum())
print('Sex 값 분포 :\n', train['Sex'].value_counts())

print('\nCabin 값 분포: \n', train['Cabin'].value_counts())

print('\nEmbarked 값 분포: \n', train['Embarked'].value_counts())
train['Cabin'] = train['Cabin'].str[:1]

print(train['Cabin'].head(3))
train.groupby(['Sex','Survived'])['Survived'].count().to_frame()
sns.barplot(x='Sex', y='Survived', data=train)
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train)
def get_category(age):

    cat = ''

    if age <= -1 : cat = 'Unknown'

    elif age <= 5 : cat = 'Baby'

    elif age <= 12 : cat = 'Child'

    elif age <= 18 : cat = 'Teenager'

    elif age <= 25 : cat = 'Student'

    elif age <= 35 : cat = 'Young Adult'

    elif age <= 60: cat = 'Adult'

    else : cat = 'Elderly'

    

    return cat
plt.figure(figsize=(12, 6))
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']
train['Age_cat'] = train['Age'].apply(lambda x: get_category(x))

sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=train, order=group_names)

train.drop('Age_cat', axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder



def encode_features(dataDF) :

    features = ['Cabin', 'Sex', 'Embarked']

    for feature in features:

        le = LabelEncoder()

        le = le.fit(dataDF[feature])

        dataDF[feature] = le.transform(dataDF[feature])

        

    return dataDF



train = encode_features(train)

train.head()
def fillna(df):

    df['Age'].fillna(df['Age'].mean(), inplace=True)

    df['Cabin'].fillna('N', inplace=True)

    df['Embarked'].fillna('N', inplace=True)

    df['Fare'].fillna(0, inplace=True)

    return df
def drop_features(df):

    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

    return df
def format_features(df):

    df['Cabin'] = df['Cabin'].str[:1]

    features = ['Cabin', 'Sex', 'Embarked']

    for feature in features:

        le = LabelEncoder()

        le.fit(df[feature])

        df[feature] = le.transform(df[feature])

    return df
def transform_features(df):

    df = fillna(df)

    df = drop_features(df)

    df = format_features(df)

    return df
titanic_df = pd.read_csv('../input/titanic/train.csv')

y_titanic_df = titanic_df['Survived']

X_titanic_df = titanic_df.drop(['Survived'], axis=1)



X_titanic_df = transform_features(X_titanic_df)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size = 0.2, random_state = 11)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
dt_clf = DecisionTreeClassifier(random_state=11)

rf_clf = RandomForestClassifier(random_state=11)

lr_clf = LogisticRegression()
dt_clf.fit(X_train, y_train)

dt_pred = dt_clf.predict(X_test)

print('DecisionTreeClassifier 정확도 : {0:.4f}'.format(accuracy_score(y_test, dt_pred)))
rf_clf.fit(X_train, y_train)

rf_pred = rf_clf.predict(X_test)

print('RandomForestClassifier 정확도 : {0:.4f}'.format(accuracy_score(y_test,rf_pred)))
lr_clf.fit(X_train, y_train)

lr_pred = lr_clf.predict(X_test)

print('LogisticRegression 정확도 : {0:.4f}'.format(accuracy_score(y_test, lr_pred)))