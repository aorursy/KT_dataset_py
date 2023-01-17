# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra|
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import missingno as msno

import copy

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
orig_train = pd.read_csv('../input/train.csv')
orig_test = pd.read_csv('../input/test.csv')
df_train = copy.deepcopy(orig_train)
df_test = copy.deepcopy(orig_test)
df_train.loc[df_train.Fare.isnull(), 'Fare'] = orig_train['Fare'].mean()
df_test.loc[df_test.Fare.isnull(), 'Fare'] = orig_test['Fare'].mean()
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale
df_train['Fare_scaled'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_test['Fare_scaled'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
import seaborn as sns
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Fare_scaled'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare_scaled'].skew()), ax=ax)
g = g.legend(loc='best')
df_train['Initial'] = orig_train.Name.str.extract('([A-Za-z]+)\.')
df_test['Initial'] = orig_test.Name.str.extract('([A-Za-z]+)\.')
df_train['Initial'].replace(['Lady', 'Mlle','Mme','Ms','Dr','Major','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss', 'Woman','Woman','Miss','Master','Mr','Woman','Master','Spec','Spec','Spec','Master','Master', 'Woman'],inplace=True)

df_test['Initial'].replace(['Lady', 'Mlle','Mme','Ms','Dr','Major','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss', 'Woman','Woman','Miss','Master','Mr','Woman','Master','Spec','Spec','Spec','Master','Master', 'Woman'],inplace=True)
pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r')
Master_mean = df_train.loc[df_train.Initial == 'Master', 'Age'].mean()
Miss_mean = df_train.loc[df_train.Initial == 'Miss', 'Age'].mean()
Mr_mean = df_train.loc[df_train.Initial == 'Mr', 'Age'].mean()
Mrs_mean = df_train.loc[df_train.Initial == 'Mrs', 'Age'].mean()
Spec = df_train.loc[df_train.Initial == 'Spec', 'Age'].mean()
Woman_mean = df_train.loc[df_train.Initial == 'Woman', 'Age'].mean()


for i in df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Master'), 'Age'].index:
    df_train.Age[i] =  round(Master_mean + np.random.rand()*10)
    
for i in df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Miss'), 'Age'].index:
    df_train.Age[i] =  round(Miss_mean + np.random.rand()*10)
    
for i in df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mr'), 'Age'].index:
    df_train.Age[i] =  round(Mr_mean + np.random.rand()*10)
    
for i in df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mrs'), 'Age'].index:
    df_train.Age[i] =  round(Mrs_mean + np.random.rand()*10)

for i in df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Spec'), 'Age'].index:
    df_train.Age[i] =  round(Spec + np.random.rand()*10)
    
for i in df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Woman'), 'Age'].index:
    df_train.Age[i] =  round(Woman_mean + np.random.rand()*10)
Master_mean = df_test.loc[df_test.Initial == 'Master', 'Age'].mean()
Miss_mean = df_test.loc[df_test.Initial == 'Miss', 'Age'].mean()
Mr_mean = df_test.loc[df_test.Initial == 'Mr', 'Age'].mean()
Mrs_mean = df_test.loc[df_test.Initial == 'Mrs', 'Age'].mean()
Spec = df_test.loc[df_test.Initial == 'Spec', 'Age'].mean()
Woman_mean = df_test.loc[df_test.Initial == 'Woman', 'Age'].mean()


for i in df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Master'), 'Age'].index:
    df_test.Age[i] =  round(Master_mean + np.random.rand()*10)
    
for i in df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Miss'), 'Age'].index:
    df_test.Age[i] =  round(Miss_mean + np.random.rand()*10)
    
for i in df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mr'), 'Age'].index:
    df_test.Age[i] =  round(Mr_mean + np.random.rand()*10)
    
for i in df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mrs'), 'Age'].index:
    df_test.Age[i] =  round(Mrs_mean + np.random.rand()*10)

for i in df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Spec'), 'Age'].index:
    df_test.Age[i] =  round(Spec + np.random.rand()*10)
    
for i in df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Woman'), 'Age'].index:
    df_test.Age[i] =  round(Woman_mean + np.random.rand()*10)
pd.crosstab(df_train['Age'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
df_train['Age_scaled'] = minmax_scale((df_train['Age']-df_train['Age'].mean())/(df_train['Age'].std()))
df_test['Age_scaled'] = minmax_scale((df_test['Age']-df_train['Age'].mean())/(df_train['Age'].std()))
pd.crosstab(df_train['Age_scaled'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Spec': 4, 'Woman': 5})
df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Spec': 4, 'Woman': 5})
df_train['Initial'] = minmax_scale((df_train['Initial']-df_train['Initial'].mean())/(df_train['Initial'].std()))
df_test['Initial'] = minmax_scale((df_test['Initial']-df_train['Initial'].mean())/(df_train['Initial'].std()))
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_train.loc[df_train['Embarked'].isnull()] = int(np.random.randn()*10%3)
df_test.loc[df_test['Embarked'].isnull()] = int(np.random.randn()*10%3)
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')
df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
df_train['FamilySize'] = orig_train['SibSp']+orig_train['Parch']+1
df_test['FamilySize'] = orig_test['SibSp']+orig_test['Parch']+1
df_train['FamilySize'] = minmax_scale((df_train['FamilySize']-df_train['FamilySize'].mean())/(df_train['FamilySize'].std()))
df_test['FamilySize'] = minmax_scale((df_test['FamilySize']-df_train['FamilySize'].mean())/(df_train['FamilySize'].std()))
df_train['Cabin'] = orig_train['Cabin']
df_test['Cabin'] = orig_test['Cabin']
df_train['Cabin'] = orig_train.Cabin.str.extract('([A-Za-z]+)\w')
df_test['Cabin'] = orig_test.Cabin.str.extract('([A-Za-z]+)\w')
for i in df_train.loc[(df_train.Cabin.isnull())&(df_train.Pclass == 1)].index:
    df_train.Cabin[i] = int((np.random.randn()*10)%5)
    
for i in df_train.loc[(df_train.Cabin.isnull())&(df_train.Pclass == 2)].index:
    df_train.Cabin[i] = int((np.random.randn()*10)%4)+3
    
for i in df_train.loc[(df_train.Cabin.isnull())&(df_train.Pclass == 3)].index:
    df_train.Cabin[i] = int((np.random.randn()*10)%4)+3
for i in df_test.loc[(df_test.Cabin.isnull())&(df_test.Pclass == 1)].index:
    df_test.Cabin[i] = int((np.random.randn()*10)%5)
    
for i in df_test.loc[(df_test.Cabin.isnull())&(df_test.Pclass == 2)].index:
    df_test.Cabin[i] = int((np.random.randn()*10)%4)+3
    
for i in df_test.loc[(df_test.Cabin.isnull())&(df_test.Pclass == 3)].index:
    df_test.Cabin[i] = int((np.random.randn()*10)%4)+3
df_train['Cabin'].replace(['A', 'B','C','D','E','F','G'],
                        [0, 1, 2, 3, 4, 5, 6],inplace=True)

df_test['Cabin'].replace(['A', 'B','C','D','E','F','G'],
                        [0, 1, 2, 3, 4, 5, 6],inplace=True)
df_train = pd.get_dummies(df_train, columns=['Cabin'], prefix='Cabin')
df_test = pd.get_dummies(df_test, columns=['Cabin'], prefix='Cabin')
df_train = pd.get_dummies(df_train, columns=['Pclass'], prefix='Pclass')
df_test = pd.get_dummies(df_test, columns=['Pclass'], prefix='Pclass')
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Initial'], color='b', label='Skewness : {:.2f}'.format(df_train['Initial'].skew()), ax=ax)
g = g.legend(loc='best')
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Age'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Age'], axis=1, inplace=True)
df_test.shape
train_X = df_train.drop('Survived', axis=1)
train_Y = df_train['Survived'].as_matrix().reshape(len(df_train['Survived']), 1)

test_X = df_test
test_X.shape
train_X.head()
df_test.isnull().sum()
from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(18,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=train_X, y=train_Y, batch_size=32, epochs=300, validation_split=0.3)
#validation_data=(val_x, val_y),
predict = model.predict_classes(test_X)
predict.shape
predict = predict.reshape(418,)
my_first_submission = pd.DataFrame({"PassengerId": orig_test["PassengerId"], "Survived": predict})
my_first_submission.to_csv("my_first_submission.csv", index=False)


