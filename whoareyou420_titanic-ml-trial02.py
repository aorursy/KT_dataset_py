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
df_train.loc[df_train.Fare.isnull(), 'Fare_scaled'] = orig_train['Fare'].mean()

df_test.loc[df_test.Fare.isnull(), 'Fare_scaled'] = orig_test['Fare'].mean()



df_train['Fare_scaled'] = orig_train['Fare'].map(lambda i : np.log(i) if i>0 else 0)

df_test['Fare_scaled'] = orig_test['Fare'].map(lambda i : np.log(i) if i > 0 else 0)
import seaborn as sns



plt.style.use('seaborn')

sns.set(font_scale=2.5)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(df_train['Fare_scaled'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

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
df_train['Age_cat'] = 0

df_train.loc[df_train['Age'] < 10, 'Age_cat'] = 0

df_train.loc[(10 <= df_train['Age']) & (df_train['Age'] < 20), 'Age_cat'] = 1

df_train.loc[(20 <= df_train['Age']) & (df_train['Age'] < 30), 'Age_cat'] = 2

df_train.loc[(30 <= df_train['Age']) & (df_train['Age'] < 40), 'Age_cat'] = 3

df_train.loc[(40 <= df_train['Age']) & (df_train['Age'] < 50), 'Age_cat'] = 4

df_train.loc[(50 <= df_train['Age']) & (df_train['Age'] < 60), 'Age_cat'] = 5

df_train.loc[(60 <= df_train['Age']) & (df_train['Age'] < 70), 'Age_cat'] = 6

df_train.loc[70 <= df_train['Age'], 'Age_cat'] = 7
df_test['Age_cat'] = 0

df_test.loc[df_test['Age'] < 10, 'Age_cat'] = 0

df_test.loc[(10 <= df_test['Age']) & (df_test['Age'] < 20), 'Age_cat'] = 1

df_test.loc[(20 <= df_test['Age']) & (df_test['Age'] < 30), 'Age_cat'] = 2

df_test.loc[(30 <= df_test['Age']) & (df_test['Age'] < 40), 'Age_cat'] = 3

df_test.loc[(40 <= df_test['Age']) & (df_test['Age'] < 50), 'Age_cat'] = 4

df_test.loc[(50 <= df_test['Age']) & (df_test['Age'] < 60), 'Age_cat'] = 5

df_test.loc[(60 <= df_test['Age']) & (df_test['Age'] < 70), 'Age_cat'] = 6

df_test.loc[70 <= df_test['Age'], 'Age_cat'] = 7
pd.crosstab(df_train['Age_cat'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Spec': 4, 'Woman': 5})

df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Spec': 4, 'Woman': 5})
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})

df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
df_train['FamilySize'] = orig_train['SibSp']+orig_train['Parch']+1

df_test['FamilySize'] = orig_test['SibSp']+orig_test['Parch']+1
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
df_train.loc[df_train['Embarked'].isnull()] = int(np.random.randn()*10%3)

df_test.loc[df_test['Embarked'].isnull()] = int(np.random.randn()*10%3)
heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Cabin', 'Fare_scaled', 'Embarked', 'Initial', 'Age_cat', 'FamilySize']] 



colormap = plt.cm.RdBu

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,

           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})



del heatmap_data
df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
df_train.head()
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Age'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Age'], axis=1, inplace=True)
df_test.head()
train_X = df_train.drop('Survived', axis=1)

train_Y = df_train['Survived'].as_matrix().reshape(len(df_train['Survived']), 1)



test_X = df_test
df_test.isnull().sum()
from keras.models import Sequential

from keras.layers import Dense, Dropout
model = Sequential()

model.add(Dense(24,kernel_initializer='RandomNormal', activation='relu', input_shape=(15,)))

model.add(Dense(48, kernel_initializer='RandomNormal', activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(48, kernel_initializer='RandomNormal', activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(48, kernel_initializer='RandomNormal', activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(16, kernel_initializer='RandomNormal', activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.summary()
from keras.optimizers import Adam, SGD
opt = Adam(lr=0.005, decay=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=train_X, y=train_Y, batch_size=16, epochs=300, validation_split=0.2)
predict = model.predict_classes(test_X)
predict.shape

predict = predict.reshape(418,)
my_first_submission = pd.DataFrame({"PassengerId": orig_test["PassengerId"], "Survived": predict})

my_first_submission.to_csv("my_first_submission.csv", index=False)