# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl # plotting data

import matplotlib.pyplot as plt # plotting data

import seaborn as sns # plotting



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train
df_train['Survived'].value_counts()
#Visualize the survival counts

sns.countplot(df_train['Survived'],label='Count')
sns.countplot(df_train['Sex'],label='Count')
plt.subplots(figsize=(10,10))

sns.countplot('Sex',hue='Survived',data=df_train,palette='RdBu_r')

plt.show()
plt.subplots(figsize=(15,10))

sns.countplot('Pclass',hue='Survived',data=df_train,palette='RdBu_r')

plt.show()
df_train.pivot_table('Survived', index='Sex', columns='Pclass').plot()
plt.subplots(figsize=(15,10))

sns.countplot(df_train['Age'],label='Age')

plt.show()
age = pd.cut(df_train['Age'], [0,18,29,40,65,100])

plt.subplots(figsize=(15,10))

sns.countplot(age,label='Age')

plt.show()
plt.subplots(figsize=(15,10))

sns.countplot(age,hue='Survived',data=df_train,palette='RdBu_r')

plt.show()
df_train['Range'] = age
sns.catplot(x='Range', hue='Survived', col='Sex',

                data=df_train, kind='count',saturation=.5);
sns.catplot(x='Range',y='Survived', hue='Pclass', col='Sex',

                data=df_train, kind='bar', ci=None, saturation=.5, aspect=1.2);
plt.subplots(figsize=(15,10))

sns.countplot('SibSp',hue='Survived',data=df_train,palette='RdBu_r')

plt.xlabel('Siblings or Spouse Aboard')

plt.legend(loc = 'upper right')

plt.show()
sns.catplot(x='SibSp',y='Survived', hue='Pclass', col='Sex',

                data=df_train, kind='bar', ci=None, saturation=.5, aspect=1.2);
plt.subplots(figsize=(15,10))

sns.countplot('Parch',hue='Survived',data=df_train,palette='RdBu_r')

plt.xlabel('Parent or Child Aboard')

plt.legend(loc = 'upper right')

plt.show()
sns.catplot(x='Parch',y='Survived', hue='Pclass', col='Sex',

                data=df_train, kind='bar', ci=None, saturation=.5, aspect=1.2);
fares = pd.cut(df_train['Fare'], [0,50,100,150,200,250,300,350,400,450,1000])

df_train['F_Range'] = fares



plt.subplots(figsize=(15,10))

sns.countplot('F_Range',hue='Survived',data=df_train,palette='RdBu_r')

plt.xlabel('Fare Paid')

plt.legend(loc = 'upper right')

plt.show()
sns.catplot(x='F_Range', hue='Survived', col='Pclass',

                data=df_train, kind='count', saturation=.5, aspect=2, col_wrap=1);
sns.catplot(x='F_Range',y='Survived', hue='Pclass', col='Sex',

                data=df_train, kind='bar', ci=None, saturation=.5, aspect=2, col_wrap=1);
df_corr = pd.read_csv('/kaggle/input/titanic/train.csv') # pulling a clean copy without our added columns



df_corr = pd.get_dummies(df_corr, columns=['Sex','Pclass'],

                        drop_first=True)



# Let's drop some data from the frame that we know isn't helpful

df_corr.drop(['Name','Ticket','Cabin','Embarked','PassengerId'], axis=1, inplace=True)



corr=df_corr.corr()



plt.figure(figsize=(16,12))

sns.heatmap(corr,annot=True,fmt='.2f')

plt.show()
from sklearn.ensemble import RandomForestClassifier
trees = 106

branches = 5
features = ['Pclass', 'Sex', 'Range', 'SibSp', 'Parch','F_Range']
df_test = pd.read_csv("/kaggle/input/titanic/test.csv") # import testing set



# add fare ranges

test_fares = pd.cut(df_test['Fare'], [0,50,100,150,200,250,300,350,400,450,1000])

df_test['F_Range'] = test_fares



# add age ranges

test_age = pd.cut(df_test['Age'], [0,18,29,40,65,100])

df_test['Range'] = test_age
df_test.head()
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()
y = df_train['Survived']



X = pd.get_dummies(df_train[features])

X_imp = my_imputer.fit_transform(X)

X_test = pd.get_dummies(df_test[features])

X_test_imp = my_imputer.fit_transform(X_test)

forest =  RandomForestClassifier(n_estimators=trees,

                                 max_depth=branches,

                                 random_state=1)

forest.fit(X_imp,y)

prediction = forest.predict(X_test_imp)



output = pd.DataFrame({'PassengerId': df_test.PassengerId,

                       'Survived': prediction})

output.to_csv('my_submission.csv', index=False)

print('Done.')
output.head()