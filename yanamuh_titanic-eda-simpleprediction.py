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
import numpy as np

import pandas as pd

import missingno as msno

import matplotlib.pyplot as plt

import seaborn as sns



df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
df_test.head()
df_train.head()
df_train.info()
# check missing values

msno.matrix(df_train)
df_train.isna().sum()
# for Embarked column, there are 2 missing values, 

# fill it with modus data in the column



# cek modus

df_train.Embarked.value_counts()
# modus data in Embarked columns is S

df_train.Embarked = df_train.Embarked.fillna('S')
# for Cabin column, there are too many missing values

# just drop the column

df_train.drop('Cabin', axis=1, inplace=True)
# fill Age column with median

df_train.Age.median()
df_train.Age.fillna(28, inplace=True)
# Analysis

df_ = df_train[['Survived','Pclass','Sex','Age','Fare','Embarked']]

df_.describe()
# Fare quantile

df_['fare_quantile'] = pd.qcut(df_['Fare'].copy(), q=5)

passanger_fare_quantile = df_['fare_quantile'].value_counts().sort_index()

passanger_fare_quantile_survived = df_.groupby('fare_quantile').sum()['Survived']

fare_plot = pd.DataFrame({'total_passangers':passanger_fare_quantile,

              'survived_passangers':passanger_fare_quantile_survived

             })

fare_plot.index.rename('Fare Quantile (Pounds)',inplace=True)

print('Distribusi penumpang berdasarkan harga tiket:\n{}\n'.format(fare_plot))

plt.style.use('seaborn-dark')

ax = fare_plot.plot(kind='bar', rot=45, title='Passanger distribution by ticket fare')

ax.set_xlabel('Fare Quantile')

ax.set_ylabel('Number of Passangers')

plt.legend(bbox_to_anchor=(1,1))

plt.show()
# Survival rate vs Age

print(df_.groupby('Survived').agg({'Age':'mean'}))

sns.distplot(df_.Age, label='Age distribution').set_title('Passanger distribution by age')

plt.legend()

plt.show()
# distribusi usia

df_['age_quantile'] = pd.qcut(df_['Age'], q=4)

age_quantile = df_['age_quantile'].value_counts().sort_index()

age_quantile_survived = df_.groupby('age_quantile').sum()['Survived']



age_plot = pd.DataFrame({'total_passangers':age_quantile,

              'survived_passangers':age_quantile_survived

             })

age_plot.index.rename('Age Quantile (year)',inplace=True)

ax = age_plot.plot(kind='bar', rot=0, title='Passanger distribution by age')

ax.set_xlabel('Age Quantile (year)')

ax.set_ylabel('Number of Passangers')

plt.legend(bbox_to_anchor=(1,1))

plt.show()

print('Distribusi penumpang berdasarkan usia:\n{}\n'.format(age_plot))

# Survival rate vs Sex

passanger_by_sex = df_.Sex.value_counts().sort_index()

passanger_by_sex_survived = df_.groupby('Sex').sum()['Survived']

sex_plot = pd.DataFrame({'total_passangers':passanger_by_sex,

              'survived_passangers':passanger_by_sex_survived

             })

sex_plot.index.rename('Sex',inplace=True)



# plot

ax_sex = sex_plot.plot(kind='bar', rot=0, title='Passanger distribution by sex')

ax_sex.set_xlabel('Sex')

ax_sex.set_ylabel('Number of Passangers')

plt.legend(bbox_to_anchor=(1,1))

plt.show()



print('Distribusi penumpang berdasarkan jenis kelamin:\n{}\n'.format(sex_plot))
# Survival rate vs Pclass 

passanger_by_pclass = df_.Pclass.value_counts().sort_index()

passanger_by_pclass_survived = df_.groupby('Pclass').sum()['Survived']

pclass_plot = pd.DataFrame({'total_passangers':passanger_by_pclass,

              'survived_passangers':passanger_by_pclass_survived

             })

pclass_plot.index.rename('Pclass', inplace=True)



pclass_plot = pclass_plot.plot(kind='bar', rot=0, title='Passanger distribution by ticket fare')

pclass_plot.set_xlabel('Pclass')

pclass_plot.set_ylabel('Number of Passangers')

plt.legend(bbox_to_anchor=(1,1))

plt.show()

print('Distribusi penumpang berdasarkan usia:\n{}\n'.format(pclass_plot))
df_.groupby('Embarked').agg({'Survived':'sum'})
df_train_ = df_train[['Survived','Pclass','Sex','Age','Fare']]

df_train_.head()
# encode label

df_train_ = pd.get_dummies(df_train_, columns=['Sex'])

df_train_.head()
from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler()

df_train_.loc[:,['Pclass','Age','Fare']] = mmscaler.fit_transform(df_train.loc[:,['Pclass','Age','Fare']])

df_train_.head()
# model

# split data features and target

x_train = df_train_.drop('Survived', axis=1)

y_train = df_train_['Survived']
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import roc_auc_score
rf = RandomForestClassifier()

param_grid = {

           'n_estimators' : np.arange(4,16,2),

           'criterion' : ['gini','entropy'],

           'max_depth' : np.arange(1,10),

           'min_samples_leaf':np.arange(1,30),

           'random_state':np.arange(2,30)

}

rcv = RandomizedSearchCV(rf, param_grid, scoring='roc_auc', cv=10, return_train_score=True)

rcv.fit(x_train,y_train)
print('Best parameters :\n{}\n'.format(rcv.best_params_))

print('Best score :\n{}\n'.format(rcv.best_score_))

print('Mean train scores :\n{}\n'.format(rcv.cv_results_['mean_train_score']))

print('Mean test scores :\n{}\n'.format(rcv.cv_results_['mean_test_score']))
df_test = df_test[['Pclass','Sex','Age','Fare']]

df_test.head()
df_test.info()
df_test.isna().sum()
# handle missing values

df_test.Fare = df_test.Fare.fillna(df_test.Fare.mean())

df_test.Age = df_test.Age.fillna(df_test.Age.median())
# encode label

df_test = pd.get_dummies(df_test, columns=['Sex'])
# scale the data

df_test.loc[:,['Pclass','Age','Fare']] = mmscaler.fit_transform(df_test.loc[:,['Pclass','Age','Fare']])

df_test.head()
# test model

result = rcv.predict(df_test)
pass_id = pd.read_csv('../input/titanic/test.csv')['PassengerId']

result_csv = pd.DataFrame({'PassengerId':pass_id,

                           'Survived':result

                          })

# save

result_csv.to_csv('titanic_prediction_fin.csv', index = False)