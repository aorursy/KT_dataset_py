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
%matplotlib inline

import pandas as pd

import numpy as np

from pandas_profiling import ProfileReport

from sklearn.preprocessing import StandardScaler
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df_pred = pd.read_csv('/kaggle/input/titanic/test.csv')
# Null 'Age'

data = [df, df_pred]

for dataset in data :

    avg_mean = dataset["Age"].mean(axis = 0)

    avg_std = dataset["Age"].std()

    sum_null = dataset["Age"].isnull().sum()

    age_slice = dataset["Age"].copy()

    age_slice[ np.isnan(age_slice) ] = np.random.normal(avg_mean, avg_std)

    dataset["Age"] = age_slice.astype(int)
df.isnull().sum()

df
for dataset in data:

    dataset['Title'] = dataset['Name'].apply(lambda x: x.split(',')[1][1:]).apply(lambda x: x.split('.')[0])



for dataset in data:

    dataset.drop(columns=['Cabin', 'Ticket', 'Name'], inplace = True)
df['Age'] = df['Age'].astype(int)

df_pred['Age'] = df_pred['Age'].astype(int)
df_pred.isnull().sum()
df['Embarked'].fillna('S', inplace = True)

df_pred['Fare'].fillna(df_pred['Fare'].median(), inplace = True)
df.isnull().sum()
categorical_cols = ['Pclass','Embarked', 'Sex']

df = pd.get_dummies(df, columns = categorical_cols, drop_first = True)

df_pred = pd.get_dummies(df_pred, columns = categorical_cols, drop_first = True)



train_cols = ['Age', 'SibSp', 'Parch','Fare','Pclass_2', 'Pclass_3',

              'Embarked_Q', 'Embarked_S', 'Sex_male']

target_col = 'Survived'



df.sample(frac=1)

df_pred.sample(frac=1)



from sklearn.metrics import classification_report

from sklearn.svm import SVC,LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.pipeline import Pipeline



from sklearn.model_selection import cross_val_score





df['Fare'] = df['Fare'].astype(int)

df_pred['Fare'] = df_pred['Fare'].astype(int)



from sklearn.preprocessing import MinMaxScaler

continuous_features = ['Fare','Age']

for col in continuous_features:

    transf = df[col].values.reshape(-1,1)

    scaler = MinMaxScaler().fit(transf)

    df[col] = scaler.transform(transf)

for col in continuous_features:

    transf = df_pred[col].values.reshape(-1,1)

    scaler = MinMaxScaler().fit(transf)

    df_pred[col] = scaler.transform(transf)
all_col = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass_2',

           'Pclass_3', 'Embarked_Q', 'Embarked_S', 'Sex_male']
pipe = Pipeline(steps=[

                       ('clf', SVC()),]

               )

cross_val_score(pipe, df[all_col], df[target_col], cv=5).mean()
clf = SVC()

clf.fit(df[all_col], df[target_col])

pred = clf.predict(df_pred[all_col])
ans = pd.DataFrame(df_pred['PassengerId'])

ans['Survived'] = pred
ans.to_csv('submission.csv', index=False)