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
import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import pickle

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
df = pd.read_csv("../input/catch-me-if-you-can/train_sessions.csv", index_col='session_id')

df = df.sort_values(by='time1')

df.head()
X = df.iloc[:,:-1]

y = df.target
#функция для вывода информации по данным

def description(df):

    print(f'Dataset Shape:{df.shape}')

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values   

    summary['Uniques'] = df.nunique().values

    return summary

print('Описание данных:')

description(df)
times = ['time%s' % i for i in range(1,11)]

df[times] = df[times].apply(pd.to_datetime)

df.get_ftype_counts()
site = ['site%s' % i for i in range(1,11)]

df[site] = df[site].fillna(0).astype(np.int)
df['weekday'] = df['time1'].dt.dayofweek

df['year'] = df['time1'].dt.year

df['hour'] = df['time1'].dt.hour

df['morning'] = ((df.hour >= 6) & (df.hour < 12)).astype(int)

df['daytime'] = ((df.hour >= 12) & (df.hour < 18)).astype(int)

df['evening'] = ((df.hour >= 18) & (df.hour < 24)).astype(int)

df['night']   = ((df.hour >= 24) & (df.hour < 6)).astype(int)

df.head()
#Переменный для визуализации

session_hour = df['hour'].values

session_week = df['weekday'].values

session_year = df['year'].values
plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)

sns.countplot(x = session_hour)

plt.title('Все пользователи')



plt.subplot(1, 2, 2)

sns.countplot(session_hour[y==1])

plt.title('Иван')



plt.show()
plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)

sns.countplot(x = session_week)

plt.title('Все пользователи')



plt.subplot(1, 2, 2)

sns.countplot(session_week[y==1])

plt.title('Иван')



plt.show()
plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)

sns.countplot(x = session_year)

plt.title('Все пользователи')



plt.subplot(1, 2, 2)

sns.countplot(session_year[y==1])

plt.title('Иван')



plt.show()
new_df = df.drop(['time1','time2','time3','time4','time5','time6','time7','time8','time9','time10'],axis=1)

new_df.head()
X_new = new_df.drop(['target'], axis=1)

y_new = new_df.target
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=42)
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}

logreg=LogisticRegression()

logreg_cv=GridSearchCV(logreg,grid,cv=10)

logreg_cv.fit(X_train,y_train)
print(logreg_cv.best_score_)

print(logreg_cv.best_params_)
pred = logreg_cv.predict_proba(X_test)[:,1]
roc_auc_score(y_test, pred)