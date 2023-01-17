import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

import random

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()
data.shape
random.seed(42)
# Label numerical and categorical columns

num_col = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary']

cat_col = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 

           'specialisation', 'status']
data.groupby('gender').sl_no.count()
na_values = []

for column in data.columns:

    na_values.append(data[column].isna().sum())

print(na_values)
data[data.salary.isna()].status.unique()
data[~data.salary.isna()].status.unique()
data.salary.fillna(0, inplace=True)
data[cat_col].describe()
data[num_col].describe(percentiles=[0.1, 0.5, 0.9])
# We create a dataframe consisting only of placed workers, for which we will like to analyyze the salary.

placed = data[data.status=='Placed']
placed.salary.describe(percentiles=[0.25, 0.5, 0.75 ,0.95])
# Introduce a new column indicating that a person is in top quartile of the salary.

placed.eval('top_sal = salary>300000', inplace=True)
sns.pairplot(placed[(placed.salary>300000) | (placed.salary<240000)] [['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'top_sal']], hue='top_sal')

plt.show()
# Correlation coefficients with top_sal and salary

placed[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'top_sal', 'salary']].corr()[['top_sal','salary']]
sns.pairplot(placed[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'gender', 'salary']], hue = 'gender')
data.eval('male = gender=="M" ', inplace=True)

data.eval('placed = status=="Placed" ', inplace=True)
data.male.mean()
data.groupby('workex')['male', 'salary', 'placed'].mean().round(2)
data.groupby('degree_t')['male', 'salary', 'placed'].mean().round(2)
data.groupby('specialisation')['male', 'salary', 'placed'].mean().round(2)
data.groupby('hsc_b')['male', 'salary', 'placed'].mean().round(2)
placed = placed[placed.salary<400000]  # remove outlier salaries
placed.groupby(['workex', 'gender'])['salary'].aggregate(['mean','median', 'count']).round(0)
placed.groupby(['specialisation', 'gender'])['salary'].aggregate(['mean','median', 'count']).round(0)
placed.groupby(['degree_t', 'gender'])['salary'].aggregate(['mean','median', 'count']).round(0)
data[data.salary>400000].male.mean()
placed.groupby(['workex', 'degree_t', 'specialisation',  'gender'])['salary'].aggregate(['mean','median', 'count']).round(0)
most_common = data[(data.hsc_s=='Commerce') & (data.degree_t =='Comm&Mgmt')

                   & (data.workex=='No') & (data.specialisation=='Mkt&Fin') & (data.placed==True)]
most_common[['ssc_p', 'hsc_p', 'degree_p',  'etest_p', 'mba_p',

             'salary', 'male']].corr()[['salary', 'male']]
most_common[most_common.male==1][['ssc_p', 'hsc_p', 'degree_p',  'etest_p', 'mba_p',

             'salary']].corr().salary
most_common[most_common.male==0][['ssc_p', 'hsc_p', 'degree_p',  'etest_p', 'mba_p',

             'salary']].corr().salary
sns.pairplot(most_common[['ssc_p', 'hsc_p', 'degree_p','etest_p', 'mba_p', 'salary', 'gender']], hue='gender')

plt.show()
sns.pairplot(data[['ssc_p', 'hsc_p', 'degree_p','etest_p', 'mba_p', 'status']], hue='status')

plt.show()
f, axes = plt.subplots(1, 3, sharey=True, figsize=(17,6))



sns.boxplot(x='status', y='degree_p', data=data[data.degree_t=='Comm&Mgmt'], ax=axes[0])

sns.boxplot(x='status', y='degree_p', data=data[data.degree_t=='Sci&Tech'], ax=axes[1])

sns.boxplot(x='status', y='degree_p', data=data[data.degree_t=='Others'], ax=axes[2])

axes[0].title.set_text('Commerce & Management')

axes[1].title.set_text('Science & Technology')

axes[2].title.set_text('Others')

plt.show()
f, axes = plt.subplots(1, 6, sharey=True, figsize=(17,6))



sns.boxplot(x='status', y='degree_p', data=data[data.gender=='M'], ax=axes[0])

sns.boxplot(x='status', y='degree_p', data=data[data.gender=='F'], ax=axes[1])

sns.boxplot(x='status', y='ssc_p', data=data[data.gender=='M'], ax=axes[2])

sns.boxplot(x='status', y='ssc_p', data=data[data.gender=='F'], ax=axes[3])

sns.boxplot(x='status', y='ssc_p', data=data[data.gender=='M'], ax=axes[4])

sns.boxplot(x='status', y='ssc_p', data=data[data.gender=='F'], ax=axes[5])

axes[0].title.set_text('Men')

axes[1].title.set_text('Women')

axes[2].title.set_text('Men')

axes[3].title.set_text('Women')

axes[4].title.set_text('Men')

axes[5].title.set_text('Women')

plt.show()
from sklearn.model_selection import train_test_split

data_one = pd.get_dummies(data[['gender', 'ssc_p', 'hsc_p', 

                                'hsc_s', 'degree_p', 'degree_t',

                                  'workex', 'etest_p', 'specialisation', 'mba_p']], drop_first=True)
y_one = data.placed
X_train, X_test, y_train, y_test = train_test_split(data_one, y_one, test_size=0.3, random_state=42)
# Accuracy of base estimator, just guessing that everyone is placed.

data.placed.mean()
lr = LogisticRegression(penalty='l2',

    tol=0.001,

    C=50,

    random_state=42,

    solver='lbfgs',

    max_iter=1000,

    class_weight={1:1, 0:2})

lr.fit(X_train,y_train)

lr.score(X_train,y_train)
lr.score(X_test,y_test)
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(lr, X_test, y_test)

plt.show()
coeff = list(zip(data_one.columns, lr.coef_[0].round(2)))
pd.DataFrame(coeff, columns=['variable', 'coefficent']).set_index('variable')
# A borderline candidate with barely passing grades

female_candidate = [60  , 55  , 60  , 60  , 55,  0.  ,  1.  ,  0.  ,  0.  ,

        0.  ,  1.  ,  1.  ]

male_candidate = [60  , 55  , 60  , 60  , 55,  1  ,  1.  ,  0.  ,  0.  ,

        0.  ,  1.  ,  1.  ]
lr.predict_proba([female_candidate])
lr.predict_proba([male_candidate])