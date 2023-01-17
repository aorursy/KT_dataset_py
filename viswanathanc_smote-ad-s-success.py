# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



!pip install category_encoders



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization library

import seaborn as sns # visualization library



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/advertsuccess/Train.csv", index_col='id')
data.info()
data.netgain.value_counts().plot(kind='barh')

plt.title('Net gain ( \'True\' or \'False\' )', fontsize=20)

plt.show()
data.head()
plt.figure(figsize=(15,4))

plt.title('Relationship Status',fontsize=20)

sns.countplot(data.realtionship_status)

plt.show()
plt.figure(figsize=(15,4))

plt.title('Industry',fontsize=20)

sns.countplot(data.industry)

plt.show()
plt.figure(figsize=(15,4))

plt.title('Genre',fontsize=20)

sns.countplot(data.genre)

plt.show()
plt.figure(figsize=(15,4))

plt.hist(data['average_runtime(minutes_per_week)'],bins=25)

plt.title('average_runtime(minutes_per_week)',fontsize=20)

plt.show()
plt.figure(figsize=(15,4))

plt.title('Gender',fontsize=20)

sns.countplot(data.targeted_sex)

plt.show()
plt.figure(figsize=(15,4))

plt.title('Air time',fontsize=20)

sns.countplot(data.airtime)

plt.show()
plt.figure(figsize=(5,10))

plt.title('Air Location',fontsize=20)

data.airlocation.value_counts().plot(kind='barh')

plt.show()
plt.figure(figsize=(15,4))

plt.title('Expensive',fontsize=20)

sns.countplot(data.expensive)

plt.show()
plt.figure(figsize=(15,4))

plt.title('Money back Guarantee - Yes or No',fontsize=20)

sns.countplot(data.money_back_guarantee)

plt.show()
#Preprocessing

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from category_encoders import BinaryEncoder

from sklearn.metrics import precision_score

#expensive is an ordinal categorical variable

exp_dict = {'Low':0,'Medium':1,'High':2}

data['expensive'] = data.expensive.map(exp_dict)



#Binary Categorical

Bin_columns = ['targeted_sex','money_back_guarantee']

Bin_Encoder = BinaryEncoder()



#Multi class nominal categorical

cat_columns = ['realtionship_status', 'industry', 'genre', 'airtime', 'airlocation' ]

OHE = OneHotEncoder(sparse=False)



encoding = ColumnTransformer(transformers=[('cat',OHE,cat_columns),

                                               ('bin',Bin_Encoder,Bin_columns)])



clf = Pipeline(steps=[('encoder',encoding),('Std',StandardScaler()),('LR',LogisticRegression())])



y, X = data['netgain'],data.drop('netgain',axis=1)



X_train, X_test, y_train,  y_test = train_test_split(X,y)



clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)



precision_score(y_pred,y_test)
# Applying SMOTE

from imblearn.over_sampling import SMOTE 



#SMOTE have to be applied for training set only



preprocessor = Pipeline(steps=[('encoder',encoding),('Std',StandardScaler())])



X_train = preprocessor.fit_transform(X_train)

X_test = preprocessor.transform(X_test)



print("Shape of train dataset before applying SMOTE:",X_train.shape)



X_train, y_train = SMOTE().fit_resample(X_train,y_train)



print("Shape of train dataset after applying SMOTE:",X_train.shape)
lr2 = LogisticRegression()

lr2.fit(X_train,y_train)

y_pred2 = lr2.predict(X_test)

precision_score(y_pred2,y_test)