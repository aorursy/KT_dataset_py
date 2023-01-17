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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
#importing our dataset

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')



#creating a copy of data set for visualization

data=train_data

sub
train_data.head()
test_data.head()
data
# USING MATPLOTLIB PACKAGE



plt.scatter(x=data['Age'],y=data['Fare'],color='red')

plt.xlabel('Age of Passengers')

plt.ylabel('fare of ticket')

plt.plot(data['Survived'],data['Pclass'],color='cyan')

plt.xlabel('Survived')

plt.ylabel('Pclass')


#using seaborn functions as matplot can be used only on numerical data 



sns.catplot(x='Pclass',y='Age',hue='Survived',kind='violin',split=True,data=train_data)
sns.jointplot(x='Pclass',y='Fare',data=train_data)
#CLASSIFYING DATA ACCORDINGLY

sns.relplot(x='Age',y='Fare',hue='Survived',style='Parch',size='Pclass',data=train_data)

plt.figure(figsize=(30,15))

#TRY TO UNDERSTAND THE FIGURE CLOSELY FOR BETTER UNDERSTANDING
#TO FIND THR TREND

sns.lmplot(x='Pclass',y='Age',hue='Survived',data=train_data)

train_data.dropna(axis=0, subset=['Survived'], inplace=True)

y= train_data.Survived

train_data.drop(['Survived'], axis=1, inplace=True)
train_data
train_data.isnull().nunique()
df = pd.concat([train_data,test_data])
df
df.isnull().sum()
from sklearn.impute import SimpleImputer 

#now we will be imputing our data

imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df))

df_imputed.columns = df.columns
df = pd.get_dummies(df_imputed)
df
train = df.iloc[:891,]

test =df.iloc[891:,]
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid =train_test_split(train,y,train_size=.80,random_state=1)
from sklearn.linear_model import LinearRegression

lr =LinearRegression()

lr.fit(X_train,y_train)

accuracy_lr = lr.score(X_valid,y_valid)
accuracy_lr
from sklearn.ensemble import RandomForestClassifier

model =  RandomForestClassifier(n_estimators=100, random_state=1)

model.fit(X_train, y_train)

acc = model.score(X_valid,y_valid)
acc
from sklearn.svm import SVC

models=SVC()

models.fit(X_train,y_train)

acu= models.score(X_valid,y_valid)

acu
pred = model.predict(test)
pred
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':sub.Survived})

output.to_csv('submission.csv', index=False)