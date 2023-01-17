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
train_df = pd.read_csv('/kaggle/input/titanic/train.csv').set_index('PassengerId')

test_df = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')
train_df.head()
train_df.isna().mean().plot(kind='barh')
fill_mean = train_df[['Age','Fare']].mean()



train_df[['Age','Fare']] = train_df[['Age','Fare']].fillna(fill_mean)

test_df[['Age','Fare']] = test_df[['Age','Fare']].fillna(fill_mean)
test_df.isna().mean()
train_df['Sex'].unique()
train_df.reset_index().groupby(['Sex','Survived']).count()['PassengerId']
train_df.reset_index().groupby(['Sex']).mean()['Survived']
train_df['Age'].plot(kind='hist')
_ ,outbins = pd.cut(train_df['Age'], 10, retbins=True)

outbins[0] = 0 #Fix the edge case

train_df['AgeGroup'] = pd.cut(train_df['Age'],outbins)

test_df['AgeGroup'] = pd.cut(test_df['Age'],outbins)
test_df.loc[test_df['AgeGroup'].isna()]
pd.crosstab(train_df['AgeGroup'], train_df['Survived'])
train_df.groupby('AgeGroup')['Survived'].mean().plot(kind='bar')
train_df[['Pclass','Survived']].groupby('Pclass').count()
train_df[['Pclass','Survived']].groupby('Pclass').sum()
train_df[['Pclass','Survived']].groupby('Pclass').sum() / train_df[['Pclass','Survived']].groupby('Pclass').count()
pd.crosstab([train_df['Pclass'], train_df['Sex'],train_df['Survived']],train_df['AgeGroup']) # train_df.groupby(['Pclass','Sex','Survived']).count()
train_df.pivot_table(index=['Pclass','Sex'], columns='AgeGroup', values='Survived', aggfunc='mean')
col_interest = ['Sex', 'AgeGroup', 'Pclass']

target = 'Survived'

x = train_df[col_interest]

y = train_df[target]



final_test_df = test_df[col_interest]
# We import first the modules we would need in this section

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.10,random_state=42)
#In OOP language, We need to instantiate an object of OneHotEncoder first

#If you are not familiar with OOP, think of it as a way to create a type of OneHotEncoder with certain parameters.  

encoder = OneHotEncoder()



#Understand which of these are categories and how it would be transformed into multiple columns

encoder.fit(x_train)



x_train_enc = encoder.transform(x_train)

x_val_enc = encoder.transform(x_val)

x_test_enc = encoder.transform(final_test_df)
# Similar idea with OneHotEncoder, we want to create a Logistic Regression of our own flavor

estimator = LogisticRegression(C=1.0,class_weight='balanced', solver='lbfgs')



# The meaning comes from the idea of "best fit" line. So, we're actually training the model to get the best parameters that fits best to our data points

estimator.fit(x_train_enc,y_train)
estimator.score(x_val_enc,y_val)
test_df['Survived'] =estimator.predict(x_test_enc)

test_df.head()
test_df.reset_index()[['PassengerId','Survived']].to_csv('../working/submit.csv', index=False)