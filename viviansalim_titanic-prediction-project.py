#importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#importing the files

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#load training data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
#load testing data

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
survived = train_data [train_data['Survived']==1]
no_survived = train_data [train_data['Survived']==0]
print('Total =', len(train_data))
print('Number of pessengers survived =', len(survived))
print('Number of pessengers not survived =', len(no_survived))
print ('% Survived =', 1. * len(survived)/len(train_data)*100)
train_data.describe()
# checking for missing values

print("Missing values  in the training data")
display(train_data.isnull().sum())
print("Missing values in the test data")
display(test_data.isnull().sum())


#replace the missing data in age with the mean

#test_data['Age']=test_data['Age'].fillna(test_data['Age'].mean())
#train_data['Age']=train_data['Age'].fillna(train_data['Age'].mean())

def Fill_Age(data):
    age = data[0]
    sex = data[1]
    
    if pd.isnull(age):
        if sex is 'male':
            return 29
        else:
            return 25
    else:
        return age
train_data['Age'] = train_data[['Age','Sex']].apply(Fill_Age, axis = 1)
test_data['Age'] = train_data[['Age','Sex']].apply(Fill_Age, axis = 1)
train_data['Age'].hist(bins=20)
#dropping the cabin data from the dataframe

test_data.drop('Cabin', axis=1, inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)

train_data.head()
#embarked value count

train_data['Embarked'].value_counts(normalize=True)
# replacing the missing embarked values with the most frequent value

train_data['Embarked']=train_data['Embarked'].fillna('S')
#fare value replacement with the mean value
test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mean())
# checking for missing values again

print("Missing values  in the training data")
display(train_data.isnull().sum())
print("Missing values in the test data")
display(test_data.isnull().sum())
#replace sex with 0 and 1
male = pd.get_dummies(train_data['Sex'], drop_first = True)
train_data.drop(['Sex'], axis=1, inplace=True)
train_data = pd.concat([train_data, male], axis =1)
train_data.drop(['Name', 'Ticket', 'Embarked'], axis =1, inplace =True)
x = train_data.drop('Survived', axis=1).values
y = train_data['Survived'].values
plt.figure (figsize = [6,12])
plt.subplot(211)    

sns.countplot(x= 'Pclass', data = train_data)

plt.subplot(212)

sns.countplot(x= 'Pclass', hue = 'Survived', data = train_data)

plt.figure (figsize = [6,12])
plt.subplot(211)    

sns.countplot(x= 'SibSp', data = train_data)

plt.subplot(212)

sns.countplot(x= 'SibSp', hue = 'Survived', data = train_data)
plt.figure (figsize = [6,12])
plt.subplot(211)    

sns.countplot(x= 'Parch', data = train_data)

plt.subplot(212)

sns.countplot(x= 'Parch', hue = 'Survived', data = train_data)
#plt.figure (figsize = [6,12])
#plt.subplot(211)    

#sns.countplot(x= 'Embarked', data = train_data)

#plt.subplot(212)

#sns.countplot(x= 'Embarked', hue = 'Survived', data = train_data)
plt.figure (figsize = [6,12])
plt.subplot(211)    

sns.countplot(x= 'male', data = train_data)

plt.subplot(212)

sns.countplot(x= 'male', hue = 'Survived', data = train_data)
plt.figure (figsize = [40,30])

sns.countplot(x= 'Age', hue = 'Survived', data = train_data)
train_data['Age'].hist(bins = 40)
plt.figure (figsize = [40,20])

sns.countplot(x= 'Fare', hue = 'Survived', data = train_data)
train_data['Fare'].hist(bins= 40)
#pd.crosstab(train_data['Sex'], train_data['Survived'])
#pd.crosstab(train_data['Pclass'], train_data['Survived'])
#pd.crosstab(train_data['SibSp'], train_data['Survived'])
#creating a heatmap with correlation (credit: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec)

corr = train_data.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
train_data.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.2, random_state = 10)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression (random_state =0)
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)
y_predict
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot =True, fmt ='d')
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
