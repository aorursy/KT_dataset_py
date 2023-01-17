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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train=pd.read_csv("../input/titanic/train.csv")

test= pd.read_csv('../input/titanic/test.csv')
train.head(5)
test.tail(2)
train.isnull().sum()

test.isnull().sum()
train["Age"]=train["Age"].fillna(train["Age"].mean())

test["Age"]=test["Age"].fillna(test["Age"].mean())

test["Fare"]=test["Fare"].fillna(test["Fare"].mean())
train.head()
train.SibSp.value_counts()
train["SibSp"]= train["SibSp"].replace(8, "6") 
train.SibSp.value_counts()
train.Parch.value_counts()
test.Parch.value_counts()
test["Parch"]= test["Parch"].replace(9, "7") 
test.Parch.value_counts()
train.Cabin = train.Cabin.fillna('Unknown_Cabin')

train['Cabin'] = train['Cabin'].str[0]
train.isnull().sum()
test.Cabin = test.Cabin.fillna('Unknown_Cabin')

test['Cabin'] = test['Cabin'].str[0]
test.isnull().sum()
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

train.head()
train.isnull().sum()
First_NAMES=train.Name.str.split(",").map(lambda x: x[0])

First_NAMES.value_counts()[:50]

# Top 10  Publisher names

plot = First_NAMES.value_counts().nlargest(50).plot(kind='bar', title="Top 10 First names", figsize=(12,6))
train.head()
sns.barplot(x=train['Sex'].value_counts().index,y=train['Sex'].value_counts().values)

plt.title('Genders other rate')

plt.ylabel('Rates')

plt.legend(loc=0)

plt.show()
plt.figure(figsize=(10,7))

sns.barplot(x = "Pclass", y = "Fare", hue = "Sex", data = train)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10,7))

sns.barplot(x = "Pclass", y = "Age", hue = "Sex", data = train)

plt.xticks(rotation=45)

plt.show()
# Data to plot

labels = 'Pclass_1', 'Pclass_2', 'Pclass_3'

sizes = train.groupby('Pclass')['Fare'].mean().values

colors = ['gold', 'yellowgreen', 'lightcoral']

explode = (0.1, 0, 0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Fare for every Pclass')

plt.axis('equal')

plt.show()
# Data to plot

labels = 'Male', 'Female'

sizes = train.groupby('Sex')['Fare'].mean().values

colors = ['gold', 'yellowgreen']

explode = (0.1, 0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Fare for Every Sex')

plt.axis('equal')

plt.show()
sns.barplot(x=train['Survived'].value_counts().index,y=train['Survived'].value_counts().values)

plt.title('Death and Survived')

plt.ylabel('Rates')

plt.legend(loc=0)

plt.show()
sns.countplot(train['Survived'],hue=train['Sex'])

plt.show()
sns.countplot(y=train['Pclass'],palette="Set3",hue=train['Sex'])

plt.legend(loc=4)

plt.show()
train.drop(['Name','Ticket'],axis='columns',inplace=True)

test.drop(['Name','Ticket'],axis='columns',inplace=True)
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

train.Sex = enc.fit_transform(train.Sex)

train.Embarked = enc.fit_transform(train.Embarked)

train.Cabin=enc.fit_transform(train.Cabin)

train.head()

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

test.Sex = enc.fit_transform(test.Sex)

test.Embarked = enc.fit_transform(test.Embarked)

test.Cabin=enc.fit_transform(test.Cabin)

test.head()
train.head(1)
print(train['Fare'].quantile(0.)) 

print(train['Fare'].quantile(0.25)) 

train['Fare'] = np.where(train['Fare']<1, 4.0125, train['Fare'])

train['Fare'].min()
train['Ages'] = pd.cut(train['Age'], bins=[0,25,50,80], labels=["0", "1", "2"])

test['Ages'] = pd.cut(test['Age'], bins=[0,25,50,80], labels=["0", "1", "2"])
train.isnull().sum()
test.isnull().sum()
train.Fare.value_counts()
train.isnull().sum()
train.head()
train.drop(['Age'],axis='columns',inplace=True)

test.drop(['Age'],axis='columns',inplace=True)
train.head(2)
test.head()
col_name="Survived"

first_col = train.pop(col_name)
train.insert(9, col_name, first_col)
train.head(2)
train= train.dropna(how='any',axis=0) 

test["Fare"]=train["Fare"].fillna(train["Fare"].max())
train.isnull().sum()
test.isnull().sum()
#correlation map

f,ax = plt.subplots(figsize=(14, 14))

sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
train.drop(['Parch'],axis=1,inplace=True)
test.drop(['Parch'],axis=1,inplace=True)
train.head()
test.info()
# Everything except target variable

X = train.drop("Survived", axis=1)



# Target variable

y = train['Survived'].values
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
# Split into train & test set`

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=10) 
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

X_test.head()
predictions
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
# Calculate the Accuracy

from sklearn.metrics import accuracy_score

print(accuracy_score(predictions,y_test))
logmodel.fit(train.drop(['Survived'],axis=1),train['Survived'])
test_prediction = logmodel.predict(test)

test_prediction = [ 1 if y>=0.5 else 0 for y in test_prediction]
test_pred = pd.DataFrame(test_prediction, columns= ['Survived'])
new_test = pd.concat([test, test_pred], axis=1, join='inner')
new_test.head()
df= new_test[['PassengerId' ,'Survived']]
df.head()
df.to_csv('predictions.csv' , index=False)