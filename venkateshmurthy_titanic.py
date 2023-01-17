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

import seaborn as sns

from matplotlib import pyplot as plt
#reading in the train data

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
y_train = train.iloc[:, 1].values
#Viewing the data

train.head()
# Countplot 

sns.catplot(x ="Sex", hue ="Survived", kind ="count", data = train) 
# Countplot 

sns.catplot(x ='Embarked', hue ='Survived', kind ='count', col ='Pclass', data = train)
sns.distplot(train['Age'].dropna(), bins=15, kde=False)
#Dropping the unnecessary columns

train = train.drop(columns=['SibSp','Parch','PassengerId','Name','Ticket','Fare','Cabin'],inplace= False)
train.head()
x_train = train.drop('Survived',axis=1,inplace=False)
#Checking for missing values

train.isnull().sum()
#Treating missing values

#For 'Age' feature



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(x_train[['Age']])

x_train[['Age']]= imputer.transform(x_train[['Age']])

#For 'Embarked' feature



imputers = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputers.fit(x_train[['Embarked']])

x_train[['Embarked']]= imputers.transform(x_train[['Embarked']])
#Encoding of Categorical features



from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()



#Sex feature  

x_train['Sex']= label_encoder.fit_transform(x_train['Sex']) 



#Embarked feature

x_train['Embarked']= label_encoder.fit_transform(x_train['Embarked'])
x_train.head()
#Feature Scaling



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
#Following the same steps as per train data

#Treating missing values
test.isnull().sum()
#For 'Age' feature



imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(test[['Age']])

test[['Age']]= imputer.transform(test[['Age']])





#For 'Embarked' feature



imputers = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputers.fit(test[['Embarked']])

test[['Embarked']]= imputers.transform(test[['Embarked']])
#Dropping unnecessary columns

test = test.drop(columns= ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'],inplace=False)

test.head()
#Encoding Categorical data



#Sex feature  

test['Sex']= label_encoder.fit_transform(test['Sex']) 



#Embarked feature

test['Embarked']= label_encoder.fit_transform(test['Embarked'])
test.head()
#Feature Scaling on test data

test = sc.fit_transform(test)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(test)

from sklearn.model_selection import cross_val_score

acc_Tree = cross_val_score(classifier, x_train, y_train, cv=10, scoring='accuracy').mean()

acc_Tree
from sklearn.svm import SVC

classifier = SVC()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(test)
y_pred = classifier.predict(test)

from sklearn.model_selection import cross_val_score

acc_Tree = cross_val_score(classifier, x_train, y_train, cv=10, scoring='accuracy').mean()

acc_Tree
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(test)

from sklearn.model_selection import cross_val_score

acc_Tree = cross_val_score(classifier, x_train, y_train, cv=10, scoring='accuracy').mean()

acc_Tree
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(x_train, y_train)
y_pred = classifier.predict(test)

from sklearn.model_selection import cross_val_score

acc_Tree = cross_val_score(classifier, x_train, y_train, cv=10, scoring='accuracy').mean()

acc_Tree
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(test)

from sklearn.model_selection import cross_val_score

acc_Tree = cross_val_score(classifier, x_train, y_train, cv=10, scoring='accuracy').mean()

acc_Tree
accuracy = {'Model' : ['Logistic Regression', 'K- Nearest Neighbor', 'SVC', 'Decision Tree', 'Random Forest'],

                  'Accuracy' : [0.7890, 0.8047, 0.8226, 0.7935, 0.8037]

                 }

all_cross_val_scores = pd.DataFrame(accuracy, columns = ['Model', 'Accuracy'])

all_cross_val_scores.head()
test_df = pd.read_csv('../input/titanic/test.csv')

submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': y_pred

})

submission.to_csv('titanic_prediction.csv', index=False)
submission