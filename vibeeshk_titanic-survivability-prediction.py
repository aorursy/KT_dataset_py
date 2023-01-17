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
import seaborn as sns

import numpy as np # linear algebra

import pandas as pd 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

#from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import math

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
filepath= '/kaggle/input/titanic/train.csv'

traindata= pd.read_csv(filepath)

traindata.head()
traindata = traindata[['PassengerId','Age','Pclass','Name','Sex','SibSp','Parch','Ticket','Cabin','Embarked','Survived']]

traindata.head()

#Here we see that there are a large number of null values and it only makes sense to remove them

sns.heatmap(traindata.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#We need to tend to the missing values in the age column. For this, instead of dropping the entire column, we fill the null values by the mean of all the ages

traindata['Age'].fillna((traindata['Age'].mean()), inplace=True)  
#Thus we can see that there are no more null values in the age column

sns.heatmap(traindata.isnull(),yticklabels=False,cbar=False,cmap='viridis')
traindata.head()
#Now we plot the number of survivors 

sns.set_style('whitegrid')

sns.countplot(x='Survived',data=traindata,palette='RdBu_r')
#We see how the Sex of the passenger determines the survival of a passenger

sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=traindata,palette='RdBu_r')
#We see how the class the passenger travels in determines the survival of a passenger

sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=traindata,palette='rainbow')
#We convert the names column to just the honorofic to check if it influences the survival rate

col_one_list = traindata['Name'].tolist()

p=[]

for a in col_one_list:

    b=a.split(' ')

    #print(b)

    #p.append(b[1])

    if b[1]=='Mr.' or b[1]=='Mrs.' or b[1]=='Miss.' or b[1]=='Master.':

          p.append(b[1])

    else:

          p.append('rare')

            

traindata['honorifics'] = p

traindata.tail()
# We perform one hot encoding on the honorifics column

one_hot = pd.get_dummies(traindata['honorifics'])

traindata = traindata.drop('honorifics',axis = 1)

traindata = traindata.join(one_hot)

traindata.head()
#We remove the Cabin column because it has too many null values.

traindata=traindata.drop('Cabin',axis=1)

#We also remove the Ticket number, passenger ID and the Name columns because they are irrelavant to our model



traindata=traindata.drop('PassengerId',axis=1)

traindata=traindata.drop('Name',axis=1)

traindata=traindata.drop('Ticket',axis=1)
#Perform one hot encoding on the Pclass column

one_hot = pd.get_dummies(traindata['Pclass'])

# Drop column Product as it is now encoded

traindata = traindata.drop('Pclass',axis = 1)

# Join the encoded df

traindata = traindata.join(one_hot)

traindata.head()
# Similarly, perform one hot encoding on the Embarked column

one_hot = pd.get_dummies(traindata['Embarked'])

traindata = traindata.drop('Embarked',axis = 1)

traindata = traindata.join(one_hot)

traindata.head()
#Change the categorical variables in the Sex column to numbers

traindata['Sex'] = traindata['Sex'].replace('male', 0)

traindata['Sex'] = traindata['Sex'].replace('female', 1)

traindata.head()




y=traindata['Survived']

x=traindata.drop('Survived',axis=1)

#Splitting training and testing data

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)






#Logistic Regression

LogisticRegressor = LogisticRegression(max_iter=10000)

LogisticRegressor.fit(x_train, y_train)

y_predicted = LogisticRegressor.predict(x_test)

mse = mean_squared_error(y_test, y_predicted)

r = r2_score(y_test, y_predicted)

mae = mean_absolute_error(y_test,y_predicted)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)

print('accuracy score:')

print(accuracy_score(y_test,y_predicted))

print('f1 score:')

print(f1_score(y_test,y_predicted))
# Random Forest

rf = RandomForestClassifier()

rf.fit(x_train,y_train);

y_predicted_r = rf.predict(x_test)

mse = mean_squared_error(y_test, y_predicted_r)

r = r2_score(y_test, y_predicted_r)

mae = mean_absolute_error(y_test,y_predicted_r)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)

print('accuracy score:')

print(accuracy_score(y_test,y_predicted_r))

print('f1 score:')

print(f1_score(y_test,y_predicted_r))
math.sqrt(len(y_test)) #Therefore we use 15 as the number of neighbors
#KNN

math.sqrt(len(y_test))



classify= KNeighborsClassifier (n_neighbors=15, p =2, metric= 'euclidean')

classify.fit(x_train,y_train)

ypred1=classify.predict(x_test)



msee = mean_squared_error(y_test, ypred1)

r = r2_score(y_test, ypred1)

maee = mean_absolute_error(y_test,ypred1)

print("Mean Squared Error:",msee)

print("R score:",r)

print("Mean Absolute Error:",maee)



print('f1 score:')

print(f1_score(y_test,ypred1))

print('accuracy score:')

print(accuracy_score(y_test,ypred1))
#SVM



svclassifier = SVC(kernel='linear')

svclassifier.fit(x_train, y_train)

y_pred2 = svclassifier.predict(x_test)



mseew = mean_squared_error(y_test, y_pred2)

ra = r2_score(y_test, y_pred2)

maeew = mean_absolute_error(y_test,y_pred2)

print("Mean Squared Error:",mseew)

print("R score:",ra)

print("Mean Absolute Error:",maeew)



print('f1 score:')

print(f1_score(y_test,y_pred2))

print('accuracy score:')



print(accuracy_score(y_test,y_pred2))
#DecisionTreeClassifier

dt_model=DecisionTreeClassifier()

dt_model.fit(x_train,y_train)

dt_pred = dt_model.predict(x_test)



msee1 = mean_squared_error(y_test, y_pred2)

ra1 = r2_score(y_test, y_pred2)

maee1 = mean_absolute_error(y_test,y_pred2)

print("Mean Squared Error:",msee1)

print("R score:",ra1)

print("Mean Absolute Error:",maee1)

print('f1 score:')

print(f1_score(y_test,dt_pred))

print('accuracy score:')

print(accuracy_score(y_test,dt_pred))
#XGBClassifier

xgboost = XGBClassifier(n_estimators=1000)

xgboost.fit(x_train,y_train)

xg_pred = xgboost.predict(x_test)

msee21 = mean_squared_error(y_test, y_pred2)

ra21 = r2_score(y_test, y_pred2)

maee21 = mean_absolute_error(y_test,y_pred2)

print("Mean Squared Error:",msee21)

print("R score:",ra21)

print("Mean Absolute Error:",maee21)

print('f1 score:')

print(f1_score(y_test,xg_pred))

print('accuracy score:')

print(accuracy_score(y_test,xg_pred))
filepath= '/kaggle/input/titanic/test.csv'

testdata= pd.read_csv(filepath)

submissiondata=testdata

testdata.head()
testdata = testdata[['PassengerId','Age','Pclass','Name','Sex','SibSp','Parch','Ticket','Cabin','Embarked']]

testdata.head()

#We convert the names column to just the honorofic to check if it influences the survival rate

col_one_list = testdata['Name'].tolist() 

p=[]

for a in col_one_list:

    b=a.split(' ')

    #print(b)

    #p.append(b[1])

    if b[1]=='Mr.' or b[1]=='Mrs.' or b[1]=='Miss.' or b[1]=='Master.':

          p.append(b[1])

    else:

          p.append('rare')

            

testdata['honorifics'] = p

testdata.tail()

sns.heatmap(testdata.isnull(),yticklabels=False,cbar=False,cmap='viridis')

testdata=testdata.drop('Cabin',axis=1)

testdata=testdata.drop('PassengerId',axis=1)

testdata=testdata.drop('Name',axis=1)

testdata=testdata.drop('Ticket',axis=1)

sns.heatmap(testdata.isnull(),yticklabels=False,cbar=False,cmap='viridis')

testdata['Age'].fillna((testdata['Age'].mean()), inplace=True)
# Perform one hot encoding on the honorifics column

one_hot = pd.get_dummies(testdata['honorifics'])

testdata = testdata.drop('honorifics',axis = 1)

testdata = testdata.join(one_hot)

testdata.head()
#Perform one hot encoding on the Pclass column

one_hot = pd.get_dummies(testdata['Pclass'])

# Drop column Product as it is now encoded

testdata = testdata.drop('Pclass',axis = 1)

# Join the encoded df

testdata = testdata.join(one_hot)

testdata.head()
# Similarly, perform one hot encoding on the Embarked column

one_hot = pd.get_dummies(testdata['Embarked'])

testdata = testdata.drop('Embarked',axis = 1)

testdata = testdata.join(one_hot)

testdata.head()
#Change the categorical variables in the Sex column to numbers

testdata['Sex'] = testdata['Sex'].replace('male', 0)

testdata['Sex'] = testdata['Sex'].replace('female', 1)

testdata.head()
sns.heatmap(testdata.isnull(),yticklabels=False,cbar=False,cmap='viridis')





#Logistic Regression

LogisticRegressor = LogisticRegression(max_iter=10000)

LogisticRegressor.fit(x_train, y_train)

y_predicted = LogisticRegressor.predict(testdata)
predictionlist=y_predicted.tolist()

Passengerid=submissiondata['PassengerId'].tolist() 

output=pd.DataFrame(list(zip(Passengerid, predictionlist)),

              columns=['PassengerId','Survived'])

output.head()

output.to_csv('my_submission.csv', index=False)