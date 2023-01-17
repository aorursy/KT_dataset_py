#import python packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import cross_validation, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score 
%matplotlib inline
#import dataset from draft environment
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.columns, test.columns
#Descriptive statistic of Titanic data
train.describe()
#Check data type
train.dtypes
#titanic info
train.info()
#check missing value
train.isnull().sum()
sum(pd.isnull(train['Age']))
# proportion of "Age" missing
round(177/(len(train["PassengerId"])),4)
# proportion of "cabin" missing
round(687/len(train["PassengerId"]),4)
# proportion of "Embarked" missing
round(2/len(train["PassengerId"]),4)
# median age is 28 (as compared to mean which is ~30)
train["Age"].median(skipna=True)
#final adjustment
train_data = train
train_data["Age"].fillna(28, inplace=True)
train_data["Embarked"].fillna("S", inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)
## Create categorical variable for traveling alone
train_data['TravelBuds']=train_data["SibSp"]+train_data["Parch"]
train_data['TravelAlone']=np.where(train_data['TravelBuds']>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)
train_data.drop('TravelBuds', axis=1, inplace=True)
#create categorical variable for Pclass

train2 = pd.get_dummies(train_data, columns=["Pclass"])
train3 = pd.get_dummies(train2, columns=["Embarked"])
train4=pd.get_dummies(train3, columns=["Sex"])
train4.drop('PassengerId', axis=1, inplace=True)
train4.drop('Name', axis=1, inplace=True)
train4.drop('Ticket', axis=1, inplace=True)
train4.head(5)
final_train = train4

final_train.head()
#final adjustment
test_data = test
test_data["Age"].fillna(28, inplace=True)
test_data["Embarked"].fillna("S", inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

## Create categorical variable for traveling alone

test_data['TravelBuds']=test_data["SibSp"]+test_data["Parch"]
test_data['TravelAlone']=np.where(test_data['TravelBuds']>0, 0, 1)

test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)
test_data.drop('TravelBuds', axis=1, inplace=True)

#create categorical variable for Pclass

test2 = pd.get_dummies(test_data, columns=["Pclass"])

test3 = pd.get_dummies(test2, columns=["Embarked"])

test4=pd.get_dummies(test3, columns=["Sex"])

test4.drop('PassengerId', axis=1, inplace=True)
test4.drop('Name', axis=1, inplace=True)
test4.drop('Ticket', axis=1, inplace=True)
test4.head(5)

final_test=test4
final_test.head()
ax = train["Age"].hist(bins=15, color='green', alpha=0.8)
ax.set(xlabel='Age', ylabel='Count')
plt.show()
#Explor Age Variable
plt.figure(figsize=(10,5))
train['Age'].plot.hist(bins=35)
plt.figure(figsize=(10,5))
train[train['Survived']==0]['Age'].hist(bins=35,color='blue',
                                       label='Survived = 0', 
                                        alpha=0.6)
train[train['Survived']==1]['Age'].hist(bins=35,color='red',
                                       label='Survived = 1',
                                       alpha=0.6)
plt.legend()
plt.xlabel("The Number of Age")
sns.countplot(x='Embarked',data=train,palette='Set2')
plt.show()
#Import Packages for Machine Learning models 
## the packages is Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cross_validation import train_test_split
x = final_train.drop('Survived', axis=1)
y = final_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(x , y, test_size = 0.20)

Logmodel = LogisticRegression()
Logmodel.fit(X_train,y_train)
pred_LR = Logmodel.predict(X_test)
print(confusion_matrix(y_test,pred_LR))
print('\n')
print(classification_report(y_test,pred_LR))
Accuracy_LR = print ('1. Accuracy_L.Regression_Classifier :', 
                     accuracy_score(y_test,pred_LR)*100)
x = final_train.drop('Survived', axis=1)
y = final_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(x , y, test_size = 0.20)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred_tree = dtree.predict(X_test)
print(confusion_matrix(y_test,pred_tree))
print('\n')
print(classification_report(y_test,pred_tree))
Accuracy_DT = print ('2. Accuracy_D.Tree_Classifier :', 
                     accuracy_score(y_test,pred_tree)*100)
x = final_train.drop('Survived', axis=1)
y = final_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(x , y, test_size = 0.20)

Rfc = RandomForestClassifier(n_estimators = 300 )
Rfc.fit(X_train,y_train)
Pred_Rfc =Rfc.predict(X_test)
print(confusion_matrix(y_test,Pred_Rfc))
print ('\n')
print(classification_report(y_test,Pred_Rfc))
Accuracy_RF = print ('3. Accuracy_R.Forest_Classifier :', 
                     accuracy_score(y_test,Pred_Rfc)*100)