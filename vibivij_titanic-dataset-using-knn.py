# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# machine learning

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
train_df.head().T
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
test_df.head().T
# preview the transpose Data
train_df.head().T
# Get an idea about the data 
train_df.describe(include = 'all')
test_df.describe(include = 'all')
# looking at Data we can see like the Passenger ID and ticket number doesnt add an value to the model.
#passenger id only deleted from he training Data set as it doesnt add value.. in the test data set we will Keep 
# for later purpose of prediction. 

train_df = train_df.drop(['PassengerId','Name','Ticket'], axis=1)
test_df  = test_df.drop(['Name','Ticket'], axis=1)
# check the data set for missing value and general information. 
train_df.info()
# we can see the missing values in the train Data set 
train_df.isnull().sum()
# we can see the missing values in the test Data set 
test_df.isnull().sum()
# analyze each variable by using graph and plotting. 
# we can get a rough idea on the distribution  of each variable on a scale. 

train_df.hist(figsize=(12,8))
plt.show()
#start looking at each variable
# Embarked variable 

# 2 missing data in the train Data set .. So fill it with Mode. 

train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)

# check to verify the null is gone in the embarked 

train_df.isnull().sum()

# Plot the embarked and survival relation

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.countplot(x='Embarked', data=train_df , ax = axis1 )
sns.countplot(x='Survived', hue="Embarked", data=train_df, order=[1,0] , ax = axis2)

combined = train_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=combined,order=['S','C','Q'],ax=axis3)

# convert the category variable to numeric by dummy variable method


train_df1 = pd.get_dummies(train_df, prefix ='Embark', columns = ['Embarked'])
test_df1 = pd.get_dummies(test_df, prefix ='Embark', columns = ['Embarked'])


train_df1.head().T
# Fare 
sns.boxplot(x='Fare', data=train_df1)

# since there is a missing "Fare" values
test_df1["Fare"].fillna(test_df1["Fare"].median(), inplace=True)
#age 
# age has missing values in the train and test . replaced with the median 

test_df1["Age"].fillna(test_df1["Age"].median(), inplace=True)
train_df1["Age"].fillna(train_df1["Age"].median(), inplace=True)

# check if the NAN has removed with median
test_df1.isnull().sum()

# convert from float to int
train_df1['Age'] = train_df1['Age'].astype(int)
test_df1['Age']    = test_df1['Age'].astype(int)

#plot for aged and survived
# average survived passengers by age

fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = train_df1[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
# Cabin
# as mentioned earlier this feature can be deleted as it wont add value to prediction

train_df1.drop("Cabin",axis=1,inplace=True)
test_df1.drop("Cabin",axis=1,inplace=True)

train_df1.head().T
# Family

# we dont need both these  two columns Parch & SibSp, 
# we need  one column represent if the passenger is alone or not

train_df1['Family'] =  train_df1["Parch"] + train_df1["SibSp"]
train_df1['Family'].loc[train_df1['Family'] > 0] = 1
train_df1['Family'].loc[train_df1['Family'] == 0] = 0

test_df1['Family'] =  test_df1["Parch"] + test_df["SibSp"]
test_df1['Family'].loc[test_df1['Family'] > 0] = 1
test_df1['Family'].loc[test_df1['Family'] == 0] = 0

# drop Parch & SibSp
train_df1 = train_df1.drop(['SibSp','Parch'], axis=1)
test_df1   = test_df1.drop(['SibSp','Parch'], axis=1)

# plot
fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

# sns.factorplot('Family',data=train_df1,kind='count',ax=axis1)
sns.countplot(x='Family', data=train_df1, order=[1,0], ax=axis1)

# average of survived for those who had/didn't have any family member
family_perc = train_df1[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)

axis1.set_xticklabels(["With Family","Alone"], rotation=0)


#sex
# convert all category variables to dummy variable 

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

train_df2 = pd.get_dummies(train_df1, prefix ='Sex', columns = ['Sex'])

test_df2 = pd.get_dummies(test_df1, prefix ='Sex', columns = ['Sex'])

train_df2.head().T
#see the correlation between each variable 
sns.heatmap(train_df2.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()
#preprocessing
X= train_df2.drop('Survived',axis=1)
y=train_df2['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 10)
print ("X_train shape", X_train.shape)
print ("y_train shape", y_train.shape)
print ("X_test shape", X_test.shape)
print ("y_test shape", y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# starting with the logistic regression 

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_predlr = logreg.predict(X_test)
acc_logreg = round(accuracy_score(y_predlr, y_test) * 100, 2)
print("Logistic Regression accuracy is :",acc_logreg)
# checking the decision tree

from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier(max_depth=4,min_samples_leaf=4)
decisiontree.fit(X_train, y_train)
y_preddt = decisiontree.predict(X_test)
acc_decisiontree = round(accuracy_score(y_preddt, y_test) * 100, 2)
print("Decision Tree accuracy is :", acc_decisiontree)

# Gaussian Naive Bayes

#gaussian = GaussianNB()

#gaussian.fit(X_train, y_train)

#Y_pred = gaussian.predict(X_test)

#gaussian.score(X_train, y_train)

#from sklearn.ensemble import RandomForestClassifier

#randomforest = RandomForestClassifier(random_state=7)
#randomforest.fit(X_train, y_train)
#y_predrf = randomforest.predict(X_test)
#acc_randomforest = round(accuracy_score(y_predrf, y_test) * 100, 2)
#print("Random Forest accuracy is :",acc_randomforest)
# KNN Classification approach

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

knn.score(X_train, y_train)
# going with the KNN approach as the scores compared to others are high 
#use the  K fold to evaluate the module 
from sklearn.model_selection import cross_validate

scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}

modelCV = KNeighborsClassifier()

results = cross_validate(modelCV, X_train, y_train, cv=10, scoring=list(scoring.values()), 
                         return_train_score=False)

print('K-fold cross-validation results:')
for sc in range(len(scoring)):
    print(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values())[sc]].mean()
                               if list(scoring.values())[sc]=='neg_log_loss' 
                               else results['test_%s' % list(scoring.values())[sc]].mean(), 
                               results['test_%s' % list(scoring.values())[sc]].std()))
#import GridSearchCV
from sklearn.model_selection import GridSearchCV
#In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X_train, y_train)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))



X_test_original = test_df2.drop("PassengerId",axis=1).copy()



Y_pred = knn.predict(X_test_original)
finalsub = pd.DataFrame({
        'PassengerId':test_df2["PassengerId"],
        'Survived': Y_pred.astype(int) })
   
finalsub.head()
finalsub.to_csv("Subtitanic.csv",index=False)
