import pandas as pd

import numpy as np

import matplotlib

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import  cross_val_score

from sklearn.cross_validation import train_test_split

import scipy as sp



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr



# Load in the train and test datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Store test passenger ID for submission

PassengerId = test['PassengerId']
# Begin Feature Engineering

combi = pd.concat((train.loc[:,'Pclass':'Embarked'],

                      test.loc[:,'Pclass':'Embarked']))



#Create Family Size

combi['FamilySize'] = combi['SibSp'] + combi['Parch'] + 1



#Create Surname and Title

combi['Surname'],combi['temp2'] = combi['Name'].str.split(',',1).str

combi['Title'],combi['temp3'] = combi['temp2'].str.split('.',1).str



combi['Title'] = combi['Title'].map(str.strip) # To strip out whitespaces

combi['Title'].replace('Capt','Sir',inplace=True)

combi['Title'].replace('Don','Sir',inplace=True)

combi['Title'].replace('Major','Sir',inplace=True)

combi['Title'].replace('Dona','Lady',inplace=True)

combi['Title'].replace('the Countess','Lady',inplace=True)

combi['Title'].replace('Jonkheer','Lady',inplace=True)

combi['Title'].replace('Mme','Mlle',inplace=True)



#Create FamId

combi['FamId'] = combi.FamilySize.map(str).str.cat(combi.Surname)



#Cleanup FamId

x=combi['FamId'].value_counts()

for y in combi['FamId'].unique():

    if x[y] <= 2:

        combi['FamId'].replace(y,'Small',inplace=True)



#combi['FamId'].value_counts()

#Remove Unwanted Columns

for names in ['Name','temp2','temp3','Ticket','Cabin','Surname']:

    combi.pop(names)



#Get Dummies

combi = pd.get_dummies(combi)



#filling NA's with the mean of the column:

combi = combi.fillna(combi.mean())



#creating matrices for model:

X_train = combi[:train.shape[0]]

X_test = combi[train.shape[0]:]

y = train.Survived



combi.head()
# Run a Logistic Regression

clf = LogisticRegression()



# Use Full dataset to fit and predict

clf.fit(X_train, y)

predictions = clf.predict(X_test)
predictions[predictions>=0.5]=1

predictions[predictions<0.5]=0



# Generate Submission File 

submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

submission.to_csv("submissionLogistic.csv", index=False)