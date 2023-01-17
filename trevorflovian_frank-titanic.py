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



#!/usr/bin/env python



import codecs

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import scipy

from scipy.stats import norm

from scipy.interpolate import UnivariateSpline

import pandas as pd

import matplotlib.backends.backend_pdf

import csv

import math



data = pd.read_csv("/kaggle/input/general-titanic-dataset/titanic.csv")

test = pd.read_csv("../input/titanic/test.csv")

#training set has labels and is usde to train our model

train = pd.read_csv("../input/titanic/train.csv")



combined = [train, test]

# Family column

data['Family'] = (data['SibSp'] > 0) | (data['Parch'] > 0)

#print (data["Family"])



# Age Analysis

#classify as children and adults

data['AgeRange'] = pd.cut(data['Age'], [0, 15, 80], labels=['child', 'adult'])

#print(data['AgeRange'])

# remove missing ages

data_clean_age = data.dropna(subset=['Age'])

#print(data_clean_age)

f = data.groupby("Age")

#print(f)

mean_age = f.mean()

var_age = f.var()

std_age = np.sqrt(np.var(data_clean_age))

# print ("Mean: " + str(np.mean(data_clean_age)))

# print ("Variance: " +str(np.var(data_clean_age)))

# print ("Standard deviation: " +str(np.sqrt(np.var(data_clean_age))))

# print(mean_age)

# print(var_age)

# print(std_age)



plt.hist(data["Age"], bins = 100, density = True)

plt.ylabel('Count')

plt.title('Histogram for Age in Titanic')

#plt.savefig("figure3.pdf")

plt.show()







#fare analysis

data_clean_fare = data.dropna(subset=['Fare'])

f = data.groupby("Fare")

plt.hist(data["Fare"], bins = 100, density = True)

plt.ylabel('Count')

plt.title('Histogram for fare in Titanic')

#plt.savefig("figure4.pdf")

plt.show()

mean_fare = f.mean()

var_fare = f.var()

std_fare = np.sqrt(np.var(data_clean_fare))

# print ("Mean: " + str(np.mean(data_clean_fare)))

# print ("Variance: " +str(np.var(data_clean_fare)))

# print ("Standard deviation: " +str(np.sqrt(np.var(data_clean_fare))))



# print("Mean fare: "+str(mean_fare))

# print("fare Variance:"+str(var_fare))

# print("Fare Standard Dev:"+str(std_fare))







# #Normal probability plot for age



#counts, start= scipy.stats.probplot(data["Age"],sparams=(), dist='norm', fit=True, plot=None, rvalue=False)

#x = np.arange(counts.size) * 1 + start



# data["Age"].sort()

# X = np.linspace(1.0/len(data["Age"]), 1, len(data["Age"]))

# Ppf_age = scipy.stats.norm.ppf(X, mean_age, var_age)

counts, start = scipy.stats.probplot(data["Age"],sparams=(),dist = 'norm', fit = True, plot = plt)

# plt.plot(Ppf_age, counts, 'ro')

plt.xlabel('Value')

plt.ylabel('Normalized count')

plt.title('Probability plot for age in Titanic')

#plt.savefig("figure5.pdf")

plt.show()





# #Normal probability plot for fare

counts, start = scipy.stats.probplot(data["Fare"],sparams=(),dist = 'norm', fit = True, plot = plt)

plt.xlabel('Value')

plt.ylabel('Normalized count')

plt.title('Probability plot for Fare in Titanic')

#plt.savefig("figure6.pdf")

plt.show()





# Survival Rates Analysis







# remove data not needed

data.drop(['PassengerId', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

#survival rates

survived_by_class = data.groupby('Pclass')['Survived'].mean()

print(survived_by_class)

survived_by_sex = data.groupby('Sex')['Survived'].mean()

print(survived_by_sex)

survived_by_age = data.groupby('AgeRange')['Survived'].mean()

print(survived_by_age)



# Survival by Gender

print(data.groupby(['Sex', 'Survived'])['Survived'].count())



# Survival by Class and Gender

pd.crosstab([data.Sex, data.Survived], data.Pclass, margins= True).style.background_gradient(cmap='summer_r') 



fig, (axis1,axis2,axis3) = plt.subplots(1, 3, figsize=(16,6))



ax = survived_by_class.plot.bar(ax=axis1, color='#5975A4', title='Survival Rate by Class', sharey=True)

ax.set_ylabel('Survival Rate')

ax.set_ylim(0.0,1.0)

ax = survived_by_sex.plot.bar(ax=axis2, color='#5F9E6E', title='Survival Rate by Sex', sharey=True)

ax.set_ylim(0.0,1.0)

ax = survived_by_age.plot.bar(ax=axis3, color='#B55D60', title='Survival Rate by Age Range', sharey=True)

ax.set_ylim(0.0,1.0)





#import Shallow Machine Learning library (i.e., sklearn)

#import the Random Forest Algorithm



## copied from https://www.kaggle.com/diegogomez92/titanic-diego

from sklearn.ensemble import RandomForestClassifier

import pandas as pd

#this data set has all females surviving and all males not surviving

#get ball-park estimate about what feautures are important in the data set

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")



#testing data has no labels (i.e., does not include 'Survived' variable)

test = pd.read_csv("../input/titanic/test.csv")

#training set has labels and is usde to train our model

train = pd.read_csv("../input/titanic/train.csv")

#show top 5 rows of the data set 

train.head()

#DEPENDENT VARIABLE

y = train["Survived"]



#INDEPENDENT VARIABLES

#the features we will include in our model

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])





X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

x_test  = test.drop("PassengerId", axis=1).copy()





#More imports for analysis



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

# Logistic Regression courtesy of https://www.kaggle.com/startupsci/titanic-data-science-solutions/data



logreg = LogisticRegression()

logreg.fit(X, y)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X, y) * 100, 2)

acc_log





# Support Vector Machines



svc = SVC()

svc.fit(X, y)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X, y) * 100, 2)

acc_svc





# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X, y)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X, y) * 100, 2)

acc_decision_tree



# Random Forest version 1



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X, y)

Y_pred = random_forest.predict(X_test)

random_forest.score(X, y)

acc_random_forest = round(random_forest.score(X, y) * 100, 2)

acc_random_forest





output = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

output.to_csv('submission.csv', index=False)

print("Your first submission was successfully saved!")





#100 random forest trees

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y) #fit the model

predictions = model.predict(X_test) 



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")