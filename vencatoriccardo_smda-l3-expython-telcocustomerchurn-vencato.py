# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv') #get the data

data.shape #number of columns and rows

data.columns.values   #gives nme the value of the columns

#churn is our "target" boolean that tells me if my client is still doing buisiness with my company

#9.in the prevoius box i add all the extra library

#



#data acquisation:

data = pd.read_csv('../input/../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

#database dimension

print("the number of costumers is:"+ str(data.shape[0]))

print("the number of variables is:"+ str(data.shape[1]))

print("the total number of cell is:"+ str(data.shape[0]*data.shape[1]))

print("my columns are:")

print(data.columns.values)
#Question 21

data.head()

data.tail()
#type not correct

print("data is of the type: "+str(type(data)))

#number of missing value

data.isna().sum()

 #type of the columns

print(data.dtypes)

#la dimensione si fotte(linda studiaaaaaaa)

#categorical(variables that can take only certain values ex. blood type) variable in my data

#print those variable that are categorical!

print("Categorical variables:")

for x in range(data.dtypes.shape[0]):

        if(data.dtypes[x]=="object"):

            print(data.columns[x])

print("Numerical variables:")

for x in range(data.dtypes.shape[0]):

        if(data.dtypes[x]=="int64" or data.dtypes[x]=="float64"):

            print(data.columns[x])

        
#23

data.mean(axis=0) #zero means that use the index 1 equal columns

data.std(axis=0)

data.min(axis=0)

data.max(axis=0)

data.quantile(axis=0)

data.shape[1]-data.isna().sum() #tot value- missing ones

data['gender'].value_counts() #counts male and female values

data['gender'].value_counts().idxmax(axis=0) #gives me who got the most

data.groupby(['gender']).transform('count')
#25

data.hist(column='tenure') 
data.boxplot()

fig, axes = plt.subplots(nrows=1, ncols=3)

 #just a single column



data.boxplot(column='SeniorCitizen',ax=axes[0])

data.boxplot(column='tenure', ax=axes[1])

data.boxplot(column='MonthlyCharges',ax=axes[2])



fig, axes = plt.subplots(nrows=1, ncols=2)

sns.countplot(x='gender',data=data, ax=axes[0]) #gives me a graph with the number of categorical variables

sns.countplot(x='Partner',data=data ,ax=axes[1])
#28

data1 = data[['gender', 'Partner','MonthlyCharges','Churn']]

print(data1)
#change genders into binary value



data1 = data[['gender', 'Partner','MonthlyCharges','Churn']]

data1=data1.replace({'gender': {'Female': 0, 'Male': 1}})#replace with binary value

print(data1)

print(data1)