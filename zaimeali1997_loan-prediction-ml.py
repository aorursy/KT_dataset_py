import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv')
data.head()
data.describe()
print("Length of Data is: ", len(data.index))
print("Null Values in a Data: ")

data.isnull().any()
data.groupby('Gender').count()
data.groupby('Married')['Loan_ID'].count()
data.Property_Area.value_counts()
data.hist(column='ApplicantIncome', by='Education')
data.isnull().sum()
# We can also do this with apply method

data.apply(lambda x: sum(x.isnull()), axis = 0)
data.shape # to check number of columns
data.nunique()
data.info()
data.Property_Area.unique()
# Convert Object to Category

catList = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area', 'Loan_Status']

for cat in catList:

    data[cat] = data[cat].astype('category')
data.info()
fillList = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']

for fill in fillList:

    data[fill].fillna(method='ffill', inplace=True)
data.isnull().sum()
data.boxplot(column='LoanAmount')

plt.show()
data.boxplot(column='Loan_Amount_Term')

plt.show()
medianList = ['LoanAmount', 'Loan_Amount_Term']

for med in medianList:

    data[med].fillna(data[med].median(), inplace=True)
data.isnull().sum()
# For visualize the data

import seaborn as sns

from matplotlib import pyplot as plt
data['Credit_History'].hist()

plt.show()
data.groupby('Credit_History')['Loan_Status'].count()
data.groupby('Loan_Status')['Credit_History'].count()
data.groupby(['Loan_Status','Gender']).sum()
data.groupby(['Loan_Status', 'Education']).sum()
data.groupby(['Loan_Status', 'Education', 'Gender']).sum()
data.Credit_History.unique()
greaterMeanIncome = data['ApplicantIncome'] > data.ApplicantIncome.mean() # because here we are choosing the avg Income of the applicant

isGraduate = (data.Education == 'Graduate')

isMarried = (data.Married == 'No')

loanStatus = (data.Loan_Status == 'Y')

data[(isMarried) & (isGraduate) & (greaterMeanIncome) & (loanStatus)]['Self_Employed'].value_counts()
greaterMeanIncome = data['ApplicantIncome'] > data.ApplicantIncome.mean() # because here we are choosing the avg Income of the applicant

isGraduate = (data.Education != 'Graduate')

isMarried = (data.Married != 'No')

loanStatus = (data.Loan_Status == 'Y')

data[(isMarried) & (isGraduate) & (greaterMeanIncome) & (loanStatus)]['Self_Employed'].value_counts()
data.groupby(['Gender', 'Education'])['Loan_Status'].count().plot(kind='bar')
data.groupby(['Gender', 'Education', 'Married'])['Loan_Status'].count().plot(kind='bar')
data.groupby(['Gender', 'Education', 'Married', 'Self_Employed'])['Loan_Status'].count().plot(kind='bar')
# map Loan Status Y=1 , N=0 

data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0})

data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})

data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})

data['Property_Area'] = data['Property_Area'].map({'Urban':1,'Rural':2,'Semiurban':3})

data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})

# for a fast computation
# we can also change categorical values into one hot encoding through dummies which pd.get_dummies then concatenate

# but we are going to do here
data.set_index('Loan_ID', inplace=True)

data.head()
X = data.iloc[:, :-1].values

y = data.iloc[:, -1].values
print("X Shape", X.shape)

print("Y Shape", y.shape)

y = y.astype('int64')
data.iloc[:, :-1].info()
X = pd.DataFrame(X)
X.columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
X.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Property_Area']] = scaler.fit_transform(X[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Property_Area']])
X.head()
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

X = ohe.fit_transform(X)
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
print("Train X Shape", train_X.shape)

print("Test X Shape", test_X.shape)
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier() # with no parameters define

dtc.fit(train_X, train_y)

pred_y = dtc.predict(test_X)

print("Score of Decision Tree: ", dtc.score(test_X, test_y))

print("Accuracy of Decision Tree: ", accuracy_score(test_y, pred_y))

print("Confusion Matrix of Decision Tree: \n", confusion_matrix(test_y, pred_y))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42)

rfc.fit(train_X, train_y)

pred_y = rfc.predict(test_X)

print("Score of Random Forest: ", rfc.score(test_X, test_y))

print("Accuracy of Random Forest: ", accuracy_score(test_y, pred_y))

print("Confusion Matrix of Random Forest: \n", confusion_matrix(test_y, pred_y))
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()

sgd.fit(train_X, train_y)

pred_y = sgd.predict(test_X)

print("Score of SGD: ", sgd.score(test_X, test_y))

print("Accuracy of SGD: ", accuracy_score(test_y, pred_y))

print("Confusion Matrix of SGD: \n", confusion_matrix(test_y, pred_y))