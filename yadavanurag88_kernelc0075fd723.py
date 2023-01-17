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
import pandas as pd

loan = pd.read_csv("../input/loan_prediction.csv") #use read_csv to read the dataset
#to view the top 5 rows of the dataset

loan.head()
#for describe brief functions about dataset (like min, max, mean.... etc)

loan.describe()
import matplotlib.pyplot as plt

loan[loan.Loan_Amount_Term>=150].plot(kind="scatter",x="ApplicantIncome",y="CoapplicantIncome",color="pink")

plt.xlabel("ApplicantIncome")

plt.ylabel("CoapplicantIncome")

plt.title("Loan_Amount_Term>=150")

plt.grid(True)

plt.show()
X=loan.drop(['Loan_Status','Loan_ID'], axis=1)

Y=loan['Loan_Status']
X.isnull().sum()
X.info()
X['Gender'].value_counts()
X['Gender'].fillna("Male", inplace=True)
X['Married'].value_counts()
X['Married'].fillna("Yes", inplace=True)
X['Dependents'].fillna(X['Dependents'].mode()[0],inplace=True)
X['Self_Employed'].fillna(X['Self_Employed'].mode()[0],inplace=True)
X.isnull().sum()
mean_loan=X['LoanAmount'].mean()

X['LoanAmount'].fillna(mean_loan,inplace=True)
X['Loan_Amount_Term'].fillna(X['Loan_Amount_Term'].mean(),inplace=True)

X['Credit_History'].fillna(X['Credit_History'].mean(),inplace=True)
X.isnull().sum()
X=pd.get_dummies(X)
X.head()
X.columns
from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25)
X_train.shape
Y_train.shape
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()



lr.fit(X_train,Y_train)
from sklearn.svm import SVC

svc = SVC()



svc.fit(X_train,Y_train)
from sklearn.tree import DecisionTreeClassifier

dtf = DecisionTreeClassifier()

dtf.fit(X_train, Y_train)
from sklearn.naive_bayes import GaussianNB

n_b = GaussianNB()

n_b.fit(X_train, Y_train)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()  

knn.fit(X_train, Y_train)
print(lr.score(X_train, Y_train))
print(lr.score(X_test, Y_test))

print(dtf.score(X_test, Y_test))

print(n_b.score(X_test, Y_test))

print(knn.score(X_test, Y_test))

print(svc.score(X_test, Y_test))