import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv')

data.head()
data.set_index('Loan_ID', inplace=True)

data.head()
print("Data Numerical Columns Description")

data.describe()
print('Data All Columns Description')

data.describe(include='all')
print("Check Missing Value:")

data.isna().sum()
print('Data Information:')

data.info()
print("To Check Numbers of Unique Value each Column Contains:")

data.nunique()
data.boxplot(figsize=(16, 10))

plt.show()
print("Probability of getting the loan on checking the credit history")

data.groupby('Loan_Status')['Credit_History'].mean()
print("Checking the Cross Tab:")

pd.crosstab(data['Credit_History'], data['Loan_Status'])
print('Filling the missing values in Loan Amount with the mean of Loan Amount:')

data.LoanAmount.fillna(data.LoanAmount.mean(), inplace=True)

print("To check is there any missing value left in Loan Amount")

data.LoanAmount.isna().sum()
print("Checking missing values in Self Employed Column:", data.Self_Employed.isna().sum())

print("Unique Values Count:")

print(data.Self_Employed.value_counts())
# so we can fill with No

data.Self_Employed.fillna('No', inplace=True)

print("Checking missing values in Self Employed Column:", data.Self_Employed.isna().sum())
print("Self Employed person with its Education asking for a loan which we have taken an average of:")

data.pivot_table(index='Self_Employed', values='LoanAmount', columns='Education', aggfunc=np.median)
print("Filling the categorical values with mode function:")

modeList = ['Gender', 'Married', 'Dependents', 'Loan_Amount_Term', 'Credit_History']

for mCol in modeList:

    data[mCol].fillna(data[mCol].mode()[0], inplace=True)

    

print("Now Checking the missing values of the columns define in the list:")

print(data[modeList].isna().sum())
from sklearn.preprocessing import LabelEncoder

var = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

lE = LabelEncoder()

for v in var:

    data[v] = lE.fit_transform(data[v])
print("Now checking the data information")

print(data.info())

print("Data Columns Data Type:")

print(data.dtypes)

# so now all values are numeric
print("Checking the Correlation of Columns")

data[['Gender', 'Education', 'Credit_History', 'Self_Employed', 'Married', 'Property_Area', 'ApplicantIncome', 'Loan_Status']].corr()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier
def Custom_Model(X, y, model, model_name):

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

    model.fit(train_X, train_y)

    pred_y = model.predict(test_X)

    print("Accuracy of the " + model_name + ": ", model.score(test_X, test_y))
y = data['Loan_Status']
# with only one feature

# as we can see from correlation table credit history is highly correlated to loan status

X = data[['Credit_History']]

lr = LogisticRegression()

Custom_Model(X, y, lr, "Logistic Regression")
data.columns
# with more predictor with no checking

X = data[['Gender', 'Education', 'Credit_History', 'Self_Employed', 'Married', 'ApplicantIncome', 'Property_Area']]

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, lr, "Logistic Regression")
# with Correlation near 1

X = data[['Property_Area', 'Credit_History']]

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, lr, "Logistic Regression")
X = data[['Credit_History','Education','Married','Self_Employed','Property_Area']]

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, lr, "Logistic Regression")
X = data[['Gender', 'Education', 'Credit_History', 'Self_Employed', 'Married', 'ApplicantIncome', 'Property_Area']]

dt = DecisionTreeClassifier()

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, dt, "Decision Tree")
X = data[['Property_Area', 'Credit_History']]

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, dt, "Decision Tree")
X = data[['Credit_History']]

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, dt, "Decision Tree")
X = data[['Gender', 'Education', 'Credit_History', 'Self_Employed', 'Married', 'ApplicantIncome', 'Property_Area']]

rfc = RandomForestClassifier()

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, rfc, "Random Forest")
X = data[['Property_Area', 'Credit_History']]

rfc = RandomForestClassifier()

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, rfc, "Random Forest")
X = data[['Credit_History']]

rfc = RandomForestClassifier()

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, rfc, "Random Forest")
sgd = SGDClassifier()

X = data[['Gender', 'Education', 'Credit_History', 'Self_Employed', 'Married', 'ApplicantIncome', 'Property_Area']]

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, sgd, "Stochastic Gradient Descent")
X = data[['Property_Area', 'Credit_History']]

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, sgd, "Stochastic Gradient Descent")
X = data[['Credit_History']]

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, sgd, "Stochastic Gradient Descent")
X = data[['ApplicantIncome']]

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, sgd, "Stochastic Gradient Descent")
X = data[['Self_Employed']]

print("With the predictors:", X.columns)

print()

Custom_Model(X, y, sgd, "Stochastic Gradient Descent")