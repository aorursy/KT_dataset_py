import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Show the Dataset Path to get detaset



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#  Create New Variable and stores the dataset values as Data Frame



loan_train = pd.read_csv('/kaggle/input/loan-eligible-dataset/loan-train.csv')

loan_test = pd.read_csv('/kaggle/input/loan-eligible-dataset/loan-test.csv')
loan_train.head()
loan_train
print("Rows: ", len(loan_train))
print("Columns: ", len(loan_train.columns))
print("Shape : ", loan_train.shape)
loan_train_columns = loan_train.columns # assign to a variable

loan_train_columns # print the list of columns
loan_train.describe()
loan_train.info()

 
def explore_object_type(df ,feature_name):

    """

    To know, How many values available in object('categorical') type of features

    And Return Categorical values with Count.

    """    

    if df[feature_name].dtype ==  'object':

        print(df[feature_name].value_counts())
# Now, Test and Call a function for gender only

explore_object_type(loan_train, 'Gender')
# Solution is, Do you remember we have variable with name of `loan_train_columns`, Right,  let's use it

# 'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status'



for featureName in loan_train_columns:

    if loan_train[featureName].dtype == 'object':

        print('\n"' + str(featureName) + '\'s" Values with count are :')

        explore_object_type(loan_train, str(featureName))
import missingno as msno
# list of how many percentage values are missing

loan_train



loan_train.isna().sum()

# round((loan_train.isna().sum() / len(loan_train)) * 100, 2)
msno.bar(loan_train)
msno.matrix(loan_train )

loan_train['Credit_History'].fillna(loan_train['Credit_History'].mode(), inplace=True) # Mode

loan_test['Credit_History'].fillna(loan_test['Credit_History'].mode(), inplace=True) # Mode





loan_train['LoanAmount'].fillna(loan_train['LoanAmount'].mean(), inplace=True) # Mean

loan_test['LoanAmount'].fillna(loan_test['LoanAmount'].mean(), inplace=True) # Mean
loan_train.Loan_Status = loan_train.Loan_Status.replace({"Y": 1, "N" : 0})

# loan_test.Loan_Status = loan_test.Loan_Status.replace({"Y": 1, "N" : 0}) 



loan_train.Gender = loan_train.Gender.replace({"Male": 1, "Female" : 0})

loan_test.Gender = loan_test.Gender.replace({"Male": 1, "Female" : 0})



loan_train.Married = loan_train.Married.replace({"Yes": 1, "No" : 0})

loan_test.Married = loan_test.Married.replace({"Yes": 1, "No" : 0})



loan_train.Self_Employed = loan_train.Self_Employed.replace({"Yes": 1, "No" : 0})

loan_test.Self_Employed = loan_test.Self_Employed.replace({"Yes": 1, "No" : 0})



loan_train['Gender'].fillna(loan_train['Gender'].mode()[0], inplace=True)

loan_test['Gender'].fillna(loan_test['Gender'].mode()[0], inplace=True)



loan_train['Dependents'].fillna(loan_train['Dependents'].mode()[0], inplace=True)

loan_test['Dependents'].fillna(loan_test['Dependents'].mode()[0], inplace=True)



loan_train['Married'].fillna(loan_train['Married'].mode()[0], inplace=True)

loan_test['Married'].fillna(loan_test['Married'].mode()[0], inplace=True)



loan_train['Credit_History'].fillna(loan_train['Credit_History'].mean(), inplace=True)

loan_test['Credit_History'].fillna(loan_test['Credit_History'].mean(), inplace=True)

from sklearn.preprocessing import LabelEncoder

feature_col = ['Property_Area','Education', 'Dependents']

le = LabelEncoder()

for col in feature_col:

    loan_train[col] = le.fit_transform(loan_train[col])

    loan_test[col] = le.fit_transform(loan_test[col])
import matplotlib.pyplot as plt

%matplotlib inline





import seaborn as sns

sns.set_style('dark')

loan_train
loan_train.plot(figsize=(18, 8))



plt.show()
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)





loan_train['ApplicantIncome'].hist(bins=10)

plt.title("Loan Application Amount ")



plt.subplot(1, 2, 2)

plt.grid()

plt.hist(np.log(loan_train['LoanAmount']))

plt.title("Log Loan Application Amount ")



plt.show()
plt.figure(figsize=(18, 6))

plt.title("Relation Between Applicatoin Income vs Loan Amount ")



plt.grid()

plt.scatter(loan_train['ApplicantIncome'] , loan_train['LoanAmount'], c='k', marker='x')

plt.xlabel("Applicant Income")

plt.ylabel("Loan Amount")

plt.show()
plt.figure(figsize=(12, 6))

plt.plot(loan_train['Loan_Status'], loan_train['LoanAmount'])

plt.title("Loan Application Amount ")

plt.show()
plt.figure(figsize=(12,8))

sns.heatmap(loan_train.corr(), cmap='coolwarm', annot=True, fmt='.1f', linewidths=.1)

plt.show()
# import ml model from sklearn pacakge



from sklearn.linear_model import LogisticRegression



from sklearn.metrics import accuracy_score

logistic_model = LogisticRegression()
train_features = ['Credit_History', 'Education', 'Gender']



x_train = loan_train[train_features].values

y_train = loan_train['Loan_Status'].values



x_test = loan_test[train_features].values

logistic_model.fit(x_train, y_train)
# Predict the model for testin data



predicted = logistic_model.predict(x_test)

# check the coefficeints of the trained model

print('Coefficient of model :', logistic_model.coef_)

# check the intercept of the model

print('Intercept of model',logistic_model.intercept_)
# Accuray Score on train dataset

# accuracy_train = accuracy_score(x_test, predicted)

score = logistic_model.score(x_train, y_train)

print('accuracy_score overall :', score)

print('accuracy_score percent :', round(score*100,2))



# predict the target on the test dataset

predict_test = logistic_model.predict(x_test)

print('Target on test data',predict_test) 

import pickle as pkl
# save the model to disk

filename = 'logistic_model.pkl'

pkl.dump(logistic_model, open(filename, 'wb')) # wb means write as binary




