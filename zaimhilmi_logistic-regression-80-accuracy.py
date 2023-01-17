# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
telco_df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# Any results you write to the current directory are saved as output.
telco_df.head()
plt.figure(figsize=(15,5))
sns.set_style('whitegrid')
sns.distplot(telco_df['tenure'],bins=30,kde=False,color='red')
plt.figure(figsize=(15,5))
sns.set_style('whitegrid')
sns.distplot(telco_df['MonthlyCharges'],bins=30,kde=False,color='blue')
g = sns.FacetGrid(telco_df, height = 5, col="gender",  row="SeniorCitizen",hue='Churn')
g = g.map(plt.scatter, "MonthlyCharges", "tenure").add_legend()
#cleaning-up data for logistic regression
telco_df['TotalCharges'] = pd.to_numeric(telco_df['TotalCharges'], errors='coerce')
telco_df.dropna(inplace=True)

#solving collinearity problem
gender = pd.get_dummies(telco_df['gender'],drop_first=True)
Partner = pd.get_dummies(telco_df['Partner'],drop_first=True)
Dependents = pd.get_dummies(telco_df['Dependents'],drop_first=True)
PhoneService = pd.get_dummies(telco_df['PhoneService'],drop_first=True)
InternetService = pd.get_dummies(telco_df['InternetService'],drop_first=True)
Contract = pd.get_dummies(telco_df['Contract'],drop_first=True)
PaperlessBilling = pd.get_dummies(telco_df['PaperlessBilling'],drop_first=True)
PaymentMethod = pd.get_dummies(telco_df['PaymentMethod'],drop_first=True)
Churn = pd.get_dummies(telco_df['Churn'],drop_first=True)

Partner.rename(columns={'Yes': 'Partner'}, inplace=True)
Dependents.rename(columns={'Yes': 'Dependents'},inplace=True)
PhoneService.rename(columns={'Yes': 'PhoneService'},inplace=True)
telco_df.drop('MultipleLines',axis=1,inplace=True)
InternetService.rename(columns={'Fiber Optic': 'Fiber Optic','No': 'No Fiber Optic'},inplace=True)
telco_df.drop('OnlineSecurity',axis=1,inplace=True)
telco_df.drop('DeviceProtection',axis=1,inplace=True)
telco_df.drop('TechSupport',axis=1,inplace=True)
telco_df.drop('StreamingTV',axis=1,inplace=True)
PaperlessBilling.rename(columns={'Yes': 'PaperlessBilling'},inplace=True)
Churn.rename(columns={'Yes': 'Churn'},inplace=True)

telco_df.drop('gender',axis=1,inplace=True)
telco_df.drop('OnlineBackup',axis=1,inplace=True)
telco_df.drop('StreamingMovies',axis=1,inplace=True)
telco_df.drop('Contract',axis=1,inplace=True)
telco_df.drop('PaymentMethod',axis=1,inplace=True)

telco_df.drop('SeniorCitizen',axis=1,inplace=True)
telco_df.drop('Partner',axis=1,inplace=True)
telco_df.drop('Dependents',axis=1,inplace=True)
telco_df.drop('PhoneService',axis=1,inplace=True)
telco_df.drop('InternetService',axis=1,inplace=True)
telco_df.drop('PaperlessBilling',axis=1,inplace=True)
telco_df.drop('Churn',axis=1,inplace=True)
telco_df.drop('customerID',axis=1,inplace=True)
df = pd.concat([telco_df,gender,Partner,Dependents,PhoneService,
                InternetService,Contract,PaperlessBilling,PaymentMethod,Churn],axis=1)
df.head()
X = df.drop('Churn',axis=1)
y = df['Churn']
#running Logistic Regression analysis
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
#Fitting training datas
logmodel.fit(X_train,y_train)
#Predicting new data
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))
