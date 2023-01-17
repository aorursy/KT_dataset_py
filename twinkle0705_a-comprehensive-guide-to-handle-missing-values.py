import numpy as np

import pandas as pd 

import seaborn as sns
data = pd.read_csv('../input/loan-prediction/train.csv')

cat_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.head()
data.info()
import missingno as msno

msno.matrix(data)
msno.bar(data, color = 'y', figsize = (10,8))
msno.heatmap(data)
ax = msno.dendrogram(data)
data.describe()
data.isnull()
total = data.isnull().sum().sort_values(ascending=False)

percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
data.notnull().sum().sort_values(ascending=False)
data.dropna(subset = ['Loan_Amount_Term'], axis = 0, how = 'any', inplace = True)
# data.drop(['column_name'], axis = 1, inplace = True)
columns = ['LoanAmount','ApplicantIncome','CoapplicantIncome']

sns.pairplot(data[columns])
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace = True)
data['Dependents'].fillna((data['Dependents'].mode()[0]),inplace=True)
data['Gender'].value_counts()
data['Gender'].fillna('Male', inplace = True)
data['Gender'].fillna(data['Gender'].value_counts().index[0], inplace = True)
data['Self_Employed'].fillna(method='ffill',inplace=True)
cat_data.head()
total = cat_data.isnull().sum().sort_values(ascending=False)

percent = ((cat_data.isnull().sum()/cat_data.isnull().count())*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
cat_data['PoolQC'].fillna("none", inplace = True)