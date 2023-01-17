import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("rainbow")
sns.set_style('whitegrid')
%matplotlib inline
import missingno as msno

import warnings
warnings.filterwarnings('ignore')

from sklearn import model_selection
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
df = pd.read_csv("../input/train.csv")
df.head()
df.info()
df['Dependents'].unique()
msno.matrix(df)
missing_values = df.isnull().sum().sort_values(ascending = False)
missing_values
df['Dependents'].replace(to_replace = '3+', value = 3, inplace = True)
df['Dependents'].fillna(df['Dependents'].median(), inplace = True)
df['Dependents'] = df['Dependents'].astype(int)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace = True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace = True)
for column in ['Credit_History', 'Self_Employed', 'Gender', 'Married']:
    df[column].fillna(df[column].ffill(), inplace=True)
sns.boxplot('Loan_Status','LoanAmount', data = df)
sns.boxplot('Loan_Status','CoapplicantIncome', data = df)
sns.boxplot('Loan_Status','ApplicantIncome', data = df)
sns.catplot(x="Loan_Status", y="LoanAmount", hue="Gender", kind="bar", data=df);
sns.catplot(x="Loan_Status", y="LoanAmount", hue="Education", kind= "bar" , data=df)
df.drop(df[df['CoapplicantIncome']>10000].index, inplace= True)
df.drop(df[df['ApplicantIncome']>20000].index, inplace= True)
df.drop(df[df['LoanAmount']>550].index, inplace= True)
df['total_amount'] = df['ApplicantIncome'] +df['CoapplicantIncome']
df['ratio']= df['total_amount']/df['LoanAmount']
sns.distplot(df['ApplicantIncome'])
df.describe()
df.skew(axis=0).sort_values(ascending= False)
cols = ['ratio','LoanAmount','total_amount']
df[cols] = np.log1p(df[cols])
df.drop('Loan_ID', axis = 1, inplace = True)
df.drop('ApplicantIncome', axis = 1, inplace = True)
df.drop('CoapplicantIncome', axis = 1, inplace = True)
df1 = pd.get_dummies(df, columns=list(df.select_dtypes(exclude=np.number)))
df1.head()
col= ['Loan_Status_N','Loan_Status_Y']
df1.drop(columns = col,axis = 1,inplace = True)
df1.head()
df1['Loan_status'] = df["Loan_Status"]
X = df1.iloc[:,0:16]
y = df1.iloc[:,17]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rf = RandomForestClassifier()
m = rf.fit(X_train, y_train)
y_pred = m.predict(X_test)
accuracy_score(y_test, y_pred)
import sklearn.model_selection as ms
parameters= {'n_estimators':[5,50,500],
            'max_depth':[5,10,12,15],
            'max_features':[5,10,15],
            'min_samples_split' : [2, 5, 10],
            'min_samples_leaf' : [1, 2, 4]}


rf = RandomForestClassifier()
rf_model = ms.GridSearchCV(rf, param_grid=parameters, cv=5)
rf_model.fit(X_train,y_train)

print('The best value of Alpha is: ',rf_model.best_params_)
rf1 = RandomForestClassifier(n_estimators = 500,max_depth = 10, max_features=15, min_samples_leaf = 4, min_samples_split= 10,bootstrap = True)
rf_random = rf1.fit(X_train,y_train)

scores_rf = cross_val_score(rf_random, X_train, y_train, cv=5)
scores_rf.mean()
y_pred = rf_random.predict(X_test)
accuracy_score(y_test, y_pred)
from sklearn.model_selection import RandomizedSearchCV
random_grid ={'bootstrap': [True, False],
             'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
             'max_features': [1,3,5,7,9,11,13,15],
             'min_samples_leaf': [1, 2,3, 4],
             'min_samples_split': [2, 5, 10],
             'n_estimators': [50,100,500,1000]}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf = RandomForestClassifier(n_estimators = 500,max_depth = 10, max_features=5, min_samples_leaf = 3, min_samples_split= 10,bootstrap = True)
rf_random = rf.fit(X_train,y_train)

scores_rf = cross_val_score(rf_random, X_train, y_train, cv=5)
scores_rf.mean()
y_pred = rf_random.predict(X_test)
accuracy_score(y_test, y_pred)