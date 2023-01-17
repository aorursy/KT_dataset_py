# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/train_data.csv',skiprows=1,names =["Loan_ID","Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area"])

df2 = pd.read_csv('../input/train_prediction.csv',skiprows=1, names=["Loan_ID","Loan_Status"])
df = pd.merge(df1, df2, on='Loan_ID', how='outer')
df['Gender'] = df['Gender'].fillna( df['Gender'].dropna().mode().values[0] )

df['Married'] = df['Married'].fillna( df['Married'].dropna().mode().values[0] )

df['Dependents'] = df['Dependents'].fillna( df['Dependents'].dropna().mode().values[0] )

df['Self_Employed'] = df['Self_Employed'].fillna( df['Self_Employed'].dropna().mode().values[0] )

df['LoanAmount'] = df['LoanAmount'].fillna( df['LoanAmount'].dropna().mean() )

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna( df['Loan_Amount_Term'].dropna().mode().values[0] )

df['Credit_History'] = df['Credit_History'].fillna( df['Credit_History'].dropna().mode().values[0] )

df['Dependents'] = df['Dependents'].str.rstrip('+')
df['Gender'] = df['Gender'].map({'F':0,'M':1}).astype(np.int)

df['Married'] = df['Married'].map({'No':0, 'Yes':1}).astype(np.int)

df['Education'] = df['Education'].map({'Not Graduate':0, 'Graduate':1}).astype(np.int)

df['Self_Employed'] = df['Self_Employed'].map({'No':0, 'Yes':1}).astype(np.int)

df['Loan_Status'] = df['Loan_Status'].map({'N':0, 'Y':1}).astype(np.int)

df['Dependents'] = df['Dependents'].astype(np.int)
X,y  = df.iloc[:, 1:-1], df.iloc[:, -1]
X= pd.get_dummies(X)



from sklearn.preprocessing import StandardScaler

slc= StandardScaler()

X_train_std = slc.fit_transform(X)
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators =500, criterion='entropy', oob_score=True, random_state=1,n_jobs=-1)
from xgboost.sklearn import XGBClassifier

xgb1 = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=8, min_child_weight=6, gamma=0.1, subsample=0.95,

                     colsample_bytree=0.95, reg_alpha=2, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(criterion='entropy',max_depth=1)

ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=0)
from sklearn.ensemble import VotingClassifier



eclf = VotingClassifier(estimators=[('forest', forest), ('xgb', xgb1), ('adaboost', ada)], voting='hard')

eclf.fit(X_train_std, y)
dtest = pd.read_csv('../input/test_data.csv',skiprows=1,names =["Loan_ID","Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area"])
dtest['Gender'] = dtest['Gender'].fillna( dtest['Gender'].dropna().mode().values[0])

dtest['Dependents'] = dtest['Dependents'].fillna( dtest['Dependents'].dropna().mode().values[0])

dtest['Self_Employed'] = dtest['Self_Employed'].fillna( dtest['Self_Employed'].dropna().mode().values[0])

dtest['LoanAmount'] = dtest['LoanAmount'].fillna( dtest['LoanAmount'].dropna().mode().values[0])

dtest['Loan_Amount_Term'] = dtest['Loan_Amount_Term'].fillna( dtest['Loan_Amount_Term'].dropna().mode().values[0])

dtest['Credit_History'] = dtest['Credit_History'].fillna( dtest['Credit_History'].dropna().mode().values[0] )
dtest['Gender'] = dtest['Gender'].map({'F':0,'M':1})

dtest['Married'] = dtest['Married'].fillna( dtest['Married'].dropna().mode().values[0] )

dtest['Married'] = dtest['Married'].map({'No':0, 'Yes':1}).astype(np.int)

dtest['Education'] = dtest['Education'].map({'Not Graduate':0, 'Graduate':1}).astype(np.int)

dtest['Self_Employed'] = dtest['Self_Employed'].map({'No':0, 'Yes':1})

dtest['Dependents'] = dtest['Dependents'].str.rstrip('+')
dtest['Gender'] = dtest['Gender'].fillna( dtest['Gender'].dropna().mode().values[0]).astype(np.int)

dtest['Dependents'] = dtest['Dependents'].fillna( dtest['Dependents'].dropna().mode().values[0]).astype(np.int)

dtest['Self_Employed'] = dtest['Self_Employed'].fillna( dtest['Self_Employed'].dropna().mode().values[0])

dtest['LoanAmount'] = dtest['LoanAmount'].fillna( dtest['LoanAmount'].dropna().mode().values[0])

dtest['Loan_Amount_Term'] = dtest['Loan_Amount_Term'].fillna( dtest['Loan_Amount_Term'].dropna().mode().values[0])

dtest['Credit_History'] = dtest['Credit_History'].fillna( dtest['Credit_History'].dropna().mode().values[0] )

X_test = dtest.iloc[:,1:]

X_test= pd.get_dummies(X_test)

X_test_std = slc.transform(X_test)

y_test_pred = eclf.predict(X_test_std)

dtest['Loan_Status'] = y_test_pred

df_final = dtest.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'], axis=1)

df_final['Loan_Status'] = df_final['Loan_Status'].map({0:'N', 1:'Y'})

df_final.columns = ['loan_id','status']

df_final.to_csv('my_submission.csv', index=False)