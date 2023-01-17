# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

train = pd.read_csv('../input/bank-loan2/madfhantr.csv')

test = pd.read_csv('../input/bank-loan2/madhante.csv')

train

test
train.describe()
plt.figure(figsize=(7,6))

sns.heatmap(train.isnull())
plt.figure(figsize=(7,6))

sns.heatmap(test.isnull())
train.isnull().sum()
test.isnull().sum()
def impute_nan(variable,df):

    most_frequent_category = df[variable].value_counts().index[0]

    df[variable].fillna(most_frequent_category,inplace=True)

       

    



    
train_missing_categorical = ['Gender','Married','Dependents','Self_Employed','Credit_History']

for variable in train_missing_categorical:

    impute_nan(variable,train)
test_missing_categorical = ['Gender','Dependents','Self_Employed','Credit_History']

for variable in test_missing_categorical:

    impute_nan(variable,test)
train_cont_missing = ['LoanAmount','Loan_Amount_Term']

for variable in train_cont_missing:

    train[variable].fillna(train[variable].median(),inplace=True)
test_cont_missing = ['LoanAmount','Loan_Amount_Term']

for variable in test_cont_missing:

    test[variable].fillna(train[variable].median(),inplace=True)
sns.heatmap(train.isnull())
sns.heatmap(test.isnull())
sns.pairplot(train,palette = 'coolwarm')
sns.boxplot(x='Gender',y='LoanAmount',data = train)
sns.boxplot(x='Gender',y='LoanAmount',hue='Married',data = train,palette = 'coolwarm')
gender = pd.get_dummies(train['Gender'],drop_first=True)
test_gender = pd.get_dummies(test['Gender'],drop_first=True)
married =pd.get_dummies(train['Married'],drop_first=True)

married.rename(columns={'Yes':'Married'},inplace=True)
test_married= pd.get_dummies(test['Married'],drop_first=True)

test_married.rename(columns={'Yes':'Married'},inplace=True)
credit_hist = pd.get_dummies(train['Credit_History'],drop_first=True)

credit_hist.rename(columns={1.0:'Credit_History_1'},inplace=True)
test_credit_hist = pd.get_dummies(test['Credit_History'],drop_first=True)

test_credit_hist.rename(columns={1.0:'Credit_History_1'},inplace=True)
dependents = pd.get_dummies(train['Dependents'],drop_first=True)
test_dependents = pd.get_dummies(test['Dependents'],drop_first=True)
edu = pd.get_dummies(train['Education'],drop_first=True)
test_edu = pd.get_dummies(test['Education'],drop_first=True)
self_emp = pd.get_dummies(train['Self_Employed'],drop_first=True)
test_self_emp = pd.get_dummies(test['Self_Employed'],drop_first=True)
prop_ar = pd.get_dummies(train['Property_Area'],drop_first=True)

test_prop_ar = pd.get_dummies(test['Property_Area'],drop_first=True)
print(train.shape)

print('\n')

print(test.shape)
train_final = pd.concat([train,gender,married,credit_hist,dependents,edu,self_emp,prop_ar],axis=1)

test_final =  pd.concat([test,test_gender,test_married,test_credit_hist,test_dependents,test_edu,test_self_emp,test_prop_ar],axis=1)
train_final.columns
train_final.drop(['Loan_ID','Gender', 'Married', 'Dependents', 'Education','Self_Employed','Credit_History', 'Property_Area'],axis=1,inplace=True)

test_final.drop(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education','Self_Employed','Credit_History', 'Property_Area'],axis=1,inplace=True)
train_final.head(5)
test.tail()
X = train_final.drop('Loan_Status',axis=1)

y = train_final['Loan_Status']
from sklearn.model_selection import train_test_split


## Hyperparameter optimization using RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import xgboost
## Hyper Parameter Optimization



params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

    

}
classifier = xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(X,y)
random_search.best_estimator_
classifier = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.5, gamma=0.2, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.2, max_delta_step=0, max_depth=5,

              min_child_weight=7, monotone_constraints='()',

              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

              tree_method='exact', validate_parameters=1, verbosity=None)
classifier.fit(X,y)


from sklearn.model_selection import cross_val_score

score=cross_val_score(classifier,X,y,cv=10)

score
score.mean()
from sklearn.metrics import accuracy_score
result = classifier.predict(test_final)
result
submission = pd.DataFrame()

submission['Loan_ID'] = test.Loan_ID
submission['Loan_Status'] = result

submission.Loan_Status.value_counts()
