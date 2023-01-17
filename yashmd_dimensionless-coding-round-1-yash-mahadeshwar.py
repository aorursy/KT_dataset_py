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
train = pd.read_csv('/kaggle/input/credit-risk-modeling-case-study/CRM_TrainData.csv')

test = pd.read_csv('/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv')
train.head()
train.shape
train.isna().sum()/train.shape[0]

test.isna().sum()/train.shape[0]
def cleanTheData(pd_original_data):

    pd_data_Cleaning=pd_original_data

    #Cleaining Current Loan Amount

    convert_dict = {'Current Loan Amount': float} 

    pd_data_Cleaning=pd_data_Cleaning.astype(convert_dict)

    #Credit Score Cleaining

    pd_data_Cleaning['Credit Score'].fillna(pd_data_Cleaning['Credit Score'].mean(),inplace=True)

    #Annual Income

    pd_data_Cleaning['Annual Income'].fillna(pd_data_Cleaning['Annual Income'].median(),inplace=True)

    #Month Since Last Delinquent

    pd_data_Cleaning["Months since last delinquent"].fillna("0",inplace=True)

    convert_dict = {'Months since last delinquent': int} 

    pd_data_Cleaning=pd_data_Cleaning.astype(convert_dict)

    #Cleaining Years in Current Job

    mode=pd_data_Cleaning['Years in current job'].mode()

    pd_data_Cleaning['Years in current job'].replace('[^0-9]',"",inplace=True,regex=True)

    pd_data_Cleaning['Years in current job']=pd_data_Cleaning['Years in current job'].fillna(10)

    convert_dict = {'Years in current job': int} 

    pd_data_Cleaning= pd_data_Cleaning.astype(convert_dict)

    #Maximum Open Credit Cleaning

    pd_data_Cleaning["Maximum Open Credit"].replace('[a-zA-Z@_!#$%^&*()<>?/\|}{~:]',"0",regex=True,inplace=True)

    convert_dict = {'Maximum Open Credit': float} 

    pd_data_Cleaning=pd_data_Cleaning.astype(convert_dict)

    #BankRuptcies cleaining

    pd_data_Cleaning[pd_data_Cleaning.Bankruptcies.isna()==True]

    pd_data_Cleaning.Bankruptcies.fillna(0.0,inplace=True)

    convert_dict = {'Bankruptcies': int} 

    pd_data_Cleaning=pd_data_Cleaning.astype(convert_dict)

    #Tax Liens Cleaning

    pd_data_Cleaning['Tax Liens'].fillna(0.0,inplace=True)

    convert_dict = {'Tax Liens': int} 

    pd_data_Cleaning=pd_data_Cleaning.astype(convert_dict)

    #Monthly Debt Cleaning

    convert_dict = {'Monthly Debt': float} 

    pd_data_Cleaning["Monthly Debt"].replace('[^0-9.]',"",regex=True,inplace=True )

    pd_data_Cleaning["Monthly Debt"]=pd_data_Cleaning["Monthly Debt"].astype(convert_dict)

    

    return pd_data_Cleaning

train_df = cleanTheData(train)

test_df = cleanTheData(test)
train_df.isna().sum()/train_df.shape[0]
test_df.isna().sum()/test_df.shape[0]
train_df.dtypes
train_df.head()
train_df.select_dtypes("object_")

test_df.select_dtypes("object_")
#Let's join the datasets and drop columns that are not required

df = pd.concat([test_df,train_df])
df['Loan Status'].isna().sum()

df.drop(['Unnamed: 2'], axis = 1,inplace = True) 
df.select_dtypes("object_")
df['Home Ownership'].value_counts()
df["Home Ownership"].replace({"HaveMortgage": "Mortgage", "Home Mortgage": "Mortgage"}, inplace=True)
df['Home Ownership'].value_counts()
df['Purpose'].value_counts()
df["Purpose"].replace({"other": "others", "Other": "others"}, inplace=True)
df['Purpose'].value_counts()
df['Term'].value_counts()
df['Loan Status'].value_counts()[1]/df['Loan Status'].value_counts()[0]

#Not a major class imbalance problem since the minority dataset has 30% of data
float_columns = df.select_dtypes("float64").columns
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
df[float_columns] = pd.DataFrame(scale.fit_transform( df[float_columns]),columns =  float_columns)

float_columns

#Haven't scaled integer columns since we can consider them as ordinal variables
y = df['Loan Status']

Loan_id = df['Loan ID']

Customer_ID = df['Customer ID']

df.drop(['Loan Status','Loan ID','Customer ID'],1,inplace = True)

df.columns
#Using one hot encoding for object data types

df = pd.get_dummies(df,drop_first=True)
df['Target'] = y
df.head()

df['Loan ID'] = Loan_id


test_prep = df.loc[df['Target'].isnull()]

train_prep = df.loc[df['Target'].notna()]

(test_prep.shape[0]+train_prep.shape[0]) == (test.shape[0]+train.shape[0])
test_prep.drop('Target',1,inplace=True)
#So our dataset to fit in the model is read
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(train_prep.drop('Target',1),train_prep['Target'],random_state = 42)
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier(n_jobs = -1)

svc = SVC()

'''params_rf = {'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40,],

 'max_features': ['auto'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators':range(10,100,10)}

Cs = [0.1, 1, 5]

gammas = [0.01, 0.1, 1]

kernels = ['rbf', 'poly']

    

params_svc = {'C': Cs, 'gamma' : gammas,'kernel':kernels} '''
rf.fit(x_train,y_train)
rf.score(x_test,y_test)
#test_loan_id = test_prep['Loan ID']

test_prep.drop(['Target'],1,inplace = True)
preds = rf.predict(test_prep)
sub_csv = pd.DataFrame({'Loan ID':test_loan_id,'Loan Status':preds})
sub_csv.to_csv('sub3.csv')