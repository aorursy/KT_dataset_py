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
import numpy as np

import pandas as pd

import pandas_profiling

from sklearn.linear_model import LogisticRegressionCV, SGDClassifier, LogisticRegression

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix
raw_csv_data = pd.read_csv("../input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv")
raw_csv_data.profile_report()
raw_csv_data['Gender'] = raw_csv_data['Gender'].map({'Male':1,'Female':0})

raw_csv_data['Married'] = raw_csv_data['Married'].map({'Yes':1,'No':0})

raw_csv_data['Education'] = raw_csv_data['Education'].map({'Graduate':1,'Not Graduate':0})

raw_csv_data['Self_Employed'] = raw_csv_data['Self_Employed'].map({'Yes':1,'No':0})

raw_csv_data['Property_Area'] = raw_csv_data['Property_Area'].map({'Urban':1,'Rural':2,'Semiurban':3})

raw_csv_data['Loan_Status'] = raw_csv_data['Loan_Status'].map({'Y':1,'N':0})
raw_csv_data
#Keeping a checkpoint

pre_process_data = raw_csv_data
##Insert new columns for dependents

pre_process_data.insert(0, "Dependents_1", 0) 

pre_process_data.insert(1, "Dependents_2", 0) 

pre_process_data.insert(2, "Dependents_3", 0) 

pre_process_data.insert(3, "Dependents_4", 0) 
#Fill all the values for newly created Dependent Column.

for ind in pre_process_data.index: 

     if(pre_process_data['Dependents'][ind] == 0):

        pre_process_data['Dependents_1'][ind] = 1

     elif(pre_process_data['Dependents'][ind] == 1):

        pre_process_data['Dependents_2'][ind] = 1

     elif(pre_process_data['Dependents'][ind] == 2):

        pre_process_data['Dependents_3'][ind] = 1

     elif(pre_process_data['Dependents'][ind] == '3+'):

        pre_process_data['Dependents_4'][ind] = 1
##Drop the Loan Id and the Dependent Column.

pre_process_data = pre_process_data.drop(['Dependents'], axis=1)

pre_process_data = pre_process_data.drop(['Loan_ID'], axis=1)
##Get Dummies

dummies= pd.get_dummies(pre_process_data, drop_first=True)



# We will now impute values

SimImp = SimpleImputer()

pre_process_data = pd.DataFrame(SimImp.fit_transform(dummies), columns=dummies.columns)
##All the missing values have been filled.

pre_process_data.info()
unscaled_inputs=pre_process_data.iloc[:,:-1]



##Standardize the input

from sklearn.preprocessing import StandardScaler

loan_scaler = StandardScaler()

loan_scaler.fit(unscaled_inputs)
scaled_inputs = loan_scaler.transform(unscaled_inputs)

scaled_inputs
##Splitting the Target

targets = pre_process_data['Loan_Status'] 
##Split the data into train test and shuffle

from sklearn.model_selection import train_test_split

train_test_split(scaled_inputs,targets)
 ##Logistic Regression with SKlearn

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



x_train, x_test, y_train, y_test = train_test_split(scaled_inputs,targets, train_size=0.8)



##Training the Model

reg=LogisticRegression()

reg.fit(x_train,y_train)
reg.score(x_train,y_train)
##Testing the Model

reg.score(x_test,y_test)