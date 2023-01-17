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
import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns
trainset = pd.read_csv("/kaggle/input/summeranalytics2020/train.csv")

trainset = trainset.drop('EmployeeNumber', axis = 1)

trainset = trainset.set_index('Id')

df_n = trainset

f, ax = plt.subplots(figsize=(10, 8))

corr = df_n.corr()

sns.heatmap(corr,

            mask=np.zeros_like(corr, dtype=np.bool), 

            cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
xtemp = trainset.drop("Behaviour",axis = 1)

xtemp = xtemp.drop(['TotalWorkingYears','PercentSalaryHike',"YearsInCurrentRole","YearsWithCurrManager"],axis=1)

y=xtemp['Attrition']

xtemp = xtemp.drop('Attrition',axis = 1)
x = pd.get_dummies(xtemp)

x = x.drop(['Gender_Female','OverTime_No','BusinessTravel_Non-Travel','Department_Human Resources','EducationField_Human Resources','JobRole_Sales Executive','MaritalStatus_Single'],axis = 1)


testset = pd.read_csv("/kaggle/input/summeranalytics2020/test.csv")

testset = testset.drop('EmployeeNumber', axis = 1)

testset = testset.set_index('Id')

xtest = pd.get_dummies(testset)

xtest = xtest.drop(['TotalWorkingYears','PercentSalaryHike',"YearsInCurrentRole","YearsWithCurrManager",'Behaviour','Gender_Female','OverTime_No','BusinessTravel_Non-Travel','Department_Human Resources','EducationField_Human Resources','JobRole_Sales Executive','MaritalStatus_Single'],axis = 1)
from sklearn.model_selection import train_test_split

xtrain, xval,ytrain,yval = train_test_split(x,y,test_size= 0.3)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000,bootstrap=True)

rf = rf.fit(xtrain,ytrain)

yvalrf = rf.predict_proba(xval)

roc_auc_score(yval,yvalrf[:,1])
ytest = rf.predict_proba(xtest)[:,1]

ytest