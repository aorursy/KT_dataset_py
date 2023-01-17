# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import statsmodels.formula.api as smf

from imblearn.over_sampling import SMOTE
data_train = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(list(data_train.columns))

print(data_train.shape)
cleanup_nums = {"Attrition":     {"Yes": 1, "No": 0}}

data_train.replace(cleanup_nums, inplace=True)

data_train.head()
cat_vars=['BusinessTravel','Department', 'EducationField', 'Gender','MaritalStatus','JobRole','OverTime']

for var in cat_vars:

    cat_list='var'+'_'+var

    cat_list = pd.get_dummies(data_train[var], prefix=var)

    data1=data_train.join(cat_list)

    data_train=data1

cat_vars=['BusinessTravel','Department', 'EducationField', 'Gender','MaritalStatus','JobRole','OverTime']

data_vars=data_train.columns.values.tolist()

to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data_train[to_keep]

data_final.columns.values
x = data_final.drop(['Age','Over18','EmployeeCount','StandardHours'],axis =1)
X = x.loc[:, x.columns != 'Attrition']

print(list(X))

y = x.loc[:, x.columns == 'Attrition']

os = SMOTE(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

columns = X_train.columns

os_data_X, os_data_y=os.fit_sample(X_train, y_train)

os_data_X = pd.DataFrame(data=os_data_X,columns=columns)

os_data_y= pd.DataFrame(data=os_data_y,columns=['Attrition'])

# we can Check the numbers of our data

print("length of oversampled data is ",len(os_data_X))

print("Number of retained in oversampled data",len(os_data_y[os_data_y['Attrition']==0]))

print("Number of left",len(os_data_y[os_data_y['Attrition']==1]))

print("Proportion of retained in oversampled data is ",len(os_data_y[os_data_y['Attrition']==0])/len(os_data_X))

print("Proportion of left in oversampled data is ",len(os_data_y[os_data_y['Attrition']==1])/len(os_data_X))
cols=['DailyRate', 'DistanceFromHome', 'Education', 'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely', 'Department_Human Resources', 'Department_Research & Development', 'Department_Sales', 'EducationField_Human Resources', 'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree', 'Gender_Female', 'Gender_Male', 'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single', 'JobRole_Healthcare Representative', 'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative', 'OverTime_No', 'OverTime_Yes'] 

X=os_data_X[cols]

y=os_data_y['Attrition']

logit_model=smf.Logit(y,X)

result=logit_model.fit()

print(result.summary2())
cols=['DistanceFromHome', 'EnvironmentSatisfaction', 'HourlyRate', 'JobSatisfaction', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently', 'Department_Research & Development', 'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree', 'Gender_Male', 'Gender_Female','MaritalStatus_Divorced', 'MaritalStatus_Married', 'JobRole_Healthcare Representative', 'JobRole_Laboratory Technician', 'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist', 'OverTime_No'] 

X=os_data_X[cols]

y=os_data_y['Attrition']

logit_model=smf.Logit(y,X)

result=logit_model.fit()

print(result.summary2())
cols=['DistanceFromHome', 'EnvironmentSatisfaction', 'HourlyRate', 'JobSatisfaction', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently', 'Department_Research & Development', 'EducationField_Life Sciences', 'EducationField_Medical', 'EducationField_Other', 'Gender_Male', 'Gender_Female', 'MaritalStatus_Married', 'JobRole_Healthcare Representative', 'JobRole_Laboratory Technician', 'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist', 'OverTime_No'] 

X=os_data_X[cols]

y=os_data_y['Attrition']

logit_model=smf.Logit(y,X)

result=logit_model.fit()

print(result.summary2())
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)


from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()