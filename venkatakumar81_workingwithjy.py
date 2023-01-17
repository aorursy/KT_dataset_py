import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
os.chdir("C:\\Downloads\\WA_Fn-UseC_-HR-Employee-Attrition (2).csv")
HRemployee= pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition (2).csv", header =0)
HRemployee.head()
sns.distplot(HRemployee['Age'])
sns.distplot(HRemployee['DailyRate'])
plt.show()

sns.countplot(HRemployee.DailyRate)
sns.barplot(x ='Education', y='EmployeeCount', data=HRemployee)
sns.barplot(x = 'Department',y='EmployeeNumber',data = HRemployee)
sns.boxplot(HRemployee['WorkLifeBalance'], HRemployee['YearsInCurrentRole'])
HRemployee['WorkLifeBalance'] = pd.cut(HRemployee['YearsInCurrentRole'], 3, labels=['low', 'middle', 'high'])
HRemployee.head()
sns.jointplot(HRemployee.Education,HRemployee.EmployeeCount, kind = "scatter") 
sns.jointplot(HRemployee.WorkLifeBalance,HRemployee.YearsInCurrentRole,kind="scatter") 
sns.jointplot(HRemployee.YearsSinceLastPromotion,HRemployee.YearsWithCurrManager, kind = "reg")
sns.jointplot(HRemployee.YearsSinceLastPromotion,HRemployee.YearsInCurrentRole, kind = "hex") 
cont_col= ['c', 'Attrition','DailyRate','DistanceFromHome']
sns.pairplot(HRemployee[cont_col], kind="reg",diag_kind="kde",hue='Attrition')
plt.show()
sns.factorplot(x='Age',y='Attrition',hue='DailyRate',col='DistanceFromHome',col_wrap=2,kind='box',

               data= HRemployee)