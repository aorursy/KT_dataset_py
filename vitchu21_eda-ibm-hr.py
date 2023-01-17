#Importing Required Modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
#Importing Data

df=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head(2)
#Presence of Missing values

df[df.isnull()==True].count()
#Removing columns with constant values

df.head(5)
df=df.drop(columns=['EmployeeCount','Over18','StandardHours'])
df.head(2)
#Creating a new dataframe containing only the object columns.

obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()
# BusinessTravel vs Employee Attrition

bt_att=pd.crosstab(df.Attrition,df.BusinessTravel)
print('BusinessTravel vs Employee Attrition:\n',bt_att)
print('\n')

# Department vs Employee Attrition
dept_att=pd.crosstab(df.Attrition,df.Department)
print('Department vs Employee Attrition:\n',dept_att)
print('\n')

# EducationField vs Employee Attrition
ed_att=pd.crosstab(df.Attrition,df.EducationField)
print('EducationField vs Employee Attrition:\n',ed_att)
print('\n')

# Gender vs Employee Attrition
gen_att=pd.crosstab(df.Attrition,df.Gender)
print('Gender vs Employee Attrition:\n',gen_att)
print('\n')

# JobRole vs Employee Attrition
job_att=pd.crosstab(df.Attrition,df.JobRole)
print('JobRole vs Employee Attrition:\n',job_att)
print('\n')

# MaritalStatus vs Employee Attrition
ms_att=pd.crosstab(df.Attrition,df.MaritalStatus)
print('MaritalStatus vs Employee Attrition:\n',ms_att)
print('\n')

# OverTime vs Employee Attrition
ot_att=pd.crosstab(df.Attrition,df.OverTime)
print('OverTime vs Employee Attrition:\n',ot_att)
print('\n')
fig=plt.figure()

ax0=fig.add_subplot(3,3,1)
bt_att.plot(kind='bar',ax=ax0,figsize=(15,15))
plt.title('Employee Attrition vs Business Travel Frequency of Employee')
plt.xlabel('Employee Attrition')
plt.ylabel('Count of Employee')

ax1=fig.add_subplot(3,3,2)
dept_att.plot(kind='bar',ax=ax1,figsize=(15,15))
plt.title('Employee Attrition vs Department of Employee')
plt.xlabel('Employee Attrition')
plt.ylabel('Count of Employee')

ax2=fig.add_subplot(3,3,3)
ed_att.plot(kind='bar',ax=ax2,figsize=(15,15))
plt.title('Employee Attrition vs Education Field of Employee')
plt.xlabel('Employee Attrition')
plt.ylabel('Count of Employee')

ax3=fig.add_subplot(3,3,4)
gen_att.plot(kind='bar',ax=ax3,figsize=(15,15))
plt.title('Employee Attrition vs Employee Gender')
plt.xlabel('Employee Attrition')
plt.ylabel('Count of Employee')

ax4=fig.add_subplot(3,3,5)
job_att.plot(kind='bar',ax=ax4,figsize=(15,15))
plt.title('Employee Attrition vs Job Role of Employee')
plt.xlabel('Employee Attrition')
plt.ylabel('Count of Employee')

ax5=fig.add_subplot(3,3,6)
ms_att.plot(kind='bar',ax=ax5,figsize=(15,15))
plt.title('Employee Attrition vs Marital of Employee')
plt.xlabel('Employee Attrition')
plt.ylabel('Count of Employee')

ax6=fig.add_subplot(3,3,7)
ot_att.plot(kind='bar',ax=ax6,figsize=(15,15))
plt.title('Employee Attrition vs Employee Interest to work Overtime')
plt.xlabel('Employee Attrition')
plt.ylabel('Count of Employee')
plt.gcf().clear()

import scipy.stats
from scipy.stats import stats
obj_df.head(2)
#Chisquare test for investigating if employee attrition is influenced by the business travel taken by an employee 
print("Chi-square value for Business travel vs Attrition is:\n",scipy.stats.chi2_contingency(bt_att))

#Chisquare test for investigating if employee attrition is influenced by the department in which an employee works 
print("Chi-square value for Department vs Attrition is:\n",scipy.stats.chi2_contingency(dept_att))
#Chisquare test for investigating if employee attrition is influenced by the Education field of an employee
print("Chi-square value for Education field vs Attrition is:\n",scipy.stats.chi2_contingency(ed_att))
#Chisquare test for investigating if employee attrition is influenced by the Gender of an employee
print("Chi-square value for Gender vs Attrition is:\n",scipy.stats.chi2_contingency(gen_att))
#Chisquare test for investigating if employee attrition is influenced by the Job Role of an employee
print("Chi-square value for Job Role vs Attrition is:\n",scipy.stats.chi2_contingency(job_att))
#Chisquare test for investigating if employee attrition is influenced by the Marital Status of an employee
print("Chi-square value for Marital status vs Attrition is:\n",scipy.stats.chi2_contingency(ms_att))
#Chisquare test for investigating if employee attrition is influenced by the employee's interest to work over time
print("Chi-square value for Over time vs Attrition is:\n",scipy.stats.chi2_contingency(ot_att))
#Creating a new dataframe containing only the integer columns.

int_df = df.select_dtypes(include=['int64']).copy()
int_df.head()
#Checking for correlation between the continuous variables
int_df.corr()
#Generating a heat map
pd.scatter_matrix(int_df,figsize=(50,50),color='g')
plt.show()
pivot_attr=pd.pivot_table(df,index=["Attrition"])
pivot_attr
#Age of Employee vs Employee Attrition
# Point plot
sns.pointplot(x="Attrition", y="Age", data=df)
plt.title("Pointplot: Employee Attrition vs Age of Employee")
#Factor Plot
sns.factorplot("Attrition", "Age", data=df, kind="bar", size=3, aspect=2, palette="muted", legend=False)
plt.title("Factortplot: Employee Attrition vs Age of Employee")
#Swarm Plot
sns.swarmplot(x="Attrition", y="Age", data=df)
plt.title("Swarmtplot: Employee Attrition vs Age of Employee")
#Violin Plot
sns.violinplot(x="Attrition", y="Age", data=df,split=True,inner="stick")
plt.title("Swarmplot: Employee Attrition vs Age of Employee")
#Box Plot
sns.boxplot(x="Attrition", y="Age", data=df)
plt.title("Boxplot: Employee Attrition vs Age of Employee")
#Columns that have Numerical values
int_df.columns
sns.set(rc={'figure.figsize':(3,3)})
#YearsAtCompany of Employee vs Employee Attrition
sns.boxplot(x="Attrition", y="YearsAtCompany", data=df, width=0.3)
plt.title("Boxplot: Employee Attrition vs YearsAtCompany of Employee")
#NumCompaniesWorked of Employee vs Employee Attrition
sns.boxplot(x="Attrition", y="NumCompaniesWorked", data=df, width=0.3)
plt.title("Boxplot: Employee Attrition vs NumCompaniesWorked of Employee")
#MonthlyIncome of Employee vs Employee Attrition
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, width=0.3)
plt.title("Boxplot: Employee Attrition vs MonthlyIncome of Employee")
#DailyRate of Employee vs Employee Attrition
sns.boxplot(x="Attrition", y="DailyRate", data=df, width=0.3)
plt.title("Boxplot: Employee Attrition vs DailyRate of Employee")
#DistanceFromHome of Employee vs Employee Attrition
sns.boxplot(x="Attrition", y="DistanceFromHome", data=df,width=0.3)
plt.title("Boxplot: Employee Attrition vs DistanceFromHome of Employee")
#EnvironmentSatisfaction of Employee vs Employee Attrition
sns.boxplot(x="Attrition", y="EnvironmentSatisfaction", data=df,width=0.2)
plt.title("Boxplot: Employee Attrition vs EnvironmentSatisfaction of Employee")
#JobLevel of Employee vs Employee Attrition
sns.boxplot(x="Attrition", y="JobLevel", data=df,width=0.3)
plt.title("Boxplot: Employee Attrition vs JobLevel of Employee")
for i, Attrition in enumerate(df.Attrition):
    if Attrition=='Yes':
        df.Attrition[i]=1
    else:
        df.Attrition[i]=0

df.head(2)
#Correlation Heat Map
f , ax = plt.subplots(figsize = (14,12))
plt.title('Correlation of Numeric Features with Employee Attrition',y=1,size=16)
sns.heatmap(df.corr(),square = True,  vmax=0.8)

