# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

df.head()
df.isnull().any()
f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,9))

sns.countplot(x='Attrition',data=df,ax=ax1)

sns.distplot(df['MonthlyIncome'],ax=ax2)

sns.countplot(x="JobSatisfaction",data=df,ax=ax3)

sns.countplot(x="WorkLifeBalance",data=df,ax=ax4)
plt.figure()

cols = ["MonthlyIncome","Age","TotalWorkingYears","EmployeeNumber","YearsSinceLastPromotion"]

sns.pairplot(df[cols],diag_kind="kde",kind="reg")
f,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(12,8))

sns.boxplot(x=df['Education'],y=df['MonthlyIncome'],ax=ax1)

sns.boxplot(x=df['JobLevel'],y=df['MonthlyIncome'],ax=ax2)

sns.boxplot(x=df['JobSatisfaction'],y=df['MonthlyIncome'],ax=ax3)

sns.boxplot(x=df['Gender'],y=df['MonthlyIncome'],ax=ax4)

sns.boxplot(x=df['MaritalStatus'],y=df['MonthlyIncome'],ax=ax5)

sns.jointplot(x="YearsAtCompany",y="MonthlyIncome",data=df,kind="hex")
corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);