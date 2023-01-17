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
import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import LabelEncoder



pd.set_option('display.max_columns', 0)

pd.set_option('display.max_rows', 500)
emp = pd.read_csv("../input/HR-Employee-Attrition.csv")
emp.head(5)
emp.shape
emp.describe().T
emp.info()
plt.figure(figsize=(20,6))

ax = sns.boxplot(data=emp,orient="h",palette='Set1')

plt.show()
emp.tail(5)
emp["EducationField"].value_counts()
plt.figure(figsize=(10,8))

sns.boxplot(x=emp.Department, y=emp.EnvironmentSatisfaction)

plt.xlabel("Department")

plt.ylabel("JobSatisfaction")
emp["Department"].value_counts()
emp.Attrition.value_counts()
emp.Attrition.dtypes
emp['Attrition'].replace('Yes',1,inplace=True)

emp['Attrition'].replace('No',0,inplace=True)
emp.head(5)
emp['EducationField'].replace('Life Sciences',1,inplace=True)

emp['EducationField'].replace('Medical',1,inplace=True)

emp['EducationField'].replace('Marketing',1,inplace=True)

emp['EducationField'].replace('Technical Degree',4, inplace=True)

emp['EducationField'].replace('Other',5, inplace=True)

emp['EducationField'].replace('Human Resources', 6, inplace=True)
emp['Department'].value_counts()
emp['Department'].replace('Research & Development',1,inplace=True)

emp['Department'].replace('Sales',2,inplace=True)

emp['Department'].replace('Human Resources',3,inplace=True)
emp.BusinessTravel.value_counts()
emp.BusinessTravel.replace('Travel_Rarely',1,inplace=True)

emp.BusinessTravel.replace('Travel_Frequently',2,inplace=True)

emp.BusinessTravel.replace('Non-Travel',3,inplace=True)
emp.BusinessTravel.dtypes
emp.Gender
emp.Gender.replace('Male',1,inplace=True)

emp.Gender.replace('Female',0,inplace=True)
emp.Gender.value_counts()
emp.dtypes
plt.figure(figsize=(12,6))

g = sns.distplot(emp["PercentSalaryHike"])

g.set_xlabel("Salary Hike in Percentage of Employees", fontsize=12)

g.set_ylabel("Frequency", fontsize=12)

g.set_title("Percent Salary Hike Distribuition", fontsize=20)
plt.figure(figsize=(12,6))

g = sns.distplot(emp["YearsAtCompany"])

g.set_xlabel("Years Spent in Company", fontsize=12)

g.set_ylabel("Frequency", fontsize=12)

g.set_title("No of Years in Company - Distribuition", fontsize=20)
# Set the width and height of the figure

plt.figure(figsize=(25,10))



# Add title

plt.title("Monthly Income for the assigned Job Role")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=emp.JobRole, y=emp['MonthlyIncome'])



# Add label for vertical axis

plt.ylabel("Monthly Income")
# Set the width and height of the figure

plt.figure(figsize=(25,10))



# Add title

plt.title("Job Satisfaction, by Rating")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=emp.JobRole, y=emp['JobSatisfaction'])



# Add label for vertical axis

plt.ylabel("Job Satisfaction Rating")
# Set the width and height of the figure

plt.figure(figsize=(30,10))

corr = emp.corr()

ax = sns.heatmap(corr,vmin=-1,vmax=1,center=0,annot=True)
# Compute the correlation matrix

corr = emp.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 15))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.figure(figsize=(20,6))

sns.swarmplot(x=emp['JobRole'],

              y=emp['MonthlyIncome'])
# Set the width and height of the figure

plt.figure(figsize=(25,10))



# Add title

plt.title("Job Role - Total Years spent in Company")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=emp.JobRole, y=emp['YearsAtCompany'])



# Add label for vertical axis

plt.ylabel("Total Years")
plt.figure(figsize=(50,20))

emp.plot(kind="scatter",x="EmployeeNumber",y="YearsAtCompany")
emp.duplicated().sum()
# Importing necessary package for creating model

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score
cat_col = emp.select_dtypes(exclude=np.number).columns

num_col = emp.select_dtypes(include=np.number).columns

print(cat_col)

print(num_col)
#One hot encoding

encoded_cat_col = pd.get_dummies(emp[cat_col])
encoded_cat_col
emp_ready_model = pd.concat([emp[num_col],encoded_cat_col], axis = 1)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for i in cat_col:

    emp[i] = label_encoder.fit_transform(emp[i])
#Performng standard scaling

from sklearn.preprocessing import StandardScaler

 

std_scale = StandardScaler().fit(emp)

emp_std = std_scale.transform(emp)
emp_std
from sklearn.preprocessing import MinMaxScaler



minmax_scale = MinMaxScaler().fit_transform(emp)
type(minmax_scale)
X = emp.drop(columns="Attrition")

X.shape
y = emp[["Attrition"]]
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LogisticRegression()

model.fit(X_train, y_train)
model.intercept_
model.coef_
model.predict(X_train)
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
confusion_matrix(y_train,train_predict)
print(classification_report(y_train,train_predict))

print(classification_report(y_test,test_predict))
print("Train Accuracy : ",accuracy_score(y_train,train_predict))

print("Test Accuracy : ",accuracy_score(y_test,test_predict))
model_roc_auc = roc_auc_score(y_test, model.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % model_roc_auc)

plt.plot([0, 10], [0, 1],'r--')

plt.xlim([-0.1, 1.0])

plt.ylim([-0.1, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('model_ROC')

plt.show()