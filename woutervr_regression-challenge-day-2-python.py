import pandas as pd

# read in data

kaggle = pd.read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv", encoding="ISO-8859-1")

stackOverflow = pd.read_csv("../input/so-survey-2017/survey_results_public.csv")
CompensationAmount = kaggle['CompensationAmount']

CompensationAmount = CompensationAmount.str.replace(',', '')

CompensationAmount = CompensationAmount.str.replace('-', '')

CompensationAmount.fillna(0, inplace=True)

CompensationAmount = pd.to_numeric(CompensationAmount)

Age = kaggle['Age']

Age.fillna(0, inplace=True)

Age = pd.to_numeric(Age)

Age = Age[CompensationAmount>0]

CompensationAmount = CompensationAmount[CompensationAmount>0]
from sklearn.linear_model import LinearRegression

import numpy as np

import matplotlib.pyplot as plt

x = Age[:, np.newaxis]

y = CompensationAmount

reg = LinearRegression()

reg.fit(x,y)

y_pred = reg.predict(x)

plt.scatter(x, y)

plt.plot(x, y_pred, color='blue', linewidth=3)
residual = y - y_pred

plt.scatter(x,residual)
Age = Age[CompensationAmount<200000]

CompensationAmount = CompensationAmount[CompensationAmount<200000]

x = Age[:, np.newaxis]

y = CompensationAmount

reg = LinearRegression()

reg.fit(x,y)

y_pred = reg.predict(x)

residual = y - y_pred

plt.scatter(x,residual)
plt.scatter(x, y)

plt.plot(x, y_pred, color='blue', linewidth=3)
stackOverflow.head()
CompanySize = stackOverflow['CompanySize']

CompanySize.fillna('0', inplace=True)

options = CompanySize.unique()

options = options[1:-2]
size_options = [55, 10000, 15, 5, 7500, 255, 2500, 750]

CompanySize_number = []

for i in range(len(CompanySize)):

    to_append = 0

    for j in range(len(options)):

        if options[j] in CompanySize[i]:

            to_append = size_options[j]

            break

    CompanySize_number.append(to_append)

x = pd.to_numeric(CompanySize_number)
ExpectedSalary = stackOverflow['ExpectedSalary']

ExpectedSalary.fillna(0, inplace=True)

ExpectedSalary = pd.to_numeric(ExpectedSalary)

Salary = stackOverflow['Salary']

Salary.fillna(0, inplace=True)

Salary = pd.to_numeric(Salary)

salary_combined = []

for i in range(len(ExpectedSalary)):

    if ExpectedSalary[i] > Salary[i]:

        salary_combined.append(ExpectedSalary[i])

    else:

        salary_combined.append(Salary[i])

y = pd.Series(salary_combined)
y = y[x>0]

x = x[x>0]
x = x[y>0]

y = y[y>0]
from sklearn.linear_model import LinearRegression

import numpy as np

import matplotlib.pyplot as plt

x_fitting = x[:, np.newaxis]

reg = LinearRegression()

reg.fit(x_fitting,y)

y_pred = reg.predict(x_fitting)

plt.scatter(x, y)

plt.plot(x, y_pred, color='blue', linewidth=3)

plt.xlabel('Company size')

plt.ylabel('Salary')