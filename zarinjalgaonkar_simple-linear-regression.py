import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 



df=pd.read_csv('../input/salary/Salary.csv')

df.head()
X = df.iloc[:, :-1].values

y = df.iloc[:,1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
# plot the actual data points of training set

plt.scatter(X_train, y_train, color = 'red')

# plot the regression line

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title('Salary vs Experience (Training set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
# plot the actual data points of test set

plt.scatter(X_test, y_test, color = 'red')

# plot the regression line (same as above)

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title('Salary vs Experience (Test set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
# Example for a person with 12 years experience

new_salary_pred = regressor.predict([[12]])

print('The predicted salary of a person with 12 years experience is ',new_salary_pred)
y_bar = np.mean(y)

print(y_bar)
SST = sum((y_test-y_bar)**2)

SSE = sum((y_test-y_pred)**2)

SSR = SST - SSE

RSquared = SSR / SST

print(RSquared)
import statsmodels.api as sm

xdata = df['YearsExperience']

xdata = sm.add_constant(xdata)

ydata = df['Salary']

model = sm.OLS(ydata, xdata).fit()

model.summary()