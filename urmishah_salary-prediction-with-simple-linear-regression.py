import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 
df = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Linear_Regression_Introduction/master/Salary_Data.csv')
df.head()
df.isnull().sum()
df.describe()
df.shape
print("\n\nWhat are the unique categories?")

print(df["Salary"].unique())

# How many unique values are there

print("\n\nHow many unique categories there are?")

print(df["Salary"].nunique())

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(8,6))

sns.scatterplot(x = 'YearsExperience' ,y='Salary', data=df)
import seaborn as sns

plt.figure(figsize=(8,6))

sns.scatterplot(x = 'Salary' ,y='YearsExperience', data=df)
plt.figure(figsize=(8,6))

sns.scatterplot(x = 'Salary' ,y='YearsExperience', hue='Salary', data=df)
plt.figure(figsize=(12, 6))

sns.lineplot(x="YearsExperience", y="Salary", data=df)
plt.figure(figsize=(12, 6))

sns.pointplot(x="YearsExperience", y="Salary", data=df)
plt.figure(figsize=(12, 6))

sns.swarmplot(x="YearsExperience", y="Salary", data=df, size = 7)
#splitting data into features(X) and target variable(y)

X = df[['YearsExperience']]

y = df['Salary']
#splitting data into 80% train and 20% test 

# import SK Learn train test split

from sklearn.model_selection import train_test_split 



# Assign variables to capture train test split output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression() 

linear_regressor.fit(X_train, y_train) 
#finding intercept and coefficients(bo and b1)

print(linear_regressor.intercept_)

print(linear_regressor.coef_)
y_pred = linear_regressor.predict(X_test)
#plotting our predictions

plt.plot(X_test, y_test,'rx')

plt.plot(X_test, y_pred, color='black')

plt.show()




# Plotting the actual and predicted values



c = [i for i in range (1,len(y_test)+1,1)]

plt.plot(c,y_test,color='r',linestyle='-')

plt.plot(c,y_pred,color='b',linestyle='-')

plt.xlabel('Salary')

plt.ylabel('index')

plt.title('Prediction')

plt.show()
#plotting the error

c = [i for i in range(1,len(y_test)+1,1)]

plt.plot(c,y_test-y_pred,color='green',linestyle='-')

plt.xlabel('index')

plt.ylabel('Error')

plt.title('Error Value')

plt.show()
from sklearn.metrics import r2_score,mean_squared_error

r2 = r2_score(y_test,y_pred)

print('r square :',r2)