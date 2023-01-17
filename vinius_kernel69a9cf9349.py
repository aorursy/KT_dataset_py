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
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import statsmodels.api as sm

import math



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



from scipy import stats

from scipy.stats import kurtosis, skew



import pickle as pk
import pandas as pd

a = pd.read_csv("../input/austin-weather/austin_weather.csv")
a.head()
#Changing the index column 

a.index = pd.to_datetime(a['Date'])

a
c.columns
b = a.drop(['Date'], axis = 1)
b.head()
#Checking the datatypes of the dataset

b.dtypes
#Renaming columnns in the dataset

new_column_names_one = {'TempHighF':'TempHighFahrenheit',

                       'TempAvgF' : 'TempAvgFahrenheit' }

c = b.rename(columns = new_column_names_one)

c.head()
#Checking the missing values

#If there are missing values, the result will be tru

#If there are no missing values, the result will be false for any of the columns with no values

d = c.isna().any()

d
#Dropping missing values

e = c.dropna()

e


#Creating a scatter plot for the data

plt.plot(x, y, 'o', color = 'purple', label = 'Comparison between the highest and lowest temperature')

#Make sure it is formatted

plt.title('HIGH TEMPERATURE VS. LOW TEMPERATURE')

plt.xlabel('Highest temperature')

plt.ylabel('Lowest temperature')

plt.legend()

plt.show()
#Measure the corelation

c.corr()
c.columns
#Have a summary of the dataset

c.describe()
c["VisibilityAvgMiles"].describe()
#Calculation of kurtosis using the Fisher method. an alternative that can be used is the Pearson method which uses regular kurtosis

TempAvgFahrenheitKurtosis = kurtosis(c['TempAvgFahrenheit'], fisher = True)

TempAvgFahrenheitKurtosis

#print( '\nKurtosis for normal distribution :', kurtosis(y1, fisher = True)) 

#The above code calculates the kurtosis for a list after some concactenation

#From my research, it seems as if itmay not be possible to get the kurtosis of the datatype object
#Finding the skewness of the TempAvgFahrenheit column

#A negative skewness indicates that the dataset column has a steeper left tail as compared to the right tail



TempAvgFahrenheitSkewness = skew(c['TempAvgFahrenheit'])

TempAvgFahrenheitSkewness
#The second alternative for displaying the kurtosis and skewness of the features in the dataset

display("Average temperature kurtosis: " + str(TempAvgFahrenheitKurtosis))

display("Average temperature skewness: " + str(TempAvgFahrenheitSkewness))
#The second alternative for displaying the kurtosis and skewness of the features in the dataset

display("Average temperature kurtosis: {:.2}".format(TempAvgFahrenheitKurtosis))

display("Average temperature skewness: {:.2}".format(TempAvgFahrenheitSkewness))
#The larger the data, the higher the intensity of skew and kurtosis

#The reccommendation I would give to solve this problem is splitting of the dataset into smaller chunks 

#Here is an advanced format of measuring the skewness and kurtosis of a dataset

display(stats.kurtosistest(c['TempHighFahrenheit']))
#Perform a skew test

display(stats.skewtest(c['TempHighFahrenheit']))
c.dtypes
c['TempHighFahrenheit']
cat_1 = [10, 11, 12]

cat_2 = [25, 22, 30]

cat_3 = [12, 14, 15]



df1 = pd.DataFrame({'cat1':cat_1, 'cat2':cat_2, 'cat3':cat_3})

df1
c.head()
#Creating a dataframe 

num = [0, 1, 2, 3, 4]

num2 = pd.DataFrame(num)

num2
num2.columns = ['integers']

num2
#Adding a new column to a dataframe 

num2['values'] = 3

num2
num2['values'] = num2['values'] + 5
num2
c.head(5)
data = []

dataset = pd.DataFrame(data)
dataset['TempHighFahrenheit'] = c['TempHighFahrenheit']

dataset.head()
dataset['TempLowF'] = c['TempLowF']

dataset.head()
new_dataset_column = {'TempLowF':'TempLowFahrenheit'}

dataset = dataset.rename(columns = new_dataset_column)

dataset.head()
dataset.describe()
dataset.corr()
TempHighFahrenheitSkew = skew(dataset['TempHighFahrenheit'])

TempHighFahrenheitSkew
display('The skew for the highest temperatures is {:.2}'.format(TempHighFahrenheitSkew))
TempLowFahrenheitSkew = skew(dataset['TempLowFahrenheit'])

TempLowFahrenheitSkew
display('The skew for the lowest temperatures is {:.2}'.format(TempLowFahrenheitSkew))
TempHighFahrenheitKurtosis = kurtosis(dataset['TempHighFahrenheit'])

TempHighFahrenheitKurtosis
display('The kurtosis for the highest temperatures is {:.2}'.format(TempHighFahrenheitKurtosis))
TempLowFahrenheitKurtosis = kurtosis(dataset['TempLowFahrenheit'])

TempLowFahrenheitKurtosis
display('The kurtosis for the lowest temperature is {:.2}'.format(TempLowFahrenheitKurtosis))
#Performance of a kurtosis test

KurtosisTestTempHighFahrenheit = stats.kurtosis(dataset['TempHighFahrenheit'])

KurtosisTestTempHighFahrenheit
KurtosisTestTempLowFahrenheit = stats.kurtosis(dataset['TempLowFahrenheit'])

KurtosisTestTempLowFahrenheit
#Performance of a skew test

SkewTempHighFahrenheit = stats.skew(dataset['TempHighFahrenheit'])

SkewTempHighFahrenheit
SkewTempLowFahrenheit = stats.skew(dataset['TempLowFahrenheit'])

SkewTempLowFahrenheit
x = dataset['TempHighFahrenheit']

y = dataset['TempLowFahrenheit']

plt.plot(x, y, 'o', color = 'purple', label = 'Comparison between the highest and lowest temperature')

plt.title('HIGH TEMPERATURE VS. LOW TEMPERATURE')

plt.xlabel('Highest temperature')

plt.ylabel('Lowest temperature')

plt.legend()

plt.show()
#Plotting a histogram of each of the features in the dataset

dataset.hist(grid = False, color = 'blue')
#Definition of the input(X) and output variable(Y)

X = dataset.drop('TempLowFahrenheit', axis = 1)

Y = dataset[['TempLowFahrenheit']]

#Divide the data into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1, test_size = 0.2) 

X_train.size

X_test.size
y_train.size
y_test.size
#Coded by Kevin Mugo

#In a normal linear regression equation y = b0 + b1x:

#y is the output feature

#b0 is the y-intercept

#b1 is the coefficient of x

#x is the input feature

#Creation of the linear regression model

regression_model = LinearRegression()

#Pass the X_train and y_train data

regression_model.fit(X_train, y_train)
#Getting the coefficient and intercept

intercept = regression_model.intercept_[0]

coefficient = regression_model.coef_[0][0]

display('The intercept is {:.2}'.format(intercept))

display('The coefficient is {:.2}'.format(coefficient))
#Making a prediction

prediction = regression_model.predict([[67.33]])

predicted_value = prediction[0][0]

display('The predicted value is {:.6}'.format(predicted_value))
#Displaing the first five predictions

y_predict = regression_model.predict(X_test)

#Show the first five predictions

y_predict[:5]
#Evaluating the accuracy of the model that was created

#The same model will be created using the statsmodel.api library that was imported as sm in cell 1 in this notebook

#The reason why it is used is because it has a lot of predefined functions for calculating metrics like confidence intervals and p-values

#The output of statsmodel.api will be slightly different from the sklearn library but not by a very big margin

#Definition of the input

X2 = sm.add_constant(X)

#The Ordinary Least Square method is used to estimate the unknown parameters by creating a model that will minimize the sum of squared errors between the observed data and the predicted one

model = sm.OLS(y, X2)

#Fitting the data

est = model.fit()

#The code will have an error due to some technical problems in the Python library(statsmodel.api)

#Calculation of the confidence interval

#By default, 95% of the intervals are used out of a range of about 100 times. About 95% of them would have a true coefficient

est.conf_int()

#From the code above, the coefficient is between 0.823098 and 0.872064

#Comparing the answer obtained here and cell [127], there is a high success rate in this algorithm
#Hypothesis testing

#I am trying to disapprove null hypothesis(There is no relationship between X and y & the coefficient equals to zero)

#I am trying to approve alternative hypothesis(There is a relationship between X and y & the coefficient is not equal to zero

est.pvalues

#There surely exists a relationship between the two variables because the probability of the coefficient being equal to zero is lower than 0.05. 

#The probability obtained is 0.000000e+00

#The null hypothesis can therefore be ignored
#Model fit is the next process which involves the calculation of the following errors

#It involves comparing the predictions(y_predictions) and comparing them to the actual values of y(y_actuals)
#Mean Absolute Error calculates the errors but gives no magnitude(too high/low)

model_mae = mean_absolute_error(y_test, y_predict)

display("The mean absolute error is {:.5}".format(model_mae))
#Mean Square Error is like a "punisher" for it calculates the mean ofthe squares the errors

model_mse = mean_squared_error(y_test, y_predict)

display("The mean squared error is {:.5}".format(model_mse))
#The Root Mean Square Error finds the mean of the Mean Squared Error

model_rmse = math.sqrt(model_mse)

display("The root mean square error is {:.5}".format(model_rmse))
#Calculation or R-Squared(It is used to obtain the relationship between the model and the data)

#The higher the value of R-Squared, the better the model

#An increase in features increases the R-Squared

#But this does not mean an improvement of the accuracy of the model

#The new R-Squared can be calculated by evaluating the adjsuted R-Squared

model_r2 = r2_score(y_test, y_predict)

model_r2
#Create a summary of the model output

est.summary()
#Obtaining the residuals and plotting them in a histogram

(y_test - y_predict).hist(grid = True, color = '#FF0000')

plt.title("Model residuals")

plt.show()
#Scatter plots are used to plot data points on horizontal and vertical axis in the attempt to show how much one variable is affected by another. 

#Each row in the data table is represented by a marker the position depends on its values in the columns set on the X and Y axes

plt.scatter(X_test, y_test, color = 'black', label = 'Temperature')

plt.plot(X_test, y_predict, color = 'royalblue', linewidth = 3, linestyle = '-', label = 'Regression Line')

plt.title('Linear Regression Model')

plt.xlabel('TempHighFahrenheit')

plt.ylabel('TempLowFahrenheit')

plt.gca().set_facecolor('#F5F5F5')

plt.legend()

plt.show()
#Pickle the model

with open('my_linear_regression.sav', 'wb') as f:

    pk.dump(regression_model, f)

    

#Load it back again

with open('my_linear_regression.sav', 'rb') as f:

    regression_model_2 = pk.load(f)

    

#Make a new prediction

regression_model_2.predict([[67.33]])