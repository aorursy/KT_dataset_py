import pandas as pd



# read in data

coders_2016 = pd.read_csv("../input/2016-new-coder-survey-/2016-FCC-New-Coders-Survey-Data.csv")

coders_2017 = pd.read_csv("../input/the-freecodecamp-2017-new-coder-survey/2017-fCC-New-Coders-Survey-Data.csv")
# create a subset of the data with only our variables of interest (variables

# that aren't converted numbers won't work)

import pandas as pd

iswoman = pd.DataFrame(coders_2016['Gender'] == "female")

iswoman = iswoman.astype(int)

subset = coders_2016[['Age', 'CommuteTime', 'HasChildren', 

           'AttendedBootcamp', 'HasDebt',

           'HoursLearning', 'MonthsProgramming', 'Income']]

subset['IsWoman'] = iswoman

subset.dropna(inplace=True)



X = subset[['Age', 'CommuteTime', 'HasChildren', 

           'AttendedBootcamp', 'IsWoman', 'HasDebt',

           'HoursLearning', 'MonthsProgramming']]

# get a vector with our output variable

y = subset['Income']



print('Number of data points: ' + str(len(y)))
from sklearn.linear_model import ElasticNetCV

regr = ElasticNetCV(cv=10, random_state=0)

regr.fit(X, y)
print(regr.intercept_)
coefficients = pd.DataFrame()

coefficients['columns'] = X.columns

coefficients['coef'] = regr.coef_

print(coefficients)
variables_non_zero = coefficients[coefficients['coef'] != 0]['columns']

print(variables_non_zero)
# turn our list of formulas into a variable

X = subset[variables_non_zero]



# fit a glm model

from sklearn.linear_model import LinearRegression

regr = LinearRegression()

regr.fit(X, y)
import matplotlib.pyplot as plt

y_pred = regr.predict(X)

residual = y - y_pred

plt.scatter(y_pred,residual)
print(regr.intercept_)
coefficients = pd.DataFrame()

coefficients['columns'] = X.columns

coefficients['coef'] = regr.coef_

print(coefficients)
# added-variable plots for our model
coders_2017.head()
# create a subset of the data with only our variables of interest (variables

# that aren't converted numbers won't work)

import pandas as pd

iswoman = pd.DataFrame(coders_2017['Gender'] == "female")

iswoman = iswoman.astype(int)

subset = coders_2017[['Age', 'HasChildren', 

           'AttendedBootcamp', 'HasDebt',

           'HoursLearning', 'MonthsProgramming', 'Income']]

subset['IsWoman'] = iswoman

subset.dropna(inplace=True)



X = subset[['Age', 'HasChildren', 

           'AttendedBootcamp', 'IsWoman', 'HasDebt',

           'HoursLearning', 'MonthsProgramming']]

# get a vector with our output variable

y = subset['Income']



print('Number of data points: ' + str(len(y)))
from sklearn.linear_model import ElasticNetCV

regr = ElasticNetCV(cv=10, random_state=0)

regr.fit(X, y)
coefficients = pd.DataFrame()

coefficients['columns'] = X.columns

coefficients['coef'] = regr.coef_

print(coefficients)
variables_non_zero = coefficients[coefficients['coef'] != 0]['columns']

print(variables_non_zero)
# turn our list of formulas into a variable

X = subset[variables_non_zero]



# fit a glm model

from sklearn.linear_model import LinearRegression

regr = LinearRegression()

regr.fit(X, y)
import matplotlib.pyplot as plt

y_pred = regr.predict(X)

residual = y - y_pred

plt.scatter(y_pred,residual)
coefficients = pd.DataFrame()

coefficients['columns'] = X.columns

coefficients['coef'] = regr.coef_

print(coefficients)