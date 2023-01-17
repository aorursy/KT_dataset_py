import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

# importing bikes dataset
bikes = pd.read_csv("../input/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv")
bikes.head()
# dropping 'Unnamed: 0' column, because it is not necessary for our analysis
bikes = bikes.drop('Unnamed: 0', 1)
# checking data types to identify which variables are continuous, categorical and count
# note: my dependent variable, "Total" is an integer (count)
bikes.dtypes
# checking if there are any nans in the dataset
bikes.isnull().values.any()
# generating descriptive statistics summary
bikes['Total'].describe()
# checking the distribution to see if there are any outliers
sns.distplot(bikes['Total']);

#skewness and kurtosis
print("Skewness: %f" % bikes['Total'].skew())
print("Kurtosis: %f" % bikes['Total'].kurt())
# creating a new column for average temperature by adding the highest and
#the lowest temperature and dividing it by 2
bikes['Average Temp'] = (bikes['High Temp (°F)'] + bikes['Low Temp (°F)'])/2
# plotting average temperature against total number of bikes 
# linear regression
sns.lmplot('Average Temp', 'Total', data=bikes)
# since I am predicting a count value, I should fit a poisson regression. 
import statsmodels.api as sm

X = bikes['Average Temp']
y = bikes['Total']

# add intercept to input variable
X = sm.add_constant(X)

# fit poisson regression model 
model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

# add poisson fitted values to dataframe
bikes['reg_fit'] = model.fittedvalues
# plot & add a regression line
sns.regplot(bikes['Average Temp'], bikes['Total'], fit_reg=False)
plt.plot(bikes['Average Temp'], bikes['reg_fit']);
