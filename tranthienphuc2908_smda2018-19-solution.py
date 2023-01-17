# loading library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import math

print(os.listdir("../input"))

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/dfSMDA.csv')



# adding the column 'Year' to the dataframe

df['Year'] = range(len(df))

for i in range(len(df)):

    yr =  df.iloc[i,0]

    df.iloc[i,4] = yr[0:4]



# acquiring training and test data by filtering the column 'Year'

train_df = df[df['Year'] != "2018"]

test_df = df[df['Year'] == "2018"]



# droping the column 'Year'

df = df.drop(columns = ['Year'])

train_df = train_df.drop(columns = ['Year'])

test_df = test_df.drop(columns = ['Year'])
print(train_df.columns.values)
# preview the data

train_df.head()
train_df.tail()
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
print(np.percentile(train_df['Temp'], 30))
# histogram

train_df['Temp'].plot.hist()

plt.title('Histogram of Temperature')

plt.xlabel('Temperature')
# boxplot

train_df['Temp'].plot.box()

plt.title('Boxplot of Temperature')
# histogram

train_df['N_Customers'].plot.hist()

plt.title('Histogram of the Number of Customers')

plt.xlabel('Number of Customers')
# boxplot

train_df['N_Customers'].plot.box()

plt.title('Boxplot of the Number of Customers')
# histogram

train_df['Beach_Park_Closed'].plot.hist()

plt.title('Histogram of the closing of the Beach Park')
# line graph

fig, axe = plt.subplots(3, sharex=True)



# Temperature

axe[0].plot(range(17544), train_df['Temp'], 'b')

axe[0].set_title('Line graph of Temperature')



# Number of Customers

axe[1].plot(range(17544), train_df['N_Customers'], 'r')

axe[1].set_title('Line graph of the Number of Customers')



# Beach Park

axe[2].plot(range(17544), train_df['Beach_Park_Closed'], 'y')

axe[2].set_title('Line graph of the closing of the Beach Park')

axe[2].set_xlabel('Time (Hour)')
# splitting the training dataset into 2016-dataset and 2017-dataset

train_df['Year'] = range(len(train_df))

for i in range(len(train_df)):

    yr =  train_df.iloc[i,0]

    train_df.iloc[i,4] = yr[0:4]

train_df16 = train_df[train_df['Year'] == "2016"]

train_df17 = train_df[train_df['Year'] == "2017"]

train_df = train_df.drop(columns = ['Year'])
# line graph of the Number of Customers for 2016 and 2017

train_df16['N_Customers'].plot.line(color = 'b')

train_df17['N_Customers'].plot.line(color = 'r')

plt.title('Linegraph of the Number of Customers')

plt.xlabel('Time')

plt.ylabel('Number of Customers')
# finding the period of time when the Number of Customer was less than 200 in 2016

train_df16[train_df16['N_Customers'] < 200].head()         
# finding the period of time when the Beach Park was opened in 2016

train_df16[train_df16['Beach_Park_Closed'] == 0].head()         
# finding the period of time when the Number of Customer was less than 200 in 2016

train_df16[train_df16['N_Customers'] < 200].tail()         
# finding the period of time when the Beach Park was opened in 2016

train_df16[train_df16['Beach_Park_Closed'] == 0].tail()         
# finding the period of time when the Number of Customer was less than 200 in 2017

train_df17[train_df17['N_Customers'] < 200].head()         
# finding the period of time when the Beach Park was opened in 2017

train_df17[train_df17['Beach_Park_Closed'] == 0].head()         
# finding the period of time when the Number of Customer was less than 200 in 2017

train_df17[train_df17['N_Customers'] < 200].tail()         
# finding the period of time when the Beach Park was opened in 2017

train_df17[train_df17['Beach_Park_Closed'] == 0].tail()
train_df.describe(include = ['O'])
# scatter plot of Temp vs N_Customers

plt.scatter(train_df['Temp'], train_df['N_Customers'])

plt.title('Temp vs N_Customers')

plt.xlabel('Temp')

plt.ylabel('N_Customers')
# correlation coeffient between Temp and N_Customers

print("The correlation matrix of Temp and N_Customers:")

np.corrcoef(train_df['Temp'], train_df['N_Customers'])
# scatterplot of Temp vs N_Customers in the presence of Beach_Park_Closed 

cldtrain_df = train_df[train_df['Beach_Park_Closed'] == 1]

plt.scatter(cldtrain_df['Temp'], cldtrain_df['N_Customers'], color = ['red'])

opntrain_df = train_df[train_df['Beach_Park_Closed'] == 0]

plt.scatter(opntrain_df['Temp'], opntrain_df['N_Customers'], color = ['blue'])

plt.title('Temp vs N_Customers in the present of Beach_Park_Closed')

plt.xlabel('Temp')

plt.ylabel('N_Customers')
# scatterplot of Temp vs N_Customers when the Beach Park was opened 

plt.scatter(opntrain_df['Temp'], opntrain_df['N_Customers'])

plt.title('Temp vs N_Customers when the Beach Park is opened')

plt.xlabel('Temp')

plt.ylabel('N_Customers')
# loading library

import datetime



# for training dataset

train_df['Weekday'] = train_df['Date_1']

train_df['Weekday'], train_df['Hour'] = train_df['Weekday'].str.split(' ', 1).str

for i in range(len(train_df)):

    hr = train_df.iloc[i,5]

    train_df.iloc[i,5] = float(hr[0:2])

    dt = train_df.iloc[i,4]

    year, month, day = (int(x) for x in dt.split('-'))    

    ans = datetime.date(year, month, day)

    train_df.iloc[i,4] = ans.strftime("%A")



# for test dataset

test_df['Weekday'] = test_df['Date_1']

test_df['Weekday'], test_df['Hour'] = test_df['Weekday'].str.split(' ', 1).str

for i in range(len(test_df)):

    hr = test_df.iloc[i,5]

    test_df.iloc[i,5] = float(hr[0:2])

    dt = test_df.iloc[i,4]

    year, month, day = (int(x) for x in dt.split('-'))    

    ans = datetime.date(year, month, day)

    test_df.iloc[i,4] = ans.strftime("%A")
# histogram of N_Customers in the presence of Weekday

g = sns.FacetGrid(train_df, col = 'Weekday')

g.map(plt.hist, 'N_Customers')
# creating IsWeekend

train_df['IsWeekend'] = np.where((train_df['Weekday'] == "Saturday") | (train_df['Weekday'] == "Sunday"), 1, 0)

test_df['IsWeekend'] = np.where((test_df['Weekday'] == "Saturday") | (test_df['Weekday'] == "Sunday"), 1, 0)
# histogram of N_Customers in the presence of IsWeekend

g = sns.FacetGrid(train_df, col='IsWeekend')

g.map(plt.hist, 'N_Customers')
# scatterplot of Hour vs N_Customers in the presence of Beach_Park_Closed 

plt.scatter(train_df['Hour'], train_df['N_Customers'])

plt.title('Hour vs N_Customers')

plt.xlabel('Hour')

plt.ylabel('N_Customers')
# scatterplot of Hour vs N_Customers when the Beach Park was opened 

opntrain_df = train_df[train_df['Beach_Park_Closed'] == 0]

plt.scatter(opntrain_df['Hour'], opntrain_df['N_Customers'])

plt.title('Hour vs N_Customers when the Beach Park was opened')

plt.xlabel('Hour')

plt.ylabel('N_Customers')
# correlation coefficient between Hour and N_Customers 

print("The correlation matrix of Hour and N_Customers:")

np.corrcoef(train_df['Hour'], train_df['N_Customers'])
# creating Is8to10

train_df['Is8to10'] = np.where((train_df['Hour'] >= 20) & (train_df['Hour'] <= 22), 1, 0)

test_df['Is8to10'] = np.where((test_df['Hour'] >= 20) & (test_df['Hour'] <= 22), 1, 0)
# histogram of N_Customers in the presence of Is8to10

Is8to10 = train_df[train_df['Is8to10'] == 1]

Not8to10 = train_df[train_df['Is8to10'] == 0]

fig, axe = plt.subplots(2, sharex=True)

axe[0].hist(Is8to10['N_Customers'])

axe[0].set_title('Histogram of N_Customers from 8 to 10 pm')

axe[1].hist(Not8to10['N_Customers'])

axe[1].set_title('Histogram of N_Customers not from 8 to 10 pm')

axe[1].set_xlabel('N_Customers')
train_df = train_df.drop(['Date_1', 'Weekday', 'Hour'], axis = 1)

test_df = test_df.drop(['Date_1', 'Weekday', 'Hour'], axis = 1)
cor = train_df.corr()

cor
# loading library

from sklearn import linear_model

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

import statsmodels.formula.api as smf
N_CustomersTrain = train_df['N_Customers']

N_CustomersTest = test_df['N_Customers']



predictorsTrain = train_df.drop(['N_Customers'], axis = 1)

predictorsTest = test_df.drop(['N_Customers'], axis = 1)



predictorsTrain_std = (predictorsTrain - predictorsTrain.mean()) / predictorsTrain.std()

predictorsTest_std = (predictorsTest - predictorsTest.mean()) / predictorsTest.std()
# fitting model 

reg = linear_model.LinearRegression(copy_X = True, fit_intercept = True, normalize = False)

reg.fit(predictorsTrain_std, N_CustomersTrain)



# predicting number of customers

N_CustomersTrain_pred = reg.predict(predictorsTrain_std)

N_CustomersTest_pred = reg.predict(predictorsTest_std)



# computing Residual sum of square

#RSSTrain = len(train_df) * mean_squared_error(N_CustomersTrain, N_CustomersTrain_pred)

#RSSTest = len(test_df) * mean_squared_error(N_CustomersTest, N_CustomersTest_pred)



# computing Root mean squared error

RMSETrain = math.sqrt(mean_squared_error(N_CustomersTrain, N_CustomersTrain_pred))

RMSE_OLS1 = RMSETrain

RMSETest = math.sqrt(mean_squared_error(N_CustomersTest, N_CustomersTest_pred))

RMSE_OLS = RMSETest



# computing Coefficient of determination

r2Train = r2_score(N_CustomersTrain, N_CustomersTrain_pred)

r2_OLS = r2Train

r2Test = r2_score(N_CustomersTest, N_CustomersTest_pred)



#print("Residual Sum of Square on the Training set: ", RSSTrain)

print('-'*60)

print("Root mean squared error on the Training set: ", RMSETrain)

print('-'*60)

print("Coefficient of Determination on the Training set: ", r2Train)

print('-'*60)

#print("Residual Sum of Square on the Test set: ", RSSTest)

print('-'*60)

print("Root mean squared error on the Test set: ", RMSETest)

print('-'*60)

print("Coefficient of Determination on the Test set: ", r2Test)
# showing fitted model

predictorsTrain_std['N_Customers'] = N_CustomersTrain

results = smf.ols('N_Customers ~ Beach_Park_Closed + Temp + IsWeekend + Is8to10', data = predictorsTrain_std).fit()

predictorsTrain_std = predictorsTrain_std.drop('N_Customers', axis = 1)

print(results.summary())
# loading library

import itertools

from tqdm import tnrange, tqdm_notebook
# fitting function

def fit_linear_reg(X_train, y_train, X_test, y_test):

    # Fit linear regression model and return RSS and R squared values

    model_k = linear_model.LinearRegression(fit_intercept = True)

    model_k.fit(X_train, y_train)

    RSS = math.sqrt(mean_squared_error(y_test, model_k.predict(X_test)))

    R_squared = model_k.score(X_test, y_test)

    return RSS, R_squared
# best subset seletion function

def bestsubset(X_train, y_train, X_test, y_test):

    # Initialization variables

    RSS_list, R_squared_list, feature_list = [],[], []

    numb_features = []



    # Looping over features in X

    for k in tnrange(1, len(X_train.columns) + 1, desc = 'Loop...'):



        # Looping over all possible combinations: choose k

        for combo in itertools.combinations(X_train.columns, k):

            tmp_result = fit_linear_reg(X_train[list(combo)], y_train, X_test[list(combo)], y_test)   # Store temp result 

            RSS_list.append(tmp_result[0])                  # Append lists

            R_squared_list.append(tmp_result[1])

            feature_list.append(combo)

            numb_features.append(len(combo))   



    # Store in DataFrame

    df = pd.DataFrame({'numb_features': numb_features,'RMSE': RSS_list, 'R_squared': R_squared_list,'features': feature_list})

    return df
k = 10        # number of folds

np.random.seed(seed = 1)

folds = np.random.choice(k, size = len(train_df), replace = True)



# Create a DataFrame to store the results of our upcoming calculations

cv_errors = pd.DataFrame(columns = range(1, k + 1), index = range(1, 20))

cv_errors = cv_errors.fillna(0)
df = bestsubset(predictorsTrain_std[folds != 0], N_CustomersTrain[folds != 0], predictorsTrain_std[folds == 0], N_CustomersTrain[folds == 0]) 

    

# Outer loop iterates over all folds

for j in range(2, k + 1):



    result = bestsubset(predictorsTrain_std[folds != (j - 1)], N_CustomersTrain[folds != (j - 1)], predictorsTrain_std[folds == (j - 1)], N_CustomersTrain[folds == (j - 1)])

    df['RMSE'] = result['RMSE'] + df['RMSE'] 

    df['R_squared'] = result['R_squared'] + df['R_squared']



df['RMSE'] = df['RMSE'] / k

df['R_squared'] = df['R_squared'] / k

df
df_min = df[df.groupby('numb_features')['RMSE'].transform(min) == df['RMSE']]

df_max = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]

display(df_min)

display(df_max)
df['min_RMSE'] = df.groupby('numb_features')['RMSE'].transform(min)

df['max_R_squared'] = df.groupby('numb_features')['R_squared'].transform(max)
# line graph of cross-validated RMSE and R^2

fig = plt.figure(figsize = (16, 6))

ax = fig.add_subplot(1, 2, 1)



ax.scatter(df.numb_features, df.RMSE, alpha = .2, color = 'darkblue')

ax.set_xlabel('# Features')

ax.set_ylabel('RMSE')

ax.set_title('Cross validated RMSE - Best subset selection')

ax.plot(df.numb_features, df.min_RMSE, color = 'r', label = 'Best subset')

ax.legend()



ax = fig.add_subplot(1, 2, 2)

ax.scatter(df.numb_features, df.R_squared, alpha = .2, color = 'darkblue')

ax.plot(df.numb_features, df.max_R_squared, color = 'r', label = 'Best subset')

ax.set_xlabel('# Features')

ax.set_ylabel('R squared')

ax.set_title('Cross validated R_squared - Best subset selection')

ax.legend()



plt.show()
# fitting function

def fit_ridge_reg(alpha, X_train, y_train, X_test, y_test):

    # Fit ridge regression model and return RSS and R squared values

    model_k = linear_model.Ridge(alpha = alpha)

    model_k.fit(X_train, y_train)

    RSS = math.sqrt(mean_squared_error(y_test, model_k.predict(X_test)))

    R_squared = model_k.score(X_test, y_test)

    return RSS, R_squared
# ridge regression function

def ridge_reg(alphas, X_train, y_train, X_test, y_test):

    # Initialization variables

    RSS_list, R_squared_list = [],[]



    # Looping over features in X

    for alpha in alphas:

        tmp_result = fit_ridge_reg(alpha, X_train, y_train, X_test, y_test)   # Store temp result 

        RSS_list.append(tmp_result[0])                  # Append lists

        R_squared_list.append(tmp_result[1])

    

    # Store in DataFrame

    df = pd.DataFrame({'Alpha': alphas,'RMSE': RSS_list, 'R_squared': R_squared_list})

    return df
n_alphas = 200

alphas = np.logspace(-1, 4, n_alphas)

df = ridge_reg(alphas, predictorsTrain_std[folds != 0], N_CustomersTrain[folds != 0], predictorsTrain_std[folds == 0], N_CustomersTrain[folds == 0]) 

    

# Outer loop iterates over all folds

for j in range(2, k + 1):

    

    result = ridge_reg(alphas, predictorsTrain_std[folds != (j - 1)], N_CustomersTrain[folds != (j - 1)], predictorsTrain_std[folds == (j - 1)], N_CustomersTrain[folds == (j - 1)])

    df['RMSE'] = result['RMSE'] + df['RMSE'] 

    df['R_squared'] = result['R_squared'] + df['R_squared']



df['RMSE'] = df['RMSE'] / k

df['R_squared'] = df['R_squared'] / k

df
df_min = df[df['RMSE'] == np.min(df['RMSE'])]

df_max = df[df['R_squared'] == np.max(df['R_squared'])]

display(df_min)

display(df_max)
# line graph of cross-validated RMSE and R^2

fig = plt.figure(figsize = (16, 6))

ax = fig.add_subplot(1, 2, 1)



ax.set_xlabel('Alpha')

ax.set_ylabel('RMSE')

ax.set_title('Cross validated RMSE - Ridge Regression')

ax.plot(df.Alpha, df.RMSE, color = 'r')



ax = fig.add_subplot(1, 2, 2)

ax.plot(df.Alpha, df.R_squared, color = 'r')

ax.set_xlabel('Alpha')

ax.set_ylabel('R squared')

ax.set_title('Cross validated R_squared - Ridge Regression')



plt.show()
# fitting the model with the best alpha

reg = linear_model.Ridge(alpha = 0.1)

reg.fit(predictorsTrain_std, N_CustomersTrain)



# predicting the number of customers in the test dataset

N_CustomersTrain_pred = reg.predict(predictorsTrain_std)

N_CustomersTest_pred = reg.predict(predictorsTest_std)



#RSSTest = len(test_df) * mean_squared_error(N_CustomersTest, N_CustomersTest_pred)



# computing RMSE of the test dataset

RMSETrain = math.sqrt(mean_squared_error(N_CustomersTrain, N_CustomersTrain_pred))

RMSE_Ridge1 = RMSETrain

RMSETest = math.sqrt(mean_squared_error(N_CustomersTest, N_CustomersTest_pred))

RMSE_Ridge = RMSETest



# computing R^2 of the test dataset

r2Train = r2_score(N_CustomersTrain, N_CustomersTrain_pred)

r2_Ridge = r2Train

r2Test = r2_score(N_CustomersTest, N_CustomersTest_pred)



#print("Residual Sum of Square on the Test set: ", RSSTest)

print('-'*60)

print("Mean squared error on the Test set: ", RMSETest)

print('-'*60)

print("Coefficient of Determination on the Test set: ", r2Test)
def fit_lasso(alpha, X_train, y_train, X_test, y_test):

    # Fit ridge regression model and return RSS and R squared values

    model_k = linear_model.Lasso(alpha = alpha)

    model_k.fit(X_train, y_train)

    RSS = math.sqrt(mean_squared_error(y_test, model_k.predict(X_test)))

    R_squared = model_k.score(X_test, y_test)

    return RSS, R_squared
def lasso(alphas, X_train, y_train, X_test, y_test):

    # Initialization variables

    RSS_list, R_squared_list = [],[]



    # Looping over features in X

    for alpha in alphas:

        tmp_result = fit_lasso(alpha, X_train, y_train, X_test, y_test)   # Store temp result 

        RSS_list.append(tmp_result[0])                  # Append lists

        R_squared_list.append(tmp_result[1])

    

    # Store in DataFrame

    df = pd.DataFrame({'Alpha': alphas,'RMSE': RSS_list, 'R_squared': R_squared_list})

    return df
n_alphas = 200

alphas = np.logspace(-1, 4, n_alphas)

df = lasso(alphas, predictorsTrain_std[folds != 0], N_CustomersTrain[folds != 0], predictorsTrain_std[folds == 0], N_CustomersTrain[folds == 0]) 

    

# Outer loop iterates over all folds

for j in range(2, k + 1):

    

    result = lasso(alphas, predictorsTrain_std[folds != (j - 1)], N_CustomersTrain[folds != (j - 1)], predictorsTrain_std[folds == (j - 1)], N_CustomersTrain[folds == (j - 1)])

    df['RMSE'] = result['RMSE'] + df['RMSE'] 

    df['R_squared'] = result['R_squared'] + df['R_squared']



df['RMSE'] = df['RMSE'] / k

df['R_squared'] = df['R_squared'] / k

df
df_min = df[df['RMSE'] == np.min(df['RMSE'])]

df_max = df[df['R_squared'] == np.max(df['R_squared'])]

display(df_min)

display(df_max)
fig = plt.figure(figsize = (16, 6))

ax = fig.add_subplot(1, 2, 1)



ax.set_xlabel('Alpha')

ax.set_ylabel('RMSE')

ax.set_title('Cross validated RMSE - LASSO')

ax.plot(df.Alpha, df.RMSE, color = 'r')



ax = fig.add_subplot(1, 2, 2)

ax.plot(df.Alpha, df.R_squared, color = 'r')

ax.set_xlabel('Alpha')

ax.set_ylabel('R squared')

ax.set_title('Cross validated R_squared - LASSO')



plt.show()
# fitting the model with the best alpha

reg = linear_model.Lasso(alpha = 0.1)

reg.fit(predictorsTrain_std, N_CustomersTrain)



# predicting the number of customers in the test dataset

N_CustomersTrain_pred = reg.predict(predictorsTrain_std)

N_CustomersTest_pred = reg.predict(predictorsTest_std)



#RSSTest = len(test_df) * mean_squared_error(N_CustomersTest, N_CustomersTest_pred)



# computing RMSE of the test dataset

RMSETrain = math.sqrt(mean_squared_error(N_CustomersTrain, N_CustomersTrain_pred))

RMSE_LASSO1 = RMSETrain

RMSETest = math.sqrt(mean_squared_error(N_CustomersTest, N_CustomersTest_pred))

RMSE_LASSO = RMSETest



# computing R^2 of the test dataset

r2Train = r2_score(N_CustomersTrain, N_CustomersTrain_pred)

r2_LASSO = r2Train

r2Test = r2_score(N_CustomersTest, N_CustomersTest_pred)



#print("Residual Sum of Square on the Test set: ", RSSTest)

print('-'*60)

print("Mean squared error on the Test set: ", RMSETest)

print('-'*60)

print("Coefficient of Determination on the Test set: ", r2Test)
# loading library

from sklearn.decomposition import PCA

from sklearn.model_selection import KFold

from sklearn.preprocessing import scale 

from sklearn import model_selection

from sklearn.linear_model import LinearRegression
pca2 = PCA()

regr = LinearRegression()



# Split into training and test sets

#X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=1)



# Scale the data

predictorsTrain_stdreduced = pca2.fit_transform(scale(predictorsTrain_std))

n = len(predictorsTrain_stdreduced)



# 10-fold CV, with shuffle

kf_10 = model_selection.KFold( n_splits=10, shuffle=True, random_state=1)



RSS_list = []

R_squared_list = []



# Calculate MSE with only the intercept (no principal components in regression)

RSS = -1*model_selection.cross_val_score(regr, np.ones((n,1)), N_CustomersTrain.ravel(), cv=kf_10, scoring = 'neg_mean_squared_error').mean()

RSS_list.append(RSS)

R_squared = model_selection.cross_val_score(regr, np.ones((n,1)), N_CustomersTrain.ravel(), cv=kf_10).mean()    

R_squared_list.append(R_squared)



# Calculate MSE using CV for the 19 principle components, adding one component at the time.

for i in np.arange(1, 5):

    RSS = -1*model_selection.cross_val_score(regr, predictorsTrain_stdreduced[:,:i], N_CustomersTrain.ravel(), cv=kf_10, scoring = 'neg_mean_squared_error').mean()

    RSS_list.append(RSS)

    R_squared = model_selection.cross_val_score(regr, predictorsTrain_stdreduced[:,:i], N_CustomersTrain.ravel(), cv=kf_10).mean()

    R_squared_list.append(R_squared)



df = pd.DataFrame({'numb_components': range(5),'RSS': RSS_list, 'R_squared': R_squared_list})

df
fig = plt.figure(figsize = (16, 6))

ax = fig.add_subplot(1, 2, 1)



ax.set_xlabel('# Principal components in regression')

ax.set_ylabel('RSS')

ax.set_title('Cross validated RSS - PCR')

ax.plot(df.numb_components, df.RSS, color = 'r')



ax = fig.add_subplot(1, 2, 2)

ax.plot(df.numb_components, df.R_squared, color = 'r')

ax.set_xlabel('# Principal components in regression')

ax.set_ylabel('R squared')

ax.set_title('Cross validated R_squared - PCR')



plt.show()
predictorsTest_stdreduced= pca2.transform(scale(predictorsTest_std))[:,:5]



# train regression model on training data 

regr = LinearRegression()

regr.fit(predictorsTrain_stdreduced[:,:5], N_CustomersTrain)



# prediction with test data

pred1 = regr.predict(predictorsTrain_stdreduced)

pred = regr.predict(predictorsTest_stdreduced)



# computing RMSE of the test dataset

RMSETrain = math.sqrt(mean_squared_error(N_CustomersTrain, pred1))

RMSE_PCR1 = RMSETrain

RMSETest = math.sqrt(mean_squared_error(N_CustomersTest, pred))

RMSE_PCR = RMSETest



# computing R^2 of the test dataset

r2Train = r2_score(N_CustomersTrain, pred1)

r2_PCR = r2Train

r2Test = r2_score(N_CustomersTest, pred)



print('-'*60)

print("Mean squared error on the Test set: ", RMSETest)

print('-'*60)

print("Coefficient of Determination on the Test set: ", r2Test)
models = pd.DataFrame({

    'Model': ['OLS regression', 'Best subset selection', 'Ridge Regression', 

              'The LASSO', 'PCR'],

    'Test RMSE': [RMSE_OLS, RMSE_OLS, RMSE_Ridge, RMSE_LASSO, RMSE_PCR],

    'Training RMSE': [RMSE_OLS1, RMSE_OLS1, RMSE_Ridge1, RMSE_LASSO1, RMSE_PCR1],

    'Training R^2': [r2_OLS, r2_OLS, r2_Ridge, r2_LASSO, r2_PCR],})

models.sort_values(by='Test RMSE', ascending=False)