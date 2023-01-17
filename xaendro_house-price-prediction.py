# Standard Data Science libraries for mathematical operations and data management in Dataframes

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Sklearn allows us to preprocess data and contains the regression model classes

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression



import matplotlib.pyplot as plt

import seaborn as sns



# Importing os allows us to print to screen the contents of the "input" folder in which our datasets are stored.

# This makes it easy to identify our data and the path and names to use to recover it.

import os

print(os.listdir("../input"))
# We create a dataframe df and its copy df_train on which we will work.

df = pd.read_csv('../input/train.csv')

df_train = df.copy()



# Dropping target variable and Id

#df_train.drop('SalePrice', axis=1, inplace=True)

df_train.drop('Id', axis=1, inplace=True)



# Printing the first 5 rows of the dataframe

df_train.head(3)



# This is our data at the moment:
# We replace all NaN cells in our columns with 0

df_train.fillna(0, inplace=True)



df.head(3)



# This is the state of our data at this moment:
# Categorical values management

categories = df_train.select_dtypes(include=['category', object]).columns

# Uncomment the following line of code to see which of our columns contain categorical values

# print(categories)

df_categorical = pd.get_dummies(df_train, columns=categories, drop_first=True)

df_train.drop(categories.tolist(), axis=1, inplace=True)

df_train = pd.concat([df_train, df_categorical], axis=1)

df_train.head(3)
# Data standardization 



# Get column names first

names = df_train.columns



# Create the Scaler object

scaler = preprocessing.StandardScaler()



# Fit your data on the scaler object

scaled_train_df = scaler.fit_transform(df_train)

scaled_train_df = pd.DataFrame(scaled_train_df, columns=names)

scaled_train_df.head(3)
# Data Visualization



df_train = scaled_train_df



# Removing duplicate columns

df_train = df_train.loc[:,~df_train.columns.duplicated()]



#Seeing correlation

# df_train.corr()['SalePrice'].sort_values(ascending=False)[120:-120]

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):

#    print(df_train.corr()['SalePrice'].sort_values(ascending=False))



df_train.head(3)
#correlation matrix

corrmat = df_train.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

f, ax = plt.subplots(figsize=(20, 3))

# sns.heatmap(df_train[top_corr_features].corr()[['SalePrice']].sort_values(by='SalePrice', ascending = False), vmax=.99, vmin=-0.99, annot=True, center=0, cmap=sns.diverging_palette(10, 133, as_cmap=False, n=10));

sns.heatmap(df_train[top_corr_features].corr().loc['SalePrice', :].to_frame().T, vmax=.99, vmin=-0.99, annot=True, center=0, cmap=sns.diverging_palette(10, 133, as_cmap=True, n=10)).set_xticklabels(hm.get_xticklabels(), rotation=45);

# Fitting the Model

X = df_train.drop('SalePrice', axis=1)

y = df['SalePrice'].values.reshape(-1,1)

lm = LinearRegression()

lm.fit(X,y)



# Viewing training set score

lm.score(X,y)
# Test Data Pipeline

df_test = pd.read_csv('../input/test.csv')

# Removing Id column

df_test.drop('Id', axis=1, inplace=True)

# Null values management

df_test.fillna(0, inplace=True)

# Categorical values management

categories = df_test.select_dtypes(include=['category', object]).columns

#print(categories)

df_test_categorical = pd.get_dummies(df_test, columns=categories, drop_first=True)

df_test.drop(categories.tolist(), axis=1, inplace=True)

df_test = pd.concat([df_test, df_test_categorical], axis=1)





# Managing different category values:

# This block is not part of the pipeline we applied to the training data.

# It is needed because different categorical values in the test data might generate different dummy variables (and columns) in the testing data

missing = list(set(X.columns.tolist()) - set(df_test.columns.tolist()))

df_test = pd.concat([df_test,pd.DataFrame(columns=missing)], axis = 1)

surplus = list(set(df_test.columns.tolist()) - set(X.columns.tolist()))

df_test = df_test.drop(surplus, axis=1)

df_test.fillna(0, inplace=True)









# Data standardization 

names = df_test.columns

scaler = preprocessing.StandardScaler()

scaled_test_df = scaler.fit_transform(df_test)

scaled_test_df = pd.DataFrame(scaled_test_df, columns=names)

df_test = scaled_test_df

# Removing duplicate columns

df_test = df_test.loc[:,~df_test.columns.duplicated()]



df_test.head(3)
y_test = pd.read_csv('../input/sample_submission.csv')[['SalePrice']]

#y_test = scaler.fit_transform(y_test[['SalePrice']])

X_test = df_test

X_test.shape
# adapting the test set to training set

#missing = list(set(X.columns.tolist()) - set(X_test.columns.tolist()))

#print(missing)

#X_test = pd.concat([X_test,pd.DataFrame(columns=difference)])

#surplus = list(set(X_test.columns.tolist()) - set(X.columns.tolist()))

#X_test = X_test.drop(surplus, axis=1)

print(X.shape, X_test.shape)
# Predicting with Multiple Linear Regression

yhat = lm.predict(X_test)

lm.score(X_test,y_test.values.reshape(-1,1))

yhat