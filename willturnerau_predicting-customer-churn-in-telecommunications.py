import sys # access to system parameters

print("Python version: {}". format(sys.version))



import pandas as pd # functions for data processing and analysis modeled after R dataframes with SQL like features

import pandas_profiling

print("pandas version: {}". format(pd.__version__))



import numpy as np # foundational package for scientific computing

print("NumPy version: {}". format(np.__version__))



import scipy as sp # collection of functions for scientific computing and advance mathematics

print("SciPy version: {}". format(sp.__version__)) 

import scipy.stats as ss



import sklearn # collection of machine learning algorithms

print("scikit-learn version: {}". format(sklearn.__version__))





#misc libraries

import random

import time

import datetime

import os

import glob

import math





# Visualisation

import matplotlib #collection of functions for scientific and publication-ready visualization

%matplotlib inline

import matplotlib.pyplot as plt

pd.plotting.register_matplotlib_converters()

print("matplotlib version: {}". format(matplotlib.__version__))

import plotly

print("plotly version: {}". format(plotly.__version__))

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot # Offline mode

init_notebook_mode(connected=True)

import seaborn as sns

from xgboost import plot_importance





# Import common MLA libraries

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier





#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn import feature_selection, model_selection, metrics

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV

from sklearn.metrics import mean_absolute_error, roc_curve, auc, confusion_matrix, plot_confusion_matrix, accuracy_score







# Default Global settings

pd.set_option('max_columns', None)

import warnings

warnings.filterwarnings("ignore")



print("Setup Successful")
# Import the data

data = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# Copy the data so it doesn't have to be reloaded in each time

df = data
# View the pandas profiling report to understand the variables better

#pandas_profiling.ProfileReport(df)
# Check for nulls

df.isnull().sum()
# Check for errors or bad data

df.sample(50)
# Drop ID as its not valuable to the model

df.drop(columns=["customerID"], inplace=True)
# View the data types

df.dtypes
# Gather a list of the column names

df.columns.tolist()
# Print out all unique values for each variable

for col in df.columns:

    print(col, ":", df[col].unique())
# Change columns to correct data types

col_int = [] # create a list of column names to convert to integer

col_float = ['TotalCharges'] # create a list of column names to convert to float

col_string = [] # create a list of column names to convert to string

col_ordinal = [] # create a list of column names to convert to ordinal

col_nominal = ['gender',

 'SeniorCitizen',

 'Partner',

 'Dependents',

 'PhoneService',

 'MultipleLines',

 'InternetService',

 'OnlineSecurity',

 'OnlineBackup',

 'DeviceProtection',

 'TechSupport',

 'StreamingTV',

 'StreamingMovies',

 'Contract',

 'PaperlessBilling',

 'PaymentMethod'] # create a list of column names to convert to nominal

col_numeric = ['TotalCharges', 'MonthlyCharges', 'tenure']

col_date = [] # create a list of column names to convert to date





def change_dtypes(col_int, col_float, col_string, col_ordinal, col_nominal, col_date, df): 

    '''

    AIM    -> Changing dtypes to save memory

    INPUT  -> List of int column names, float column names, df

    OUTPUT -> updated df with smaller memory  

    '''

    df[col_int] = df[col_int].apply(pd.to_numeric)

    df[col_string] = str(df[col_string])

    df[col_ordinal] = df[col_ordinal].astype('object')

    df[col_nominal] = df[col_nominal].astype('object')

    for col in col_date:

        df[col] = pd.to_datetime(df[col])

    

change_dtypes(col_int, col_float, col_string, col_ordinal, col_nominal, col_date, df)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# save the target variable

target = df["Churn"]

df.drop(columns=["Churn"], inplace=True)
# Distribution of the target variable

sns.countplot(x=target)
# Examining numeric correlations with the target variable with the absolute value of pearson R correlation



corrdata = pd.concat([df,target],axis=1)

corr = corrdata.corr()

sns.heatmap(corr, annot=True)
# Standardising numeric variables, labelencoding ordinal variabnles, one-hot encoding nominal variables

# No null values to be imputed

numerical_transformer = StandardScaler()

nominal_transformer = OneHotEncoder(handle_unknown='ignore')

ordinal_transformer = LabelEncoder()

preprocessor = ColumnTransformer(transformers=[

        ('num', numerical_transformer, col_numeric),

        ('ord', ordinal_transformer, col_ordinal),

        ('nom', nominal_transformer, col_nominal)],

        remainder='passthrough')
# Split dataset into train and test

X_train, X_test, y_train, y_test = train_test_split(df, target)
# Fit the pipeline on the training set and then transform the training and test set

preprocessor.fit(X_train)

X_train = preprocessor.transform(X_train)

X_train = pd.DataFrame(X_train)



X_test = preprocessor.transform(X_test)

X_test = pd.DataFrame(X_test)

X_train.columns.tolist()
preprocessor.named_transformers_['nom'].get_feature_names()
col_dict = dict(zip([

    0,1,2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,

 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], 

                    ['tenure', 'MonthlyCharges','TotalCharges','gender_Female', 'gender_Male', 'SeniorCitizen_0', 'SeniorCitizen_1', 'Partner_No', 'Partner_Yes', 'Dependents_No',

       'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes', 'MultipleLines_No', 'MultipleLines_No phone service',

       'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No',

       'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No',

       'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No',

       'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No',

       'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',

       'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No',

       'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_Month-to-month',

       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',

       'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',

       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'])

               )
X_train = X_train.rename(columns=col_dict)

X_test = X_test.rename(columns=col_dict)
# Encode the y variable

le = LabelEncoder()

le.fit(y_train)

y_train = le.transform(y_train)

y_train = pd.DataFrame(y_train)



y_test = le.transform(y_test)

y_test = pd.DataFrame(y_test)
# Handle NAs

X_train = X_train.fillna(0)

X_test = X_test.fillna(0)
# Create objects of classification algorithms



MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(random_state = 10),

    ensemble.BaggingClassifier(random_state = 10),

    ensemble.ExtraTreesClassifier(random_state = 10),

    ensemble.GradientBoostingClassifier(random_state = 10),

    ensemble.RandomForestClassifier(random_state = 10),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(random_state = 10),

    

    #GLM

    linear_model.LogisticRegressionCV(random_state = 10),

    linear_model.PassiveAggressiveClassifier(random_state = 10),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(random_state = 10),

    linear_model.Perceptron(random_state = 10),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True, random_state = 10),

    svm.NuSVC(probability=True, random_state = 10),

    svm.LinearSVC(random_state = 10),

    

    #Trees    

    tree.DecisionTreeClassifier(random_state = 10),

    tree.ExtraTreeClassifier(random_state = 10),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    

    #xgboost

    XGBClassifier(random_state = 10)    

    ]
# Create a dataframe for the model results

result_table = pd.DataFrame(columns=["MLA", "train_score", "test_score"])



row_index = 0



# Score each algorithm and add its training and test results to result_table

for alg in MLA:

    alg.fit(X_train, y_train)

    y_train_pred = alg.predict(X_train)

    y_pred = alg.predict(X_test)

    result_table.loc[row_index, 'train_score'] = accuracy_score(y_train, y_train_pred)

    result_table.loc[row_index, 'test_score'] = accuracy_score(y_test, y_pred)

    result_table.loc[row_index, 'MLA'] = alg



    #result["MLA"] = alg

    #result_table.append(row, ignore_index = True)

    

    row_index+=1

# Display the results table, sorted in descending order

result_table = result_table.sort_values(by="test_score", ascending=False)

result_table
# Cross validate the best performing algorithm - Ridge Classifier performed the best (with random_state = 10) with default paramaters. 

# We will Ridge Classifier this further as it fits quickly, performs well and has high interperability



# Typically I would do cross validation now, however RidgeClassifierCV has built in CV so its not required. I will do it below anyway as good practice.



RC = linear_model.RidgeClassifierCV()

RC.fit(X_train, y_train)

cv_results = cross_validate(RC, X_test, y_test, cv=5)

cv_results['test_score'].mean()
# Plot the confusion matrix as a percentage of the whole

cf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 

            fmt='.2%', cmap='Blues')
# To help the business focus its activities, visualise the variable coefficients to understand the drivers of churn

cols = X_train.columns.tolist()

coefs = RC.coef_

coefs = coefs[0].tolist()

#coef_dict = dict(zip([cols, coefs]))

coefdf = pd.DataFrame(coefs)

coefdf = coefdf.rename(columns={0:'coefficient'})

coefdf["variable"] = cols

coefdf = coefdf.sort_values("coefficient", ascending=False)







fig, ax = plt.subplots(figsize=(20, 5))

coefdf.plot(x="variable", y="coefficient", kind='bar', 

             ax=ax, legend=False)

plt.show()