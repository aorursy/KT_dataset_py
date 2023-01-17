# Import common libraries

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

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection, model_selection, metrics

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV

from sklearn.metrics import mean_absolute_error, roc_curve, auc, confusion_matrix, plot_confusion_matrix



# Default Global settings

pd.set_option('max_columns', None)

#os.chdir("C:/Users/wturner/OneDrive - Macadamia Processing Co Limited/")



print("Setup Successful")
df = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')
# Work smarter not harder

# pandas_profiling.ProfileReport(df)
df.dtypes
# Get the list of columns

df.columns
# Change columns to correct data types

col_int = ['AppointmentID', 'PatientId'] # create a list of column names to convert to integer

col_float = ['Age'] # create a list of column names to convert to float

col_string = [] # create a list of column names to convert to string

col_ordinal = ['Scholarship',  'Hipertension',  'Diabetes',  'Alcoholism',  'Handcap', 'SMS_received',] # create a list of column names to convert to ordinal

col_nominal = ['Gender', 'Neighbourhood'] # create a list of column names to convert to nominal

col_date = ['ScheduledDay', 'AppointmentDay'] # create a list of column names to convert to date





def change_dtypes(col_int, col_float, col_string, col_ordinal, col_nominal, col_date, df): 

    '''

    AIM    -> Changing dtypes to save memory

    INPUT  -> List of int column names, float column names, df

    OUTPUT -> updated df with smaller memory  

    '''

    df[col_int] = df[col_int].apply(pd.to_numeric)

    df[col_float] = df[col_float].astype('float32')

    df[col_string] = str(df[col_string])

    df[col_ordinal] = df[col_ordinal].astype('object')

    df[col_nominal] = df[col_nominal].astype('object')

    for col in col_date:

        df[col] = pd.to_datetime(df[col])

    

change_dtypes(col_int, col_float, col_string, col_ordinal, col_nominal, col_date, df)
# Select the target variable and drop it from the df for data wrangling

df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})

target = df['No-show']

df.drop(columns=['No-show'], inplace=True)
# Normalising - binary classification unlikely to need normalising but its still good practice to examine the distribution



# Plot of a histogram of the Target variable

fig = go.Figure(data=[go.Histogram(x=target)])

fig.update_layout(title="Histogram of the target variable")

fig.show()
# Null Values



# Dropping observations with too many NA, and surfacing the observations possibly requiring some imputation



# If a row has at least 25% of its values as NA, then drop the row

threshold = len(df.columns) * .8

df = df.dropna(thresh=threshold)



nrows = df.shape[0]

null_cat_col = []

null_int_col = []



def nullimputer(df):

    '''

    AIM    -> Impute null values where required, drop columns with too many nulls

    INPUT  -> dataframe

    OUTPUT -> dataframe with no null values

    '''

    ### Possible improvement: columns with null between 15-30% surfaced with recommendation to feature engineer instead of dropping

    

    for col in df.columns:

        x = df[col].isnull().sum()

        numnull = x / nrows



        if numnull >= .15:

            print("Dropped column(s):",col)

            df.drop([col],axis=1, inplace=True) # If a column has at least 15% of its values as NA, then drop the column

        elif 0 < numnull < .15:

            if col in col_int:

                null_int_col.append(col)

            elif col in col_ordinal:

                null_cat_col.append(col)

            elif col in col_nominal:

                null_cat_col.append(col)

            else:

                print("Column with 0 to 15% null:",col) # If a column has 0 to 15% of its values as NA, display the name of the column

        else:

            continue #If a column has no null then continue





    print("Categorical columns with nulls:", null_cat_col)

    print("Numerical columns with nulls:", null_int_col)

    

    print("Imputing values where required...")

    for col in null_int_col:

        df[col] = df[col].fillna(df[col].median()) # Fill integer NA with median



    for col in null_cat_col:

        df[col] = df[col].fillna(df[col].mode()[0]) # Fill categorical NA with mode

    

    print("{} null values remain".format(df.isna().sum().sum()))

    

nullimputer(df)
# Number of days between the date of booking and the appointment date

df['days_until_appt'] = (df['ScheduledDay'] - df['AppointmentDay']).astype('timedelta64[D]')



# Drop the appt date columns now, unused as not doing time series analysis

df.drop(columns=['ScheduledDay', 'AppointmentDay'], inplace=True)
# Encoding ordinal variables

label = LabelEncoder()



for col in col_ordinal:

   df[col] = label.fit(df[col].values).transform(df[col].values)
# Encoding nominal variables

for col in col_nominal:

    df_dummies = pd.get_dummies(df[col])

    df = pd.concat([df, df_dummies], axis=1)

    df.drop(col, inplace=True, axis=1)
# Examining correlations with the target variable with the absolute value of pearson R correlation greater than or equal to:

corr_threshold = 0.1



corrdata = pd.concat([df,target],axis=1)

corr_matrix = abs(corrdata.corr())

corr_matrix_target = pd.DataFrame(corr_matrix["No-show"])



corr_matrix_target = corr_matrix_target[corr_matrix_target["No-show"] >= corr_threshold]

corr_matrix_target = corr_matrix_target[corr_matrix_target.index != "No-show"]







# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_matrix_target, cmap="Blues", vmax=corr_matrix_target.max(), vmin=corr_matrix_target.min(),

            square=True, linewidths=.2, cbar_kws={"shrink": .7})
# The strongest correlations between other variables

all_corr = df.corr().abs().unstack().sort_values(kind="quicksort")



all_corr[(all_corr > .3) & (all_corr != 1)].tail(10)
MLA = [

    # Ensemble Methods

    #ensemble.GradientBoostingClassifier(),

    #ensemble.RandomForestClassifier(),

    

    # Linear Models

    linear_model.LogisticRegressionCV(),

    

    # Decision Trees    

    tree.DecisionTreeClassifier(),



    # XGBoost

    XGBClassifier()  

    

    ]
# Create a dataframe for the model results

result_table = pd.DataFrame(columns=["fit_time","score_time","test_score","train_score","MLA"])



# Cross validate through the list of MLAs

for alg in MLA:

    result = cross_validate(alg, df, target, cv=5, return_train_score = True)

    result["MLA"] = alg

    resultdf = pd.DataFrame.from_dict(result)

    result_table = pd.concat([result_table, resultdf])
# Rearrange to put MLA column first and group MLAs by mean

cols = result_table.columns.tolist()

cols = cols[-1:] + cols[:-1]

result_table = result_table[cols]
result_table_copy = result_table

result_table_copy["MLA"] = result_table["MLA"].astype("string")
result_summary = result_table_copy.groupby(['MLA']).mean()



# Display the table

result_summary
result_summary = result_summary.set_index([pd.Index(["DecisionTree", "LogisticRegression", "XGBoost"])])
# Bar plot of the scoring results

sns.barplot(x=result_summary.index, y="test_score", data=result_summary)
xgb = XGBClassifier()

X_train, X_test, y_train, y_test = train_test_split(df, target, random_state = 0)

xgb = xgb.fit(X_train, y_train)
plt.subplots(figsize=(20, 25))

plt.barh(df.columns, xgb.feature_importances_)