import numpy as np

import pandas as pd

import sqlite3

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

import statsmodels.api as sm

from matplotlib import pyplot as plt

from imblearn.over_sampling import SMOTE

import os

from IPython.display import FileLink

import xgboost as xgb
# Upload the information from database and into the Data Frame

con = sqlite3.connect("/kaggle/input/lending-club-loan-data/database.sqlite")

df = pd.read_sql_query("SELECT loan_amnt, emp_length, home_ownership, annual_inc ,purpose, dti, loan_status from loan", con)
# Check the structure of the data

print(df.dtypes)



# Check the first five rows of the data

print(df.head(5))
# Convert numeric fields to floats and int

df['loan_amnt'] = pd.to_numeric(df['loan_amnt'])

df['annual_inc'] = pd.to_numeric(df['annual_inc'])

df['dti'] = pd.to_numeric(df['dti'])

# Check the structure of the data

print(df.dtypes)
# Distribution of loan amounts

print(df['loan_amnt'].mean())

print(df['loan_amnt'].median())

print(df['loan_amnt'].min())

print(df['loan_amnt'].max())
# Distribution of dti

print(df['dti'].mean())

print(df['dti'].median())

print(df['dti'].min())

print(df['dti'].max())
# Distribution of annual income

print(df['annual_inc'].mean())

print(df['annual_inc'].median())

print(df['annual_inc'].min())

print(df['annual_inc'].max())
# Distribution of loan status

df['loan_status'].value_counts()
# Distribution of employee length

df['emp_length'].value_counts()
# Distribution of purpose

df['purpose'].value_counts()
# Distribution of home ownership

df['home_ownership'].value_counts()
# Drop the Outliers (they could be replaced by mean. Since we have lots of data, we can dispose of them)

# Dti

indices = df[df['dti'] > 100].index

df = df.drop(indices)

indices = df[df['dti'] < 0].index

df = df.drop(indices)

# Annual Income

indices = df[df['annual_inc'] > 1000000].index #This value can be reviewed if it doesn't make sense

df = df.drop(indices)

indices = df[df['annual_inc'] < 1000].index

df = df.drop(indices)
#Check for missing data

print(df['dti'].isnull().sum())

print(df['annual_inc'].isnull().sum())

print(df['loan_amnt'].isnull().sum())

print(df['purpose'].isnull().sum())

print(df['loan_status'].isnull().sum())

print(df['emp_length'].isnull().sum())

print(df['home_ownership'].isnull().sum())
#Drop corrupt/unuseful data

indices = df[df['emp_length'] == 'n/a'].index

df = df.drop(indices)

indices = df[df['home_ownership'] == 'ANY'].index

df = df.drop(indices)

indices = df[df['home_ownership'] == 'NONE'].index

df = df.drop(indices)

indices = df[df['home_ownership'] == 'OTHER'].index

df = df.drop(indices)

indices = df[df['purpose'] == 'other'].index

df = df.drop(indices)

indices = df[df['annual_inc'].isnull()].index

df = df.drop(indices)
# Select only terminal loans (Default/Charged Off and Fully Paid) and replace for 1 and 0

df = df[df.loan_status != 'Current']

df = df[df.loan_status != 'In Grace Period']

df = df[df.loan_status != 'Late (16-30 days)']

df = df[df.loan_status != 'Late (31-120 days)']

df = df[df.loan_status != 'Does not meet the credit policy. Status:Fully Paid']

df = df[df.loan_status != 'Does not meet the credit policy. Status:Charged Off']

df = df[df.loan_status != 'Issued']

df['loan_status'] = df['loan_status'].replace({'Charged Off':'Default'})

df['loan_status'] = df['loan_status'].replace({'Default':1})

df['loan_status'] = df['loan_status'].replace({'Fully Paid':0})

df['loan_status'].value_counts()
# Cross table of the purpose and status

print(pd.crosstab(df['purpose'], df['loan_status'], margins = True))
# Cross table of home ownership, loan status, and employee length

print(pd.crosstab(df['emp_length'],[df['loan_status'],df['home_ownership']]))
# Cross table of home ownership, loan status, and average debt to income

print(pd.crosstab(df['home_ownership'], df['loan_status'],

                  values=df['dti'], aggfunc='mean'))
# Box plot of debt to income by loan status

df.boxplot(column = ['dti'], by = 'loan_status')

plt.title('Average Debt to Income by Loan Status')

plt.suptitle('')

plt.show()
# Set up bins for numeric variables

amnt_bins = [0,500,1000,10000,20000,30000,40000]

df['loan_amnt_binned'] = pd.cut(df['loan_amnt'],amnt_bins)

income_bins = [0,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,500000,1000000]

df['annual_inc_binned'] = pd.cut(df['annual_inc'],income_bins)

dti_bins = [0,10,20,30,40,50,60,70,80,90,100]

df['dti_binned'] = pd.cut(df['dti'],dti_bins)
# Drop continuous columns for parsimony

del df['loan_amnt']

del df['annual_inc']

del df['dti']
# Create Dummy Variables

df_num = df.select_dtypes(exclude=['object','category'])

df_str = df.select_dtypes(include=['object','category'])



df_onehot = pd.get_dummies(df_str)



df_prep = pd.concat([df_num, df_onehot], axis=1)
# Function to clean infite and null values

def clean_dataset(df):

    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    df.dropna(inplace=True)

    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    return df[indices_to_keep]
# WOE calculation

df_att = df.drop('loan_status',axis=1)

df_WOE = pd.DataFrame(columns=['Attribute','WOE'])

df_IV = pd.DataFrame(columns=['Variable','IV'])



for col in df_att.columns:

    Group = df_att[col].value_counts()

    df2 = pd.DataFrame(Group)

    df2 = df2.reset_index()

    df2.columns = ['Group', 'counts']



    Bad = pd.DataFrame (df_att[col][df.loan_status == 1].value_counts())

    Bad = Bad.reset_index()

    Bad.columns = ['Group', 'bad']



    df2['bad'] = Bad['bad']

    df2['good'] = df2['counts']-df2['bad']

    df2['total_distri'] = df2.counts/sum(df2.counts)

    df2['bad_distri'] = df2.bad/sum(df2.bad)

    df2['good_distri'] = df2.good/sum(df2.good)

    df2['WOE'] = np.log(df2.good_distri / df2.bad_distri)*100

    df2 = clean_dataset(df2)

    IV = sum ((df2.good_distri - df2.bad_distri)*(df2.WOE/100))

    df_IV = df_IV.append({'Variable':col,'IV':IV},ignore_index=True)

    df3 = df2[['Group','WOE']]

    df3['Group'] = df3['Group'].astype(str)

    

    for ind in df3.index:

        df_WOE = df_WOE.append({'Attribute': str(col)+'_'+str(df3['Group'].loc[ind]),'WOE':df3['WOE'].loc[ind]}, ignore_index=True)
# Replacing 1 for WOE in data frame

for i in range (0,54):

        df_prep[str(df_WOE.iloc[i,0])] = df_prep[str(df_WOE.iloc[i,0])].replace(1,df_WOE.iloc[i,1])
# Logistic Regression

y = df_prep['loan_status']

X = df_prep.drop('loan_status',axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

df_logistic = LogisticRegression(solver='lbfgs',max_iter=1000).fit(X_train, np.ravel(y_train))
# Predicting Probability of Default and finding max and min values

preds = df_logistic.predict_proba(X_test)

preds_df = pd.DataFrame(preds[:,1], columns = ['prob_default'])

print (preds_df['prob_default'].max())

print (preds_df['prob_default'].min())
# Default Probabilities for Bins

preds_df.sort_values(by=['prob_default'])

preds_df['Bins'] = pd.cut(preds_df['prob_default'],10)

preds_df_avg = pd.DataFrame(preds_df.groupby(['Bins']).mean())

print (preds_df_avg)
# Model Performance

preds = df_logistic.predict_proba(X_test)



preds_df = pd.DataFrame(preds[:,1], columns = ['prob_default'])



preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.50 else 0) # This threshold will be examined later



print(preds_df['loan_status'].value_counts())



target_names = ['Non-Default', 'Default']

print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))
# Accuracy score of the model

print(df_logistic.score(X_test, y_test))
# ROC curve of the probabilities of default

prob_default = preds[:, 1]

fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)

plt.plot(fallout, sensitivity, color = 'darkorange')

plt.plot([0, 1], [0, 1], linestyle='--')

plt.show()
# AUC

auc = roc_auc_score(y_test, prob_default)

print (auc)
# Confusion Matrices

thresholds = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55]

for i in thresholds:

    preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > i else 0)

    print (i)

    print(confusion_matrix(y_test,preds_df['loan_status']))
# Threshold Plot

default_recall = []

nondefault_recall = []

accuracy = []

for i in thresholds:

    preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > i else 0)

    default_recall.append(precision_recall_fscore_support(y_test,preds_df['loan_status'])[1][1])

    nondefault_recall.append(precision_recall_fscore_support(y_test,preds_df['loan_status'])[0][1])

    report = classification_report(y_test,preds_df['loan_status'],output_dict=True)

    accuracy.append(report['accuracy'])

plt.plot(thresholds,default_recall)

plt.plot(thresholds,nondefault_recall)

plt.plot(thresholds,accuracy)

plt.xlabel("Probability Threshold")

plt.legend(["Default Recall","Non-default Recall","Model Accuracy"])

plt.show()
# 0.20 is the probability that better fits this model

preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.20 else 0)

print(preds_df['loan_status'].value_counts())



target_names = ['Non-Default', 'Default']

print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))
# Balancing Proportions of Defaults and Non Defaults

X_y_train = pd.concat([X_train.reset_index(drop = True),

                       y_train.reset_index(drop = True)], axis = 1)

count_nondefault, count_default = X_y_train['loan_status'].value_counts()



nondefaults = X_y_train[X_y_train['loan_status'] == 0]

defaults = X_y_train[X_y_train['loan_status'] == 1]



nondefaults_under = nondefaults.sample(count_default)



X_y_train_under = pd.concat([nondefaults_under.reset_index(drop = True),

                             defaults.reset_index(drop = True)], axis = 0)



print(X_y_train_under['loan_status'].value_counts())
# Logistic Regression

y_train_under = X_y_train_under['loan_status']

X_train_under = X_y_train_under.drop('loan_status',axis=1)



df_logistic_under = LogisticRegression(solver='lbfgs',max_iter=1000).fit(X_train_under, np.ravel(y_train_under))



print (df_logistic_under.intercept_)
# Predicting Probability of Default and finding max and min values

preds_under = df_logistic_under.predict_proba(X_test)

preds_df_under = pd.DataFrame(preds_under[:,1], columns = ['prob_default'])

print (preds_df_under['prob_default'].max())

print (preds_df_under['prob_default'].min())
# Default Probabilities for Bins

preds_df_under.sort_values(by=['prob_default'])

preds_df_under['Bins'] = pd.cut(preds_df_under['prob_default'],10)

preds_df_under_avg = pd.DataFrame(preds_df_under.groupby(['Bins']).mean())

print (preds_df_under_avg)
# IV Analysis to decide which predictors to keep

print (df_IV)
# We must keep those superior to 0.02, so employee length and home ownership cannot be used

df_wo_IV = df.drop(['emp_length','home_ownership'],axis=1)
# Create Dummy Variables

df_num_IV = df_wo_IV.select_dtypes(exclude=['object','category'])

df_str_IV = df_wo_IV.select_dtypes(include=['object','category'])



df_onehot_IV = pd.get_dummies(df_str_IV)



df_prep_IV = pd.concat([df_num_IV, df_onehot_IV], axis=1)
# Replacing 1 for WOE in data frame (only values for the remaining variables)

for i in range (14,53):

        df_prep_IV[str(df_WOE.iloc[i,0])] = df_prep_IV[str(df_WOE.iloc[i,0])].replace(1,df_WOE.iloc[i,1])
# Logistic Regression

y_IV = df_prep_IV['loan_status']

X_IV = df_prep_IV.drop('loan_status',axis=1)



X_train_IV, X_test_IV, y_train_IV, y_test_IV = train_test_split(X_IV, y_IV, test_size=0.3, random_state=123)

df_logistic_IV = LogisticRegression(solver='lbfgs',max_iter=1000).fit(X_train_IV, np.ravel(y_train_IV))



print (df_logistic_IV.intercept_)
# Predicting Probability of Default and finding max and min values

preds_IV = df_logistic_IV.predict_proba(X_test_IV)

preds_df_IV = pd.DataFrame(preds_IV[:,1], columns = ['prob_default'])

print (preds_df_IV['prob_default'].max())

print (preds_df_IV['prob_default'].min())
# Default Probabilities for Bins

preds_df_IV.sort_values(by=['prob_default'])

preds_df_IV['Bins'] = pd.cut(preds_df_IV['prob_default'],10)

preds_df_IV_avg = pd.DataFrame(preds_df_IV.groupby(['Bins']).mean())

print (preds_df_IV_avg)
# Original Sample

X_train_gbt = X_train

y_train_gbt = y_train

X_test_gbt = X_test

y_test_gbt = y_test
# Replace ] by ) since XGBoost has problems with ], < and ,

for col in X_train_gbt.columns:

    X_train_gbt = X_train_gbt.rename({col:col.replace("]",")")}, axis=1)

    X_train_gbt = X_train_gbt.rename({col:col.replace(","," ")}, axis=1)

    X_train_gbt = X_train_gbt.rename({col:col.replace("<","less than")}, axis=1)

for col in X_test_gbt.columns:

    X_test_gbt = X_test_gbt.rename({col:col.replace("]",")")}, axis=1)

    X_test_gbt = X_test_gbt.rename({col:col.replace(","," ")}, axis=1)

    X_test_gbt = X_test_gbt.rename({col:col.replace("<","less than")}, axis=1)
# Model Training

gbt = xgb.XGBClassifier(booster='gblinear',

                      objective='binary:logistic', 

                      n_estimators=100,

                      max_depth=6).fit(X_train_gbt, np.ravel(y_train_gbt))
# Predicting Probability of Default

gbt_preds = gbt.predict_proba(X_test_gbt)

preds_gbt_df = pd.DataFrame(gbt_preds[:,1], columns = ['prob_default'])

print(preds_gbt_df['prob_default'].min())

print(preds_gbt_df['prob_default'].max()) 
# Default Probabilities for Bins

preds_gbt_df.sort_values(by=['prob_default'])

preds_gbt_df['Bins'] = pd.cut(preds_gbt_df['prob_default'],10)

preds_gbt_df_avg = pd.DataFrame(preds_gbt_df.groupby(['Bins']).mean())

print (preds_gbt_df_avg)
# Undersampled sets

X_train_gbt_under = X_train_under

y_train_gbt_under = y_train_under
# Replace ] by ) since XGBoost has problems with ], < and ,

for col in X_train_gbt_under.columns:

    X_train_gbt_under = X_train_gbt_under.rename({col:col.replace("]",")")}, axis=1)

    X_train_gbt_under = X_train_gbt_under.rename({col:col.replace(","," ")}, axis=1)

    X_train_gbt_under = X_train_gbt_under.rename({col:col.replace("<","less than")}, axis=1)
# Model Training

gbt_under = xgb.XGBClassifier(booster='gblinear',

                      objective='binary:logistic', 

                      n_estimators=100,

                      max_depth=6).fit(X_train_gbt_under, np.ravel(y_train_gbt_under))
# Predicting Probability of Default

gbt_under_preds = gbt_under.predict_proba(X_test_gbt)

preds_gbt_under_df = pd.DataFrame(gbt_under_preds[:,1], columns = ['prob_default'])

print(preds_gbt_under_df['prob_default'].min())

print(preds_gbt_under_df['prob_default'].max())
# Default Probabilities for Bins

preds_gbt_under_df.sort_values(by=['prob_default'])

preds_gbt_under_df['Bins'] = pd.cut(preds_gbt_under_df['prob_default'],10)

preds_gbt_under_df_avg = pd.DataFrame(preds_gbt_under_df.groupby(['Bins']).mean())

print (preds_gbt_under_df_avg)
# IV sets

X_train_gbt_IV = X_train_IV

y_train_gbt_IV = y_train_IV

X_test_gbt_IV = X_test_IV

y_test_gbt_IV = y_test_IV
# Replace ] by ) since XGBoost has problems with ] and ,

for col in X_train_gbt_IV.columns:

    X_train_gbt_IV = X_train_gbt_IV.rename({col:col.replace("]",")")}, axis=1)

    X_train_gbt_IV = X_train_gbt_IV.rename({col:col.replace(","," ")}, axis=1)

for col in X_test_gbt_IV.columns:

    X_test_gbt_IV = X_test_gbt_IV.rename({col:col.replace("]",")")}, axis=1)

    X_test_gbt_IV = X_test_gbt_IV.rename({col:col.replace(","," ")}, axis=1)
# Model Training

gbt_IV = xgb.XGBClassifier(booster='gblinear',

                      objective='binary:logistic', 

                      n_estimators=100,

                      max_depth=6).fit(X_train_gbt_IV, np.ravel(y_train_gbt_IV))
# Predicting Probability of Default

gbt_preds_IV = gbt_IV.predict_proba(X_test_gbt_IV)

preds_gbt_IV_df = pd.DataFrame(gbt_preds_IV[:,1], columns = ['prob_default'])

print(preds_gbt_IV_df['prob_default'].min())

print(preds_gbt_IV_df['prob_default'].max()) 
# Default Probabilities for Bins

preds_gbt_IV_df.sort_values(by=['prob_default'])

preds_gbt_IV_df['Bins'] = pd.cut(preds_gbt_IV_df['prob_default'],10)

preds_gbt_IV_df_avg = pd.DataFrame(preds_gbt_IV_df.groupby(['Bins']).mean())

print (preds_gbt_IV_df_avg)
# LogReg

df_columns = pd.DataFrame (list (X.columns),columns=['Attribute'])

df_coef = pd.DataFrame(df_logistic.coef_).transpose()

df_attr_coeff = pd.concat([df_columns, df_coef], axis=1, ignore_index=True)

df_attr_coeff.columns = ['Attribute','Coeff']

df_attr_coeff.head(5)

df_final = pd.merge(df_WOE, df_attr_coeff, left_on='Attribute', right_on='Attribute', how='left', sort=False)

df_final.to_csv('Credit_Scorecard.csv', sep=';')

print (df_logistic.intercept_)
# LogReg Under

df_columns_under = pd.DataFrame (list (X_train.columns),columns=['Attribute'])

df_coef_under = pd.DataFrame(df_logistic_under.coef_).transpose()

df_attr_coeff_under = pd.concat([df_columns_under, df_coef_under], axis=1, ignore_index=True)

df_attr_coeff_under.columns = ['Attribute','Coeff']

df_final_under = pd.merge(df_WOE, df_attr_coeff_under, left_on='Attribute', right_on='Attribute', how='left', sort=False)

df_final_under.to_csv('Credit_Scorecard_under.csv', sep=';')

print (df_logistic_under.intercept_)
# LogReg IV

df_columns_IV = pd.DataFrame (list (X_train_IV.columns),columns=['Attribute'])

df_coef_IV = pd.DataFrame(df_logistic_IV.coef_).transpose()

df_attr_coeff_IV = pd.concat([df_columns_IV, df_coef_IV], axis=1, ignore_index=True)

df_attr_coeff_IV.columns = ['Attribute','Coeff']

df_final_IV = pd.merge(df_WOE, df_attr_coeff_IV, left_on='Attribute', right_on='Attribute', how='inner', sort=False)

df_final_IV.to_csv('Credit_Scorecard_IV.csv', sep=';')

print (df_logistic_IV.intercept_)
# XGBoost

df_columns_gbt = pd.DataFrame (list (X_train.columns),columns=['Attribute']) # Columns in gbt set were altered

df_coef_gbt = pd.DataFrame(gbt.coef_)

df_attr_coeff_gbt = pd.concat([df_columns_gbt, df_coef_gbt], axis=1, ignore_index=True)

df_attr_coeff_gbt.columns = ['Attribute','Coeff']

df_final_gbt = pd.merge(df_WOE, df_attr_coeff_gbt, left_on='Attribute', right_on='Attribute', how='inner', sort=False)

df_final_gbt.to_csv('Credit_Scorecard_gbt.csv', sep=';')

print (gbt.intercept_)
# XGBoost Under

df_columns_gbt_under = pd.DataFrame (list (X_train.columns),columns=['Attribute']) # Columns in gbt set were altered

df_coef_gbt_under = pd.DataFrame(gbt_under.coef_)

df_attr_coeff_gbt_under = pd.concat([df_columns_gbt_under, df_coef_gbt_under], axis=1, ignore_index=True)

df_attr_coeff_gbt_under.columns = ['Attribute','Coeff']

df_final_gbt_under = pd.merge(df_WOE, df_attr_coeff_gbt_under, left_on='Attribute', right_on='Attribute', how='inner', sort=False)

df_final_gbt_under.to_csv('Credit_Scorecard_gbt_under.csv', sep=';')

print (gbt_under.intercept_)
# XGBoost IV

df_columns_gbt_IV = pd.DataFrame (list (X_train_IV.columns),columns=['Attribute']) # Columns in gbt set were altered

df_coef_gbt_IV = pd.DataFrame(gbt_IV.coef_)

df_attr_coeff_gbt_IV = pd.concat([df_columns_gbt_IV, df_coef_gbt_IV], axis=1, ignore_index=True)

df_attr_coeff_gbt_IV.columns = ['Attribute','Coeff']

df_final_gbt_IV = pd.merge(df_WOE, df_attr_coeff_gbt_IV, left_on='Attribute', right_on='Attribute', how='inner', sort=False)

df_final_gbt_IV.to_csv('Credit_Scorecard_gbt_IV.csv', sep=';')

print (gbt_IV.intercept_)