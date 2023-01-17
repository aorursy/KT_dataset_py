# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
# Import the numpy and pandas packages

import numpy as np
import pandas as pd
# Import the matplotlib.pyplot and seaborn packages

import matplotlib.pyplot as plt
import seaborn as sns
# Import the IPython interactiveshell packages

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#To display maximum 150 rows/columns

pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 150)
pd.set_option('display.width', 1000)
#Load / Import Primary file provided as Application_data.csv
pm1 = pd.read_csv("../input/loan-defaulter/application_data.csv")
pm1.head()
#Check the routine structre of data part 1
pm1.shape
#Check the routine structre of data part 2
pm1.info()
#Check the routine structre of data part 3
pm1.describe()
# code for column-wise null count here
print(pm1.isnull().sum(axis=0))
# Code for column-wise null percentages here
x = pm1.isnull().mean().round(4) * 100
print (x)
#Drop columns with more than 50% null values

null_ptg = pm1.isnull().sum() / len(pm1)

missing_features = null_ptg[null_ptg > 0.50].index

print("Columns With more than 50% of null values: \n \t \t",(missing_features))

pm1.drop(missing_features, axis=1, inplace=True)
#Revised DataFrame after dropping more than 50% null value columns
print("Shape of the dataframe after droping columns with more than 50% null values")
pm1.shape
#Checking null Values post data modification
pm1.isnull().mean().round(4) * 100
#Check Data description post column removal
pm1[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].describe()
pm2 = pm1[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']]

for column in pm2:
    plt.figure()
    pm2.boxplot([column])


pm1.info()
#Check the datatypes and change it like negative and date
pm1.DAYS_BIRTH = abs(pm1.DAYS_BIRTH)
pm1.DAYS_EMPLOYED = abs(pm1.DAYS_EMPLOYED)
pm1.DAYS_ID_PUBLISH = abs(pm1.DAYS_ID_PUBLISH)
pm1.DAYS_REGISTRATION = abs(pm1.DAYS_REGISTRATION)

# Deriving the age column from the DAYS_BIRTH column
pm1['age'] = (pm1['DAYS_BIRTH']/365).round(2)
pm1[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_GOODS_PRICE','DAYS_BIRTH','DAYS_EMPLOYED']].describe()
#Finding Numeric Outliers

pm1.boxplot(column=['AMT_INCOME_TOTAL'], return_type='axes');

plt.figure()

pm1.boxplot(column=['AMT_CREDIT'], return_type='axes');

plt.figure()

pm1.boxplot(column=['AMT_GOODS_PRICE'], return_type='axes');

plt.figure()

pm1.boxplot(column=['DAYS_BIRTH'], return_type='axes');

plt.figure()

pm1.boxplot(column=['DAYS_EMPLOYED'], return_type='axes');

plt.figure()
pm1.head()
# calculating the min max values of age column for binning

min(pm1['age'])

max(pm1['age'])
#Binning the age column into age groups
pm1['age_Groups'] = pd.cut(x=pm1['age'], bins=[20, 29, 39, 49,59,70],labels=['20-29', '30-39', '40-49','50-59', '60-69'])
#Binning the age column into age decades
pm1['age_by_decade'] = pd.cut(x=pm1['age'], bins=[20, 29, 39, 49,59,70], labels=['20s', '30s', '40s', '50s', '60s'])
pm1.head()
#Binning the AMT_INCOME_TOTAL column into income groups

pm1['AMT_INCOME_GROUPS'] = pd.cut(x=pm1['AMT_INCOME_TOTAL'], bins=[0, 50000, 75000, 100000,150000,500000000],labels=['Low', 'Avg', 'Above-Avg','High', 'Highest'])
pm1.head()
# Removing unnecessary columns from the application data pm1 dataframe

pm1 = pm1.drop(['REGION_POPULATION_RELATIVE','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','EXT_SOURCE_2','EXT_SOURCE_3',
                'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY',
                'YEARS_BEGINEXPLUATATION_AVG','FLOORSMAX_AVG','YEARS_BEGINEXPLUATATION_MODE','FLOORSMAX_MODE','YEARS_BEGINEXPLUATATION_MEDI','FLOORSMAX_MEDI','TOTALAREA_MODE','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE',
                'OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10',
                'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21',
                'AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR','EMERGENCYSTATE_MODE'
               ],axis = 1)

pm1.shape

pm1.head()

#Checking the imbalance percentage in TARGET column
(pm1['TARGET'].value_counts()/pm1['TARGET'].count()).round(4)*100

#Divide the data set into two columns
#pm_0 is a dataframe which contains the values of TARGET Column with values equal to 0
#pm_1 is a dataframe which contains the values of TARGET Column with values equal to 1

pm_0 = pm1[(pm1['TARGET'] == 0)]
pm_1 = pm1[(pm1['TARGET'] == 1)]

#For checking the values of TARGET Column in both pm_0 and pm_1 dataframes
pm_0['TARGET'].value_counts()
pm_1['TARGET'].value_counts()


pm_0.head()
pm_1.head()

#Univariate analysis for Catogorical variables for 0 and 1

features = ['NAME_CONTRACT_TYPE' ,'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE','NAME_HOUSING_TYPE']
plt.figure(figsize = (15, 57))
for i in enumerate(features):
    plt.subplot(4, 2, i[0]+1)
    sns.countplot(x = i[1], hue = 'TARGET', data = pm1)
    plt.xticks(rotation = 90)
# 

features = ['NAME_CONTRACT_TYPE' ,'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE']

plt.figure(figsize = (15,55))
for i in enumerate(features):
    plt.subplot(4, 2, i[0]+1)
    df = pm1[[i[1], 'TARGET']].groupby([i[1]],as_index=False).mean()
    df
    sns.barplot(x = i[1], y='TARGET', data=df)
    plt.ylabel('defaulters percentage', fontsize=10)
    plt.xticks(rotation = 90) 
pm_0.corr().unstack().reset_index()
# Correlation For numerical columns for target value 0 dataframe

corr = pm_0.corr()

corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
corrdf0 = corr.unstack().reset_index()
corrdf0.columns = ['VAR1', 'VAR2', 'Correlation']
corrdf0.dropna(subset = ['Correlation'], inplace = True)
corrdf0['Correlation'] = round(corrdf0['Correlation'], 2)
corrdf0['Correlation_abs'] = corrdf0['Correlation'].abs()
corrdf0.sort_values(by = 'Correlation_abs', ascending = False).head(10)


plt.figure(figsize=(40,15))

sns.heatmap(corr,annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=2, linecolor='black',square=True)


# Correlation For numerical columns for target value 1 dataframe

corr = pm_1.corr()

corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
corrdf1 = corr.unstack().reset_index()
corrdf1.columns = ['VAR1', 'VAR2', 'Correlation']
corrdf1.dropna(subset = ['Correlation'], inplace = True)
corrdf1['Correlation'] = round(corrdf1['Correlation'], 2)
corrdf1['Correlation_abs'] = corrdf1['Correlation'].abs()
corrdf1.sort_values(by = 'Correlation_abs', ascending = False).head(10)

plt.figure(figsize=(40,15))

sns.heatmap(corr,annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=2, linecolor='black',square=True)

#Univariate analysis for Numerical variables for 0 and 1

sns.distplot(pm_0['DAYS_BIRTH'], hist=True, color = 'aqua', label='TARGET = 0')
sns.distplot(pm_1['DAYS_BIRTH'], hist=True, color = 'orange', label='TARGET = 1')


print("People who are around",int(10000/365),"years to",int(12000/365),"years have highest defaulters")
#Univariate analysis for Numerical variables for 0 and 1

sns.distplot(pm_0['DAYS_ID_PUBLISH'], hist=False, color = 'aqua', label='TARGET = 0')
sns.distplot(pm_1['DAYS_ID_PUBLISH'], hist=False, color = 'orange', label='TARGET = 1')

print("People who have updated their identity documents recently before an year or two have highest defaulters ratio")
#Univariate analysis for Numerical variables for 0 and 1

sns.distplot(pm_0['DAYS_REGISTRATION'], hist=False, color = 'aqua', label='TARGET = 0')
sns.distplot(pm_1['DAYS_REGISTRATION'], hist=False, color = 'orange', label='TARGET = 1')

print("People who have updated their registration recently before 1-3 years have highest defaulters ratio")
#Bivariate analysis on numerical columns with respective 0's and 1's

# Bivariate Analysis graph for age vs AMT_INCOME_TOTAL 


sns.jointplot(x="age", y="AMT_INCOME_TOTAL", data=pm_0);#TARGET = 0
sns.jointplot(x="age", y="AMT_INCOME_TOTAL", data=pm_1);#TARGET = 1

print("The age vs AMT_INCOME_TOTAL for defaulters most of them lied around 25-32 years and income of around 50k to 130K")

# Bivariate Analysis graph for age vs AMT_CREDIT 

sns.jointplot(x="age", y="AMT_CREDIT", data=pm_0);#TARGET = 0
sns.jointplot(x="age", y="AMT_CREDIT", data=pm_1);#TARGET = 1

print("The age vs AMT_CREDIT for defaulters most of them lied around 25-32 years and amount credit of around 250k to 650K")

# Analysis graph for age vs AMT_GOODS_PRICE 
sns.jointplot(x="age", y="AMT_GOODS_PRICE", data=pm_0);#TARGET = 0
sns.jointplot(x="age", y="AMT_GOODS_PRICE", data=pm_1);#TARGET = 1

print("The age vs AMT_GOODS_PRICE for defaulters most of them lied around 25-32 years and goods price of around 200k to 650K")

sns.jointplot(x="AMT_INCOME_TOTAL", y="AMT_CREDIT", data=pm_0);#TARGET = 0
sns.jointplot(x="AMT_INCOME_TOTAL", y="AMT_CREDIT", data=pm_1);#TARGET = 1
print("The AMT_INCOME_TOTAL vs AMT_CREDIT for defaulters most of them lied around 100k income and amount credit of around 500k")

sns.jointplot(x="AMT_CREDIT", y="AMT_GOODS_PRICE", data=pm_0);#TARGET = 0
sns.jointplot(x="AMT_CREDIT", y="AMT_GOODS_PRICE", data=pm_1);#TARGET = 1

print("AMT_CREDIT and AMT_GOODS_PRICE are almost proportional to each other for both 0 and 1")
# Reading Previous dataframe

prev_app_DF = pd.read_csv('../input/loan-defaulter/previous_application.csv')
prev_app_DF.head()
# merging application dataframe pm1 with previous dataframe prev_app_DF

FinalDF = pm1.merge(prev_app_DF, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner')

FinalDF.head()
print("The resulting dataframe `FinalDF` has ",FinalDF.shape[0]," rows and ", FinalDF.shape[1]," columns.")

FinalDF.shape

FinalDF.info()

FinalDF.describe()
x = FinalDF.isnull().mean().round(4) * 100
print (x)
FinalDF = FinalDF.drop(['CNT_FAM_MEMBERS','age_by_decade','SK_ID_PREV','WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','FLAG_LAST_APPL_PER_CONTRACT',
                        'NFLAG_LAST_APPL_IN_DAY','RATE_DOWN_PAYMENT','RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED','NAME_TYPE_SUITE_y','NAME_PORTFOLIO',
                        'CHANNEL_TYPE','SELLERPLACE_AREA','NAME_SELLER_INDUSTRY','CNT_PAYMENT','NAME_YIELD_GROUP','PRODUCT_COMBINATION','DAYS_FIRST_DRAWING',
                       'NAME_TYPE_SUITE_x','AMT_DOWN_PAYMENT','DAYS_DECISION','CODE_REJECT_REASON','NAME_PRODUCT_TYPE','DAYS_LAST_DUE_1ST_VERSION',
                        'OCCUPATION_TYPE','ORGANIZATION_TYPE','DAYS_FIRST_DUE','DAYS_LAST_DUE','AMT_ANNUITY_y','DAYS_TERMINATION','NFLAG_INSURED_ON_APPROVAL','DAYS_BIRTH','CNT_CHILDREN',
                       ],axis = 1)
FinalDF.head()
FinalDF.shape
# Univariate analysis for finding the patterns
features = ['NAME_CONTRACT_TYPE_y', 'NAME_CONTRACT_STATUS','NAME_CLIENT_TYPE','NAME_PAYMENT_TYPE', 'NAME_CASH_LOAN_PURPOSE']
plt.figure(figsize = (15,35))
for i in enumerate(features):
    plt.subplot(3, 2, i[0]+1)
    sns.countplot(x = i[1],  data = FinalDF)
    plt.xticks(rotation = 90)
# After merging the two dataframes the analysis for the FinalDF with respective TARGET column

features = ['NAME_CONTRACT_TYPE_y', 'NAME_CONTRACT_STATUS','NAME_CLIENT_TYPE','NAME_PAYMENT_TYPE', 'NAME_CASH_LOAN_PURPOSE']

plt.figure(figsize = (15,35))
for i in enumerate(features):
    plt.subplot(3, 2, i[0]+1)
    sns.countplot(x = i[1], hue = 'TARGET', data = FinalDF)
    plt.xticks(rotation = 90)
# For better understanding of the data let analyse FinalDF with respective the percentage of TARGET column with value 1.
features = ['NAME_CONTRACT_TYPE_y', 'NAME_CONTRACT_STATUS','NAME_CLIENT_TYPE','NAME_PAYMENT_TYPE', 'NAME_CASH_LOAN_PURPOSE']
    
plt.figure(figsize = (15,35))
for i in enumerate(features):
    plt.subplot(3, 2, i[0]+1)
    df = FinalDF[[i[1], 'TARGET']].groupby([i[1]],as_index=False).mean()
    df
    sns.barplot(x = i[1], y='TARGET', data=df)
    plt.ylabel('defaulters percentage', fontsize=10)
    plt.xticks(rotation = 90)   
