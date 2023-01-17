import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

loan_data = pd.read_csv("../input/loan.csv",low_memory=False)
loan = loan_data
loan.info()
loan.head()
# approx 887379 rows,74 columns
loan.shape
loan.describe()
# inspect the structure etc.
print(loan.info(), "\n")
print(loan.shape)
loan.columns
# unique records in loan dataset
loan.nunique()
# identify the unique number of ids in loan dataset
len(loan.id.unique())
len(loan.member_id.unique())
# check if there are any duplciates or nulls on id, memberid columns because there are unique.
loan.id.notnull().sum()
loan.member_id.notnull().sum()
loan.duplicated('id').sum()
loan.duplicated('member_id').sum()
# # column-wise missing values 
loan.isnull().sum()
# summing up the missing values (column-wise) and displaying fraction of NaNs
round(100*(loan.isnull().sum()/len(loan.index)), 2)
loan = loan.dropna(axis='columns', how='all')
# summing up the missing values (column-wise) and displaying fraction of NaNs
round(100*(loan.isnull().sum()/len(loan.index)), 2)
loan.shape
# lists number of unique values by each column
loan.apply(lambda x: x.nunique())
# removing the columns which has single values like - NA, 0, 'f', 'n' etc.
unique = loan.nunique()
unique = unique[unique.values == 1]
loan.drop(labels = list(unique.index), axis =1, inplace=True)
print("So now we are left with",loan.shape ,"rows & columns.")
#' Url' and 'desc' columns may not be required for analysis, These can be removed. 
loan =loan.drop(['desc', 'url'],axis=1)
loan['issue_d'].head()
loan['issue_month'], loan['issue_year'] = loan['issue_d'].str.split('-', 1).str
#loan.replace({'int_rate':{'%':''}},regex=True,inplace = True)
loan['int_rate'].head()
#loan.replace({'revol_util':{'%':''}},regex=True,inplace = True)
loan['revol_util'].head()
# title is a title for the loan entered by the borrower. Set NA to empty string.
loan.loc[pd.isnull(loan['title'])] = ""
loan.loc[pd.isnull(loan['emp_title'])] = ""
#Columns 'term' contains char "months" which makes it non-numerical, so remove chars and make the column numeric
loan.replace({'term':{' months':''}},regex=True,inplace = True)
loan.replace({'term':{' ':''}},regex=True,inplace = True)
loan['term'].head()
# check the nulls by column wise
round(100*(loan.isnull().sum()/len(loan.index)), 2)
# lets count number of rows which has nulls.
loan['revol_util'].isna().sum()
numeric_columns = ['revol_util']
loan[numeric_columns] = loan[numeric_columns].apply(pd.to_numeric)
loan['revol_util'].fillna((loan['revol_util'].median()), inplace=True)
loan['revol_util'].isna().sum()
loan.replace({'emp_length':{' years':''}},regex=True,inplace = True)
loan.replace({'emp_length':{' year':''}},regex=True,inplace = True)
loan['emp_length'].head()
print(loan.emp_length.unique())
loan.emp_length.fillna('0',inplace=True)
loan.emp_length.replace(['0'],'n/a',inplace=True)
loan.emp_length.replace([''],'n/a',inplace=True)
print(loan.emp_length.unique())
loan['emp_length']= loan['emp_length'].apply(lambda x:x.zfill(2))
print(loan.emp_length.unique())
loan['emp_length'].head()
## Removed the xx from zip code column
loan.replace({'zip_code':{'xx':''}},regex=True,inplace = True)
#loan.zip_code.repalce(['xx'],'',inplace=True)
loan.zip_code.head()
# There are three possible loan scenarios/statuses: fully paid, current, charged-off.
# We are interested in identifying clients who default (charged-off status) so we can
# create an additional column to simplify the three statuses into a defaulted binary.
loan['defaulted'] = loan['loan_status'].apply(lambda x: 'True' if x == "Charged Off" else 'False')
loan.info()
numeric_columns = ['loan_amnt','funded_amnt','funded_amnt_inv','installment','annual_inc','total_pymnt','total_pymnt_inv','total_rec_prncp',
                    'total_rec_int','total_rec_late_fee','collection_recovery_fee','recoveries','dti']

loan[numeric_columns] = loan[numeric_columns].apply(pd.to_numeric)
loan.info()
loan['income_bin'] = round(loan['annual_inc'] / 10000, 0) * 10000
loan['loan_amnt_bin'] = round(loan['loan_amnt']/ 2500, 0) * 2500
loan['dti_bin'] = round(loan['dti'], 0)
loan['revol_util_bin'] = round(loan['revol_util'] / 10, 0) * 10
loan.info()
# Checking for outliers in the continuous variables
num_columns = loan[['annual_inc','funded_amnt','funded_amnt_inv','loan_amnt','total_pymnt']]
# Checking outliers at 25%,50%,75%,90%,95% and 99%
num_columns.describe(percentiles=[.25,.5,.75,.90,.95,.99])
# boxplot
sns.boxplot(y=loan['annual_inc'])
plt.title('annual_inc')
plt.show()
year_wise =loan.groupby(by= [loan.issue_year])[['loan_status']].count()
year_wise.rename(columns={"loan_status": "count"},inplace=True)
ax =year_wise.plot(figsize=(20,8))
year_wise.plot(kind='bar',figsize=(20,8),ax = ax)
plt.show()
plt.figure(figsize=(16,12))
sns.boxplot(data =loan, x='purpose', y='loan_amnt', hue ='loan_status')
plt.title('Purpose of Loan vs Loan Amount')
plt.show()
loan_correlation = loan.corr()
loan_correlation
f, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(loan_correlation, 
            xticklabels=loan_correlation.columns.values,
            yticklabels=loan_correlation.columns.values,cmap="YlGnBu",annot= True)
plt.show()
plt.figure(figsize=(12,8))
sns.barplot('issue_year', 'loan_amnt', data=loan, palette='tab10')
plt.title('Issuance of Loans', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average loan amount issued', fontsize=14)
palette = ["#3791D7", "#E01E1B"]
sns.barplot(x="issue_year", y="loan_amnt", hue="loan_status", data=loan, palette=palette, estimator=lambda x: len(x) / len(loan) * 100)
ax[1].set(ylabel="(%)")
fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(nrows=2, ncols=2, figsize=(14,6))

# Change the Palette types tomorrow!

sns.violinplot(x="grade", y="loan_amnt", data=loan, palette="Set2", ax=ax1 )
sns.violinplot(x="sub_grade", y="loan_amnt", data=loan, palette="Set2", ax=ax2)
sns.boxplot(x="grade", y="funded_amnt", data=loan, palette="Set2", ax=ax3)
sns.boxplot(x="sub_grade", y="funded_amnt_inv", data=loan, palette="Set2", ax=ax4)
fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(nrows=2, ncols=2, figsize=(14,6))
sns.boxplot(x="purpose", y="loan_amnt", data=loan, palette="Set2", ax=ax1)
sns.boxplot(x="home_ownership", y="loan_amnt", data=loan, palette="Set2", ax=ax2)
sns.boxplot(x="term", y="loan_amnt", data=loan, palette="Set2", ax=ax3)
sns.boxplot(x="verification_status", y="loan_amnt", data=loan, palette="Set2", ax=ax4)
#sns.boxplot(x="emp_length", y="loan_amnt", data=loan, palette="Set2", ax=ax4)

plt.figure(figsize=(15,9))
ax = sns.boxplot(y="emp_length", x="loan_amnt", data=loan)
ax = plt.xlabel('Loan Amount')
ax = plt.ylabel('Employee Length')
ax = plt.title('Loan Amount Vs Employee Length')
plt.figure(figsize=(10,6))

#pal = {"Good": "#6bad97", "Bad": "#d8617f"}

ax = sns.violinplot(x="loan_status", y="loan_amnt", data=loan)
ax = plt.xlabel('Loan status')
ax = plt.ylabel('Loan Amount')
ax = plt.title('Loan Amount Distribution')
