import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
Bank_Personal_Loan = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx','Data')
# overview of the dataset
Bank_Personal_Loan.info()
Bank_Personal_Loan.head()
# check the unique number value of features
Bank_Personal_Loan.nunique()
# dataset statistics information
Bank_Personal_Loan.describe().transpose()
# fill up the [experience < 0] as 0 year
Bank_Personal_Loan.loc[Bank_Personal_Loan['Experience'] < 0, 'Experience'] = 0 
Bank_Personal_Loan.Experience.describe()
# check missing data
Bank_Personal_Loan.isnull().sum()
# overview of the achievement from last marketing campaign
Bank_Personal_Loan.groupby(['Personal Loan']).size()
Personal_Loan_Percentage = Bank_Personal_Loan.groupby(['Personal Loan']).size()/Bank_Personal_Loan.groupby(['Personal Loan']).size().sum()
Personal_Loan_Percentage
# overview of correlation between features
Bank_Personal_Loan.corr().style.background_gradient(cmap='BuGn')
#find corrlation between other features and Personal Loan
Bank_Personal_Loan.corr()['Personal Loan']
#1. Correlation between income and Personal Loan
Income_Loan_Percentage = Bank_Personal_Loan.groupby('Personal Loan')['Income'].agg([np.mean,'count'])
Income_Loan_Percentage.rename(columns={'mean': 'Income_Mean','count':'Number of People'})
#2. Correlation between CC Avg and Personal Loan
CCAvg_Loan_Percentage = Bank_Personal_Loan.groupby('Personal Loan')['CCAvg'].agg([np.mean,'count'])
CCAvg_Loan_Percentage.rename(columns={'mean': 'CCAvg_Mean','count':'Number of People'})
#3.Correlation between CD_Acount and Personal Loan
CD_Account_Loan_Percentage = Bank_Personal_Loan.groupby('CD Account')['Personal Loan'].agg([np.mean,'count'])
CD_Account_Loan_Percentage.rename(columns={'mean': 'Personal_Loan_Mean','count':'Number of People'})

#4 Correlation between Education and Personal Loan
Education_Loan_Percentage = Bank_Personal_Loan.groupby('Education')['Personal Loan'].agg([np.mean,'count'])
Education_Loan_Percentage.rename(columns={'mean': 'Personal_Loan_Mean','count':'Number of People'})
#5 Correlation between Family and Personal Loan
Family_Loan_Percentage = Bank_Personal_Loan.groupby('Family')['Personal Loan'].agg([np.mean,'count'])
Family_Loan_Percentage.rename(columns={'mean': 'Personal_Loan_Mean','count':'Number of People'})
#6 Correlation between Family and Personal Loan
Morrgage_Loan_Percentage = Bank_Personal_Loan.groupby('Personal Loan')['Mortgage'].agg([np.mean,'count'])
Morrgage_Loan_Percentage.rename(columns={'mean': 'Mortgage_Mean','count':'Number of People'})
# set the mortgage as the nominal feature; set mortgage > 0 as mortgage = 1
Bank_Personal_Loan.loc[Bank_Personal_Loan['Mortgage'] > 0, 'Mortgage'] = 1 
Bank_Personal_Loan.groupby(['Mortgage']).size()
Mortgage_Loan_Percentage = Bank_Personal_Loan.groupby('Mortgage')['Personal Loan'].agg([np.mean,'count'])
Mortgage_Loan_Percentage.rename(columns={'mean': 'Personal_Loan_Mean','count':'Number of People'})