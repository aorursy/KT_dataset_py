# coding:utf-8



# The computer should first preinstall the xlrd module

# Use anaconda to integrade and install pd np module



import pandas as pd

import numpy as np



# The proceeded data table and this script should be placed in the same directory

# Returns a DataFrame

df = pd.read_excel('/kaggle/input/20200325/SimFin-data(Microsoft).xlsx', sheet_name='MICROSOFT CO') 

# df # View the introduced table
# Get data from 3 tables

sheet1Temp = df[3:14]

Profit_and_LossStatement = sheet1Temp.iloc[0:, 2:]

Profit_and_LossStatement.index =  ['Revenue', 'Cost of revenue', 'Gross Profit', 'Operating Expenses', 'Operating Income (Loss)', 

                                   'Non-Operating Income (Loss)', 'Pretax Income (Loss), Adjusted','Abnormal Gains (Losses)','Pretax Income (Loss)','Income Tax (Expense) Benefit, net',

                                   'Net Income Available to Common Shareholders'] # Rename index



sheet2Temp = df[17:43]

BalanceSheet = sheet2Temp.iloc[0:, 2:]

BalanceSheet.index = [ "Assets", "Cash, Cash Equivalents & Short Term Investments", "Accounts & Notes Receivable", "Inventories", 

                      "Other Short Term Assets", "Total Current Assets", "Property, Plant & Equipment, Net", "Long Term Investments & Receivables",

                      "Other Long Term Assets", "Total Noncurrent Assets", "Total Assets", "Liabilities", "Payables & Accruals", "Short Term Debt",

                      "Other Short Term Liabilities", "Total Current Liabilities", "Long Term Debt", "Other Long Term Liabilities", 

                      "Total Noncurrent Liabilities", "Total Liabilities", "Share Capital & Additional Paid-In Capital", "Retained Earnings", 

                      "Other Equity", "Equity Before Minority Interest", "Total Equity", "Total Liabilities & Equity"]



sheet3Temp = df[46:64]

CashFlowStatement = sheet3Temp.iloc[0:, 2:]

CashFlowStatement.index = [ "Net Income/Starting Line", "Depreciation & Amortization", "Non-Cash Items", "Change in Working Capital", 

                           "Cash from Operating Activities", "Change in Fixed Assets & Intangibles", "Net Change in Long Term Investment", 

                           "Net Cash From Acquisitions & Divestitures", "Other Investing Activities", "Cash from Investing Activities", 

                           "Dividends Paid", "Cash From (Repayment of) Debt", "Cash From (Repurchase of) Equity", "Other Financing Activities", 

                           "Cash from Financing Activities","Net Cash Before FX","Effect of Foreign Exchange Rates", "Net Changes in Cash"]



Profit_and_LossStatement.columns = BalanceSheet.columns = CashFlowStatement.columns =  [

    "FY '10", "FY '11", "FY '12", "FY '13", "FY '14", "FY '15", "FY '16", "FY '17", "FY '18", "FY '19", ] # Rename column

# Print confirmation data

# Profit_and_LossStatement

# BalanceSheet

# CashFlowStatement
# Calculate the EPS of the stock in the past 10 years

# Calculation of the first index of the value investment principle



NetIncomeAvailableToCommonShareholders = Profit_and_LossStatement.loc['Net Income Available to Common Shareholders']

ShareCapital_AdditionalPaidInCapital = BalanceSheet.loc['Share Capital & Additional Paid-In Capital']

yearsList = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

L68 = NetIncomeAvailableToCommonShareholders[9] / ShareCapital_AdditionalPaidInCapital[9]



EPSList = []          

for index in range(len(yearsList)):

    EPSList.append(NetIncomeAvailableToCommonShareholders[index] / ShareCapital_AdditionalPaidInCapital[index])

EPSList
# Calculate the EPS Compound Annual Growth Rate in the past 10 years



CAGRList = []

for index in range(len(yearsList)-1):

    CAGRList.append((L68/(NetIncomeAvailableToCommonShareholders[index] / ShareCapital_AdditionalPaidInCapital[index]))**(1/(2019-yearsList[index]))-1)

CAGRList.append("NaN") # The denominator of the exponent in the calculation formula for 2019 is 0, empty array in 2019

# CAGRList
# Calculate the Debt to Equity Ratio in the past 10 years

# Calculation of the second index of the value investment principle



TotalLiabilities = BalanceSheet.loc['Total Liabilities']

TotalEquity = BalanceSheet.loc['Total Equity']



DebtToEquityRatioList = []

for index in range(len(yearsList)):

    temp = ('%.1f' %(TotalLiabilities[index] / TotalEquity[index] * 100)) + "%" # Percentage after one decimal place

    DebtToEquityRatioList.append(temp)

# DebtToEquityRatioList
# Calculate the Interest Coverage Ratio in the past 10 years

# Calculation of the third index of the value investment principle



OperatingIncome_Loss = Profit_and_LossStatement.loc['Operating Income (Loss)']

Non_OperatingIncome_Loss = Profit_and_LossStatement.loc['Non-Operating Income (Loss)']



InterestCoverageRatioList = []

for index in range(len(yearsList)):

    InterestCoverageRatioList.append((OperatingIncome_Loss[index] - Non_OperatingIncome_Loss[index])/ Non_OperatingIncome_Loss[index])

# InterestCoverageRatioList
# Combining arrays

outList = pd.DataFrame({'EPS': EPSList,

                    'CAGR': CAGRList,

                    'Debt to Equity Ratio': DebtToEquityRatioList,

                    'Interest Coverage Ratio': InterestCoverageRatioList})

outList.index =  yearsList

outList = outList.T # Transpose matrix
# Write excel

with pd.ExcelWriter('answer.xlsx') as writer:

    outList.to_excel(writer)
# Complete

complete = "Finish"

complete
# Estimate the EPS of the stock in 2030



import statsmodels.api as sm

import matplotlib.pyplot as plt



y = pd.Series(EPSList)

y



x = pd.Series(yearsList)

x



# Linear regression fit

x_n = sm.add_constant(x) # Add a comstant term to the model, that is, the intercept of the regression line on the y axis

model = sm.OLS(y, x_n) 

results = model.fit() # Use fit() method of OLS object for model fitting



results.summary() # View the result of model fitting

results.params # View the parameters of the final simulation



# Select 100 data points that are equally spaced from the minimum to the maximum

X_prime=np.linspace(x.min(), x.max(),100)[:,np.newaxis]

X_prime=sm.add_constant(X_prime)

# Calculating predictions

y_hat=results.predict(X_prime)

 

# The followings are used for plotting

plt.figure()

plt.title(u"Estimate the EPS of the stock in 2030")

plt.xlabel(u"Year")

plt.ylabel(u"EPS") # Name the x axis and y axis seperately

plt.axis([2005, 2035, 0, 0.6])

plt.scatter(x, y, marker="o",color="b", s=5)

plt.plot(X_prime, y_hat, linewidth=1, color="r") # Add regression line, in red

plt.show()



print("if errors, please run this cells again")



# View the parameters of the final model

results.params

print("The regression line is: Y = " + str(results.params[0]) + " * X + " + str(results.params.const) )

print("When x=2030, y = " + str(results.params[0] * 2030 + results.params.const))
outList