import pandas as pd

import numpy as np

def excel_to_df(excel_sheet):

 df = pd.read_excel(excel_sheet)

 df.dropna(how='all', inplace=True)



 index_PL = int(df.loc[df['Data provided by SimFin']=='Profit & Loss statement'].index[0])

 index_CF = int(df.loc[df['Data provided by SimFin']=='Cash Flow statement'].index[0])

 index_BS = int(df.loc[df['Data provided by SimFin']=='Balance Sheet'].index[0])



 df_PL = df.iloc[index_PL:index_BS-1, 1:]

 df_PL.dropna(how='all', inplace=True)

 df_PL.columns = df_PL.iloc[0]

 df_PL = df_PL[1:]

 df_PL.set_index("in million USD", inplace=True)

 (df_PL.fillna(0, inplace=True))

 



 df_BS = df.iloc[index_BS-1:index_CF-2, 1:]

 df_BS.dropna(how='all', inplace=True)

 df_BS.columns = df_BS.iloc[0]

 df_BS = df_BS[1:]

 df_BS.set_index("in million USD", inplace=True)

 df_BS.fillna(0, inplace=True)

 



 df_CF = df.iloc[index_CF-2:, 1:]

 df_CF.dropna(how='all', inplace=True)

 df_CF.columns = df_CF.iloc[0]

 df_CF = df_CF[1:]

 df_CF.set_index("in million USD", inplace=True)

 df_CF.fillna(0, inplace=True)

 

 df_CF = df_CF.T

 df_BS = df_BS.T

 df_PL = df_PL.T

    

 return df, df_PL, df_BS, df_CF



def combine_regexes(regexes):

 return "(" + ")|(".join(regexes) + ")"
# Do the below in kaggle

#!pip install plotly==4.4.1

#!pip install chart_studio

#!pip install xlrd # for reading_excel files with extensions .xlsx into a pandas dataframe
# Add 'tk_library_py.py' file given by your tutor, as a utility script under 'File'

# Look for it under usr/bin on the right drawer



# import excel_to_df function

import os

#import tk_library_py

#from tk_library_py import excel_to_df

import matplotlib.pyplot as plt
# Show the files and their pathnames

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Add your simfin-data.xlsx using the '+ Add Data' top right button

_, NVIDIA_PL, NVIDIA_BS, NVIDIA_CF = excel_to_df("/kaggle/input/nvidiaamd2/NVIDIA .xlsx")

_, AMD_PL, AMD_BS, AMD_CF = excel_to_df("/kaggle/input/nvidiaamd2/AMD.xlsx")
NVIDIA_BS
AMD_BS
del(NVIDIA_BS["Assets"])

NVIDIA_BS
del(AMD_BS["Assets"])
AMD_BS
NVIDIA_BS["_Total Current Assets"] = NVIDIA_BS["Cash, Cash Equivalents & Short Term Investments"] + NVIDIA_BS["Accounts & Notes Receivable"] + NVIDIA_BS["Inventories"] + NVIDIA_BS["Other Short Term Assets"]
AMD_BS["_Total Current Assets"] = AMD_BS["Cash, Cash Equivalents & Short Term Investments"] + AMD_BS["Accounts & Notes Receivable"] + AMD_BS["Inventories"] + AMD_BS["Other Short Term Assets"]
NVIDIA_BS[["_Total Current Assets", "Total Current Assets"]]
AMD_BS[["_Total Current Assets", "Total Current Assets"]]
NVIDIA_BS["_NonCurrent Assets"] = NVIDIA_BS["Property, Plant & Equipment, Net"] + NVIDIA_BS["Other Long Term Assets"]
AMD_BS["_NonCurrent Assets"] = AMD_BS["Property, Plant & Equipment, Net"] + AMD_BS["Other Long Term Assets"]
NVIDIA_BS["_Total Assets"] = NVIDIA_BS["_NonCurrent Assets"] + NVIDIA_BS["_Total Current Assets"] 
AMD_BS["_Total Assets"] = AMD_BS["_NonCurrent Assets"] + AMD_BS["_Total Current Assets"] 
NVIDIA_BS["_Total Liabilities"] = NVIDIA_BS["Total Current Liabilities"] + NVIDIA_BS["Total Noncurrent Liabilities"]
AMD_BS["_Total Liabilities"] = AMD_BS["Total Current Liabilities"] + AMD_BS["Total Noncurrent Liabilities"]
NVIDIA_BS[["_Total Liabilities", "Total Liabilities"]]
AMD_BS[["_Total Liabilities", "Total Liabilities"]]
%matplotlib inline

NVIDIA_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
%matplotlib inline

AMD_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

'''



asset_columns = [ x for x in good_stuff.strip().split("\n") ]

asset_columns
NVIDIA_BS[ asset_columns ].plot()
AMD_BS[ asset_columns ].plot()
import chart_studio

# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 

# https://chart-studio.plot.ly/feed/#/



# Un-remark the code below and add your own your_username and own your_apikey

chart_studio.tools.set_credentials_file(username='WUZHIHANG', api_key='Hv6wWJYO9yzQrpR1d2TL')
import chart_studio.plotly as py

import plotly.graph_objs as go

#from tk_library_py import combine_regexes

NVIDIA_BS["working capital"] = NVIDIA_BS["Total Current Assets"] - NVIDIA_BS["Total Current Liabilities"]
NVIDIA_BS[["working capital"]].plot()
AMD_BS["working capital"] = AMD_BS["Total Current Assets"] - AMD_BS["Total Current Liabilities"]
AMD_BS[["working capital"]].plot()
plt.plot(NVIDIA_PL.index,NVIDIA_BS["working capital"],label='NVIDIA working capital')

plt.plot(AMD_BS.index,AMD_BS["working capital"],label='AMD working capital')

plt.legend(loc="best")

plt.show()
NVIDIA_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
AMD_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
NVIDIA_BS["Inventories"].plot()
NVIDIA_BS[ ["Property, Plant & Equipment, Net", "Other Long Term Assets"] ].plot()
# NVIDIA mobil has no preferred stock, no intengible assets, and no goodwill



NVIDIA_BS["book value"] = NVIDIA_BS["Total Assets"] - NVIDIA_BS["Total Liabilities"]

NVIDIA_BS["book value"].plot()
NVIDIA_BS["current ratio"] = NVIDIA_BS["Total Current Assets"] / NVIDIA_BS["Total Current Liabilities"]
NVIDIA_BS["current ratio"].plot()
AMD_BS["current ratio"] = AMD_BS["Total Current Assets"] / AMD_BS["Total Current Liabilities"]
AMD_BS["current ratio"].plot()
#Calculate 1.	Price-to-Earnings Growth Ratio (PEG forward)  

#using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate

#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp



NVIDIA_PE_RATIO = 38.23 # FROM Ycharts WEBSITE: https://ycharts.com/companies/NVDA/pe_ratio



# FROM Forbes WEBSITE: https://www.forbes.com/sites/kramermichael/2020/02/16/nvidias-stock-faces-a-rough-road-ahead/

GROWTH_RATE = 0.227 # Current 3-year compounded annual earnings growth rate,which includes the significant rebound in 2021



NVIDIA_PEG_ratio = NVIDIA_PE_RATIO / (GROWTH_RATE*100)



print("NVIDIA's PEG Ratio is", NVIDIA_PEG_ratio)
AMD_PE_RATIO = 136.55 # FROM Ycharts WEBSITE: https://ycharts.com/companies/AMD/pe_ratio



# FROM Yahoo WEBSITE: https://finance.yahoo.com/quote/amd/analysis/

GROWTH_RATE =1.33 # Average forecast next year 2021



AMD_PEG_ratio = AMD_PE_RATIO / (GROWTH_RATE*100)



print("AMD's PEG Ratio is", AMD_PEG_ratio)
#End of Value Investing Stock Analysis Template
#ROE=Net Income/Share Holder's Equity

NVIDIA_PL['NVIDIA ROE']=NVIDIA_PL["Net Income Available to Common Shareholders"]/NVIDIA_BS["Total Equity"]
NVIDIA_PL['NVIDIA ROE']
NVIDIA_PL[['NVIDIA ROE']]
plt.plot(NVIDIA_PL.index,NVIDIA_PL['NVIDIA ROE'])
AMD_PL['AMD ROE']=AMD_PL["Net Income Available to Common Shareholders"]/AMD_BS["Total Equity"]
AMD_PL['AMD ROE']
AMD_PL.index
AMD_PL['AMD ROE']
plt.plot(AMD_PL.index,AMD_PL['AMD ROE'])
plt.plot(NVIDIA_PL.index,NVIDIA_PL['NVIDIA ROE'],label='NVIDIA ROE')

plt.plot(AMD_PL.index,AMD_PL['AMD ROE'],label='AMD ROE')

plt.legend(loc="best")

plt.show()
#ROA=Net Income/Total Assets

NVIDIA_PL['NVIDIA ROA']=NVIDIA_PL["Net Income Available to Common Shareholders"]/NVIDIA_BS["Total Assets"]
NVIDIA_PL['NVIDIA ROA']
AMD_PL['AMD ROA']=AMD_PL["Net Income Available to Common Shareholders"]/AMD_BS["Total Assets"]
AMD_PL['AMD ROA']
plt.plot(NVIDIA_PL.index,NVIDIA_PL['NVIDIA ROA'],label='NVIDIA ROA')

plt.plot(AMD_PL.index,AMD_PL['AMD ROA'],label='AMD ROA')

plt.legend(loc="best")

plt.show()
#EPS=Net Income/shares

NVIDIA_PL['NVIDIA EPS']= NVIDIA_PL["Net Income Available to Common Shareholders"]/NVIDIA_BS["shares(Basic)"]
NVIDIA_PL['NVIDIA EPS']
AMD_PL['AMD EPS']= AMD_PL["Net Income Available to Common Shareholders"]/AMD_BS["shares(Basic)"]
AMD_PL['AMD EPS']
plt.plot(NVIDIA_PL.index,NVIDIA_PL['NVIDIA EPS'],label='NVIDIA EPS')

plt.plot(AMD_PL.index,AMD_PL['AMD EPS'],label='AMD EPS')

plt.legend(loc="best")

plt.show()
#Debt to Equity ratio=Total Liabilities/Total Equity

NVIDIA_BS['Debt to Equity ratio']=NVIDIA_BS["Total Liabilities"]/NVIDIA_BS["Total Equity"]
NVIDIA_BS['Debt to Equity ratio']
AMD_BS['Debt to Equity ratio']=AMD_BS["Total Liabilities"]/AMD_BS["Total Equity"]
AMD_BS['Debt to Equity ratio']
plt.plot(NVIDIA_BS.index,NVIDIA_BS['Debt to Equity ratio'],label='NVIDIA Debt to Equity ratio')

plt.plot(AMD_BS.index,AMD_BS['Debt to Equity ratio'],label='AMD Debt to Equity ratio')

plt.legend(loc="best")

plt.show()
#Margin of Safety=(Sales-Break Even Point)/Sales

#Break Even Point=Fixed Expenses/Contribution Margin

#Contribution Margin=(Sales-Variable Expense)/Sales

NVIDIA_Contribution_Margin=(NVIDIA_PL["Revenue"]+NVIDIA_PL["Cost of revenue"])/NVIDIA_PL["Revenue"]

NVIDIA_Break_Even_Point=(-NVIDIA_PL["Operating Expenses"])/NVIDIA_Contribution_Margin

NVIDIA_Margin_of_Safety=(NVIDIA_PL["Revenue"]-NVIDIA_Break_Even_Point)/NVIDIA_PL["Revenue"]

NVIDIA_Margin_of_Safety.mean()
AMD_Contribution_Margin=(AMD_PL["Revenue"]+AMD_PL["Cost of revenue"])/AMD_PL["Revenue"]

AMD_Break_Even_Point=(-AMD_PL["Operating Expenses"])/AMD_Contribution_Margin

AMD_Margin_of_Safety=(AMD_PL["Revenue"]-AMD_Break_Even_Point)/AMD_PL["Revenue"]

AMD_Margin_of_Safety
#Step1 EPS Annual Compounded Growth Rate from 2016 to 2019

EPS_Year16=3.079482

EPS_Year19=4.591133

NVIDIA_EPS_Annual_Compounded_Growth_Rate_new=pow((EPS_Year19/EPS_Year16),1/3)-1
print("{:.2%}".format(NVIDIA_EPS_Annual_Compounded_Growth_Rate_new.real))
#Step1 EPS Annual Compounded Growth Rate from 2017 to 2019

AMD_Year17=0.040000

AMD_Year19=0.310000

AMD_EPS_Annual_Compounded_Growth_Rate_new=pow((AMD_Year19/AMD_Year17),1/2)-1
print("{:.2%}".format(AMD_EPS_Annual_Compounded_Growth_Rate_new.real))
#Step2 Estimate EPS 10 years from now

#choose the EPS Annual Compounded Growth Rate from 2015 to 2019

NVIDIA_Estimate_EPS=EPS_Year19*((1+0.1424)**9)
print(NVIDIA_Estimate_EPS)
#Step2 Estimate EPS 10 years from now use regression

AMD_PL.reset_index(inplace=True)
import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score



data_x_train=np.array([AMD_PL.index]).T

data_y_train=np.array([AMD_PL['AMD EPS']]).T

data_x_test=np.array([[AMD_PL.index[-1]+x for x in range(1,11)]]).T

# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(data_x_train, data_y_train)



# Make predictions using the testing set

data_y_pred = regr.predict(data_x_test)



plt.scatter(data_x_train,data_y_train)

plt.scatter(data_x_test,data_y_pred)
data_y_pred
#Step3 Estimate Stock Price 10 Years from now

#NVIDIA_Future_Stock_Price=NVIDIA_Estimate_EPS*Average PE Ratio

NVIDIA_Average_PE_Ratio=38.23#from https://ycharts.com/companies/NVDA/pe_ratio

NVIDIA_Future_Stock_Price=NVIDIA_Estimate_EPS*NVIDIA_Average_PE_Ratio
print(NVIDIA_Future_Stock_Price)
#Step3 Estimate Stock Price 10 Years from now

#AMD_Future_Stock_Price=AMD_Estimate_EPS*Average PE Ratio

AMD_Average_PE_Ratio=136.55#from https://ycharts.com/companies/AMD/pe_ratio

AMD_Future_Stock_Price=-0.08*AMD_Average_PE_Ratio
print(AMD_Future_Stock_Price)
#Step4 Determine Target Buy Price Today Based on Desired Returns

NVIDIA_Discount_Rate=0.10 #from https://finbox.com/NASDAQGS:NVDA/models/wacc

NVIDIA_Target_Buy_Price=NVIDIA_Future_Stock_Price/((1+NVIDIA_Discount_Rate)**9)
print(NVIDIA_Target_Buy_Price)
#Step4 Determine Target Buy Price Today Based on Desired Returns

AMD_Discount_Rate=0.115 #from https://finbox.com/NASDAQGS:NVDA/models/wacc

AMD_Target_Buy_Price=AMD_Future_Stock_Price/((1+AMD_Discount_Rate)**9)
print(AMD_Target_Buy_Price)
#Step5 Add Margin of Safety

#35.49% off the target buy price

NVIDIA_Current_Target_Buy_Price=NVIDIA_Target_Buy_Price*(1-0.3549)
print(NVIDIA_Current_Target_Buy_Price)
#Step5 Add Margin of Safety

#19.9093% off the target buy price

AMD_Current_Target_Buy_Price=AMD_Target_Buy_Price*(1-0.199093)
print(AMD_Current_Target_Buy_Price)
NVIDIA_PL.columns
#Interest Coverage Ratio=EBIT/Interest Expenses

NVIDIA_Interest_Coverage_Ratio=NVIDIA_PL["EBIT"]/NVIDIA_PL["Interest Expense,net"]
NVIDIA_Interest_Coverage_Ratio
AMD_Interest_Coverage_Ratio=AMD_PL["EBIT"]/AMD_PL["Interest Expense,net"]
AMD_Interest_Coverage_Ratio
#Inventory turnover=Cost of revenue/Inventories

NVIDIA_PL['Inventory turnover']=(-NVIDIA_PL["Cost of revenue"])/NVIDIA_BS["Inventories"]
print(NVIDIA_PL['Inventory turnover'])
df=AMD_PL

df.index=AMD_BS.index
AMD_BS['Inventory turnover']=(-df["Cost of revenue"])/AMD_BS["Inventories"]
AMD_BS['Inventory turnover']
plt.plot(NVIDIA_PL.index,NVIDIA_PL['Inventory turnover'],label='NVIDIA Inventory turnover')

plt.plot(AMD_BS.index,AMD_BS['Inventory turnover'],label='AMD Inventory turnover')

plt.legend(loc="best")

plt.show()
#Total Asset Turnover=Revenue/Total Assets

NVIDIA_PL['Total Asset Turnover']=NVIDIA_PL["Revenue"]/NVIDIA_BS["Total Assets"]
print(NVIDIA_PL['Total Asset Turnover'])
NVIDIA_PL["Revenue"].plot()
df["Revenue"].plot()
plt.plot(NVIDIA_PL.index,NVIDIA_PL["Revenue"],label='NVIDIA Revenue')

plt.plot(AMD_BS.index,df["Revenue"],label='AMD Revenue')

plt.legend(loc="best")

plt.show()
AMD_BS['Total Asset Turnover']=df["Revenue"]/AMD_BS["Total Assets"]
print(AMD_BS['Total Asset Turnover'])
plt.plot(NVIDIA_PL.index,NVIDIA_PL['Total Asset Turnover'],label='NVIDIA Total Asset Turnover')

plt.plot(AMD_BS.index,AMD_BS['Total Asset Turnover'],label='AMD Total Asset Turnover')

plt.legend(loc="best")

plt.show()
#Debt Asset ratio=Total Liabilities/Total Assets

NVIDIA_BS['Debt Asset ratio']=NVIDIA_BS["Total Liabilities"]/NVIDIA_BS["Total Assets"]
print(NVIDIA_BS['Debt Asset ratio'])
AMD_BS['Debt Asset ratio']=AMD_BS["Total Liabilities"]/AMD_BS["Total Assets"]
print(AMD_BS['Debt Asset ratio'])
plt.plot(NVIDIA_BS.index,NVIDIA_BS['Debt Asset ratio'],label='NVIDIA Debt Asset ratio')

plt.plot(AMD_BS.index,AMD_BS['Debt Asset ratio'],label='AMD Debt Asset ratio')

plt.legend(loc="best")

plt.show()
AMD=pd.read_csv("/kaggle/input/amdstock/AMD (2).csv",index_col="Date",parse_dates=["Date"])

AMD=AMD.dropna()
sub =AMD[ '2010-01': '2020-03'][ 'Close']



train = sub.ix[ '2010-01': '2020-03']



test = sub.ix[ '2015-02': '2020-03']



plt.figure(figsize=( 10, 10))



print(train)



plt.plot(train)



plt.show()
import statsmodels.api as sm

model = sm.tsa.ARIMA(sub, order=( 1, 0, 0))



results = model.fit()



predict_sunspots = results.predict(start=str('2016-01'),end=str('2020-03'),dynamic=False)



print(predict_sunspots)



fig, ax = plt.subplots(figsize=(12, 8))



ax = sub.plot(ax=ax)



predict_sunspots.plot(ax=ax)



plt.show()



results.forecast()[0]

print(results.forecast()[0])