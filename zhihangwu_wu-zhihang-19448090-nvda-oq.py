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
# Show the files and their pathnames
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Add your simfin-data.xlsx using the '+ Add Data' top right button
_, NVIDIA_PL, NVIDIA_BS, NVIDIA_CF = excel_to_df("/kaggle/input/nvidia3/NVDA.xlsx")
NVIDIA_BS
del(NVIDIA_BS["Assets"])

NVIDIA_BS
NVIDIA_BS["_Total Current Assets"] = NVIDIA_BS["Cash, Cash Equivalents & Short Term Investments"] + NVIDIA_BS["Accounts & Notes Receivable"] + NVIDIA_BS["Inventories"] + NVIDIA_BS["Other Short Term Assets"]
NVIDIA_BS[["_Total Current Assets", "Total Current Assets"]]
NVIDIA_BS["_NonCurrent Assets"] = NVIDIA_BS["Property, Plant & Equipment, Net"] + NVIDIA_BS["Other Long Term Assets"]
NVIDIA_BS["_Total Assets"] = NVIDIA_BS["_NonCurrent Assets"] + NVIDIA_BS["_Total Current Assets"] 
NVIDIA_BS["_Total Liabilities"] = NVIDIA_BS["Total Current Liabilities"] + NVIDIA_BS["Total Noncurrent Liabilities"]
NVIDIA_BS[["_Total Liabilities", "Total Liabilities"]]
%matplotlib inline
NVIDIA_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''
Cash, Cash Equivalents & Short Term Investments
Accounts & Notes Receivable
Inventories
Other Short Term Assets
'''

asset_columns = [ x for x in good_stuff.strip().split("\n") ]

asset_columns
NVIDIA_BS[ asset_columns ].plot()
import chart_studio
# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 
# https://chart-studio.plot.ly/feed/#/

# Un-remark the code below and add your own your_username and own your_apikey
chart_studio.tools.set_credentials_file(username='WUZHIHANG', api_key='Hv6wWJYO9yzQrpR1d2TL')
import chart_studio.plotly as py
import plotly.graph_objs as go
#from tk_library_py import combine_regexes

assets = go.Bar(
    x=NVIDIA_BS.index,
    y=NVIDIA_BS["Total Assets"],
    name='Assets'
)
liabilities = go.Bar(
    x=NVIDIA_BS.index,
    y=NVIDIA_BS["Total Liabilities"],
    name='Liabilities'
)

shareholder_equity = go.Scatter(
    x=NVIDIA_BS.index,
    y=NVIDIA_BS["Total Equity"],
    name='Equity'
)

data = [assets, liabilities, shareholder_equity]
layout = go.Layout(
    barmode='stack'
)

fig_bs = go.Figure(data=data, layout=layout)
#py.iplot(fig_bs, filename='Total Assets and Liabilities')
fig_bs.show()
asset_data = []
columns = '''
Cash, Cash Equivalents & Short Term Investments
Accounts & Notes Receivable
Inventories
Other Short Term Assets
Property, Plant & Equipment, Net
Other Long Term Assets
'''


for col in columns.strip().split("\n"):
    asset_bar = go.Bar(
        x=NVIDIA_BS.index,
        y=NVIDIA_BS[ col ],
        name=col
    )    
    asset_data.append(asset_bar)
    
layout_assets = go.Layout(
    barmode='stack'
)

fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)
#py.iplot(fig_bs_assets, filename='Total Assets Breakdown')
fig_bs_assets.show()
liability_data = []
columns = '''
Payables & Accruals
Short Term Debt
Other Short Term Liabilities
Long Term Debt
Other Long Term Liabilities
'''


for col in columns.strip().split("\n"):
    liability_bar = go.Bar(
        x=NVIDIA_BS.index,
        y=NVIDIA_BS[ col ],
        name=col
    )    
    liability_data.append(liability_bar)
    
layout_liabilitys = go.Layout(
    barmode='stack'
)

fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)
#py.iplot(fig_bs_liabilitys, filename='Total liabilities Breakdown')
fig_bs_liabilitys.show()
NVIDIA_BS["working capital"] = NVIDIA_BS["Total Current Assets"] - NVIDIA_BS["Total Current Liabilities"]
NVIDIA_BS[["working capital"]].plot()
NVIDIA_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
# Using Chart Studio in Plotly 

PR_data = []
columns = '''
Accounts & Notes Receivable
Payables & Accruals
'''

for col in columns.strip().split("\n"):
    PR_Scatter = go.Scatter(
        x=NVIDIA_BS.index,
        y=NVIDIA_BS[ col ],
        name=col
    )    
    PR_data.append(PR_Scatter)
    
layout_PR = go.Layout(
    barmode='stack'
)

fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)
#py.iplot(fig_bs_PR, filename='Accounts & Notes Receivable vs Payables & Accruals')
fig_bs_PR.show()
NVIDIA_BS["Inventories"].plot()
NVIDIA_BS[ ["Property, Plant & Equipment, Net", "Other Long Term Assets"] ].plot()
# Using Plotly

AAA_data = []
columns = '''
Property, Plant & Equipment, Net
Other Long Term Assets
'''


for col in columns.strip().split("\n"):
    AAA_bar = go.Bar(
        x=NVIDIA_BS.index,
        y=NVIDIA_BS[ col ],
        name=col
    )    
    AAA_data.append(AAA_bar)
    
layout_AAA = go.Layout(
    barmode='stack'
)

fig_bs_AAA = go.Figure(data=AAA_data, layout=layout_AAA)
#py.iplot(fig_bs_AAA, filename='Total Long Term Assets')
fig_bs_AAA.show()
equity_columns = '''
Share Capital & Additional Paid-In Capital
Treasury Stock
Retained Earnings
Other Equity
Equity Before Minority Interest
'''

equity_columns = [ x for x in equity_columns.strip().split("\n")]

equity_columns
NVIDIA_BS[ equity_columns ].plot()
# Using Plotly

equity_data = []
columns = '''
Share Capital & Additional Paid-In Capital
Treasury Stock
Retained Earnings
Other Equity
Equity Before Minority Interest
'''


for col in columns.strip().split("\n"):
    equity_Scatter = go.Scatter(
        x=NVIDIA_BS.index,
        y=NVIDIA_BS[ col ],
        name=col
    )    
    equity_data.append(equity_Scatter)
    
layout_equity = go.Layout(
    barmode='stack'
)

fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)
#py.iplot(fig_bs_equity, filename='Total Equity')
fig_bs_equity.show()
# NVIDIA mobil has no preferred stock, no intengible assets, and no goodwill

NVIDIA_BS["book value"] = NVIDIA_BS["Total Assets"] - NVIDIA_BS["Total Liabilities"]

NVIDIA_BS["book value"].plot()
NVIDIA_BS["current ratio"] = NVIDIA_BS["Total Current Assets"] / NVIDIA_BS["Total Current Liabilities"]
NVIDIA_BS["current ratio"].plot()
#Calculate 1.	Price-to-Earnings Growth Ratio (PEG forward)  
#using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate
#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp

PE_RATIO = 16.38 # FROM SIMFIN WEBSITE: https://simfin.com/data/companies/121214

# FROM NASDAQ WEBSITE: https://www.nasdaq.com/symbol/xom/earnings-growth
GROWTH_RATE = 0.1394 # Forcast over the five next years

PEG_ratio = PE_RATIO / (GROWTH_RATE*100)

print("NVIDIA Mobil Corp's PEG Ratio is", PEG_ratio)
#End of Value Investing Stock Analysis Template
#ROE=Net Income/Share Holder's Equity
NVIDIA_ROE=NVIDIA_PL["Net Income Available to Common Shareholders"]/NVIDIA_BS["Total Equity"]
NVIDIA_ROE
#ROA=Net Income/Total Assets
NVIDIA_ROA=NVIDIA_PL["Net Income Available to Common Shareholders"]/NVIDIA_BS["Total Assets"]
NVIDIA_ROA
#EPS=Net Income/shares
NVIDIA_EPS= NVIDIA_PL["Net Income Available to Common Shareholders"]/NVIDIA_BS["Shares (Basic)"]
NVIDIA_EPS
#Debt to Equity ratio=Total Liabilities/Total Equity
NVIDIA_Debt_to_Equity_ratio=NVIDIA_BS["Total Liabilities"]/NVIDIA_BS["Total Equity"]
NVIDIA_Debt_to_Equity_ratio
#Margin of Safety=(Sales-Break Even Point)/Sales
#Break Even Point=Fixed Expenses/Contribution Margin
#Contribution Margin=(Sales-Variable Expense)/Sales
NVIDIA_Contribution_Margin=(NVIDIA_PL["Revenue"]+NVIDIA_PL["Cost of revenue"])/NVIDIA_PL["Revenue"]
NVIDIA_Break_Even_Point=(-NVIDIA_PL["Operating Expenses"])/NVIDIA_Contribution_Margin
NVIDIA_Margin_of_Safety=(NVIDIA_PL["Revenue"]-NVIDIA_Break_Even_Point)/NVIDIA_PL["Revenue"]
NVIDIA_Margin_of_Safety.mean()
#Step1 EPS Annual Compounded Growth Rate from 2010 to 2018
EPS_Year10=0.440118
EPS_Year18=6.810855
NVIDIA_EPS_Annual_Compounded_Growth_Rate=pow((EPS_Year18/EPS_Year10),1/8)-1
print("{:.2%}".format(NVIDIA_EPS_Annual_Compounded_Growth_Rate.real))
#Step1 EPS Annual Compounded Growth Rate from 2017 to 2018
EPS_Year17=5.086811
EPS_Year18=6.810855
NVIDIA_EPS_Annual_Compounded_Growth_Rate_new=pow((EPS_Year18/EPS_Year17),1)-1
print("{:.2%}".format(NVIDIA_EPS_Annual_Compounded_Growth_Rate_new.real))
#Step2 Estimate EPS 10 years from now
#choose the EPS Annual Compounded Growth Rate from 2017 to 2018
NVIDIA_Estimate_EPS=EPS_Year18*((1+0.3389)**9)
print(NVIDIA_Estimate_EPS)
#Step3 Estimate Stock Price 10 Years from now
#NVIDIA_Future_Stock_Price=NVIDIA_Estimate_EPS*Average PE Ratio
Average_PE_Ratio=37.63#from https://ycharts.com/companies/NVDA/pe_ratio
NVIDIA_Future_Stock_Price=NVIDIA_Estimate_EPS*Average_PE_Ratio
print(NVIDIA_Future_Stock_Price)
#Step4 Determine Target Buy Price Today Based on Desired Returns
Discount_Rate=0.105 #from https://finbox.com/NASDAQGS:NVDA/models/wacc
Target_Buy_Price=NVIDIA_Future_Stock_Price/((1+Discount_Rate)**9)
print(Target_Buy_Price)
#Step5 Add Margin of Safety
#25% off the target buy price
Current_Target_Buy_Price=Target_Buy_Price*(1-0.3045)
print(Current_Target_Buy_Price)
NVIDIA_PL.columns
#Interest Coverage Ratio=EBIT/Interest Expenses
NVIDIA_Interest_Coverage_Ratio=NVIDIA_PL["EBIT"]/NVIDIA_PL["Interest Expense,net"]
NVIDIA_Interest_Coverage_Ratio
#Inventory turnover=Cost of revenue/Inventories
NVIDIA_Inventory_turnover=(-NVIDIA_PL["Cost of revenue"])/NVIDIA_BS["Inventories"]
print(NVIDIA_Inventory_turnover)
#Total Asset Turnover=Revenue/Total Assets
NVIDIA_Total_Asset_Turnover=NVIDIA_PL["Revenue"]/NVIDIA_BS["Total Assets"]
print(NVIDIA_Total_Asset_Turnover)
NVIDIA_PL["Revenue"].plot()
#Debt Asset ratio=Total Liabilities/Total Assets
NVIDIA_Debt_Asset_ratio=NVIDIA_BS["Total Liabilities"]/NVIDIA_BS["Total Assets"]
print(NVIDIA_Debt_Asset_ratio)