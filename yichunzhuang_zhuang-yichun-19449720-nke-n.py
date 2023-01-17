#!pip install plotly==4.4.1

#!pip install chart_studio

#!pip install xlrd 
import os

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
for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
_, nike_PL, nike_BS, nike_CF = excel_to_df("/kaggle/input/simfin-datanike/SimFin-data.xlsx")
nike_BS
del(nike_BS["Assets"])

nike_BS
nike_BS["_Total Current Assets"] = nike_BS["Cash, Cash Equivalents & Short Term Investments"] + nike_BS["Accounts & Notes Receivable"] + nike_BS["Inventories"] + nike_BS["Other Short Term Assets"]
nike_BS[["_Total Current Assets", "Total Current Assets"]]
nike_BS["_NonCurrent Assets"] = nike_BS["Property, Plant & Equipment, Net"] + nike_BS["Other Long Term Assets"]
nike_BS["_Total Assets"] = nike_BS["_NonCurrent Assets"] + nike_BS["_Total Current Assets"] 
nike_BS[["_Total Assets","Total Assets"]]
nike_BS["_Total Liabilities"] = nike_BS["Total Current Liabilities"] + nike_BS["Total Noncurrent Liabilities"]
nike_BS[["_Total Liabilities", "Total Liabilities"]]
nike_BS["_Total Equity"] = nike_BS["Total Assets"] - nike_BS["Total Liabilities"]
nike_BS[["_Total Equity", "Total Equity"]]
%matplotlib inline

nike_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

'''



asset_columns = [ x for x in good_stuff.strip().split("\n") ]

asset_columns
nike_BS[ asset_columns ]
nike_BS[ asset_columns ].plot()
nike_BS[ ["Property, Plant & Equipment, Net", "Other Long Term Assets"] ].plot()
good_stufftwo = '''

Payables & Accruals

Short Term Debt

Other Short Term Liabilities

'''



liability_columns = [ x for x in good_stufftwo.strip().split("\n") ]
liability_columns
nike_BS[ liability_columns ]
nike_BS[ liability_columns ].plot()
nike_BS[ ["Long Term Debt", "Other Long Term Liabilities"] ].plot()
equity_columns = '''

Preferred Equity

Share Capital & Additional Paid-In Capital

Retained Earnings

Other Equity

Equity Before Minority Interest

'''



equity_columns = [ x for x in equity_columns.strip().split("\n")]

equity_columns
nike_BS[ equity_columns ]
nike_BS[ equity_columns ].plot()
#Net Asset Value = Current Assets - Current Liabilities - Long Term Liabilities



nike_NAV = nike_BS["Total Current Assets"] - nike_BS["Total Current Liabilities"] - nike_BS["Long Term Debt"]
nike_NAV
nike_NAV.plot()
#Working Capital = Current Assets - Current Liabilities

nike_wc = nike_BS["Total Current Assets"] - nike_BS["Total Current Liabilities"]
nike_wc
nike_wc.plot()
import chart_studio

# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 

# https://chart-studio.plot.ly/feed/#/



chart_studio.tools.set_credentials_file(username='ZhuangYichun', api_key='wLrwgIkcbNrXm9DGNtpD')
import chart_studio.plotly as py

import plotly.graph_objs as go

from tk_library_py import combine_regexes

# Chart of Balance Sheet

assets = go.Bar(

    x=nike_BS.index,

    y=nike_BS["Total Assets"],

    name='Assets'

)

liabilities = go.Bar(

    x=nike_BS.index,

    y=nike_BS["Total Liabilities"],

    name='Liabilities'

)



shareholder_equity = go.Scatter(

    x=nike_BS.index,

    y=nike_BS["Total Equity"],

    name='Equity'

)



data = [assets, liabilities, shareholder_equity]

layout = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

fig_bs.show()
#Chart of Total Assets Breakdown

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

        x=nike_BS.index,

        y=nike_BS[ col ],

        name=col

    )    

    asset_data.append(asset_bar)

    

layout_assets = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

fig_bs.show()
#Chart of Total Liabilities Breakdown

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

        x=nike_BS.index,

        y=nike_BS[ col ],

        name=col

    )    

    liability_data.append(liability_bar)

    

layout_liabilitys = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

fig_bs.show()
# Chart of Total Equity Breakdown



equity_data = []

columns = '''

Preferred Equity

Share Capital & Additional Paid-In Capital

Retained Earnings

Other Equity

Equity Before Minority Interest

'''





for col in columns.strip().split("\n"):

    equity_Scatter = go.Scatter(

        x=nike_BS.index,

        y=nike_BS[ col ],

        name=col

    )    

    equity_data.append(equity_Scatter)

    

layout_equity = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

fig_bs.show()
nike_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
# Chart of Account Reveivables & Account Payables



PR_data = []

columns = '''

Accounts & Notes Receivable

Payables & Accruals

'''



for col in columns.strip().split("\n"):

    PR_Scatter = go.Scatter(

        x=nike_BS.index,

        y=nike_BS[ col ],

        name=col

    )    

    PR_data.append(PR_Scatter)

    

layout_PR = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

fig_bs.show()
# NIKE has no preferred stock, no intengible assets, and no goodwill



nike_bookvalue = nike_BS["Total Assets"] - nike_BS["Total Liabilities"]

nike_bookvalue
nike_bookvalue.plot()
nike_CF["Annual Shares Outstanding"] = [1976,1943,1879,1833,1812,1769,1743,1692,1659,1618]



nike_CF.columns

#Calculate Earning per Share

#Using this formula - EPS = (Net Income - Preferred Dividends)/ Common Shares Outstanding

#https://www.investopedia.com/terms/e/eps.asp



nike_EPS = nike_CF["Net Income/Starting Line"]/nike_CF["Annual Shares Outstanding"]
nike_EPS
nike_EPS.plot()
#2010-2019, 10 years of data, but only 9 years to grow, so the number of time periods is nine. 



present_value = float(nike_EPS[1])

future_value = float(nike_EPS[-1])

time_period= len(nike_EPS)
print(present_value)

print(future_value)

print(time_period)
def eps_growth(present_value, future_value, time_period):

    return (future_value / present_value) ** (1 / (time_period - 1)) - 1



nike_eps_growth = eps_growth(present_value, future_value, time_period)

print(nike_eps_growth)
print("{:.2%}".format(nike_eps_growth.real))
def tenyeareps_forecast (firstyear_value, eps_growth, periods):

    return firstyear_value * ((1 + eps_growth) ** periods)
firstyear_value = 2.49

eps_growth = 0.095

periods = 10

nike_tenyeareps_forecast = tenyeareps_forecast(firstyear_value, eps_growth, periods)



print(nike_tenyeareps_forecast)
#From https://ycharts.com/companies/NKE/pe_ratio, the average PE ratio of past five years of NIKE is 35.20.



def Estimate_SP (nike_tenyeareps_forecast, Avg_PE):

    return nike_tenyeareps_forecast * Avg_PE
Avg_PE=35.20

nike_tenyear_SP=Estimate_SP (nike_tenyeareps_forecast, Avg_PE)

print(nike_tenyear_SP)
#Future Value = 217.21

#Years = 10

#Discount Rate = 0.086, from Finbox.com



import numpy as np



discount_rate = 0.086

future_value = 217.21

PMT = 0

periods = 10



nike_buyprice = np.pv(discount_rate, periods, PMT, future_value, when='end' )

print(nike_buyprice)
# Margin of Safety=25%

margin = 0.25

nike_margin = -nike_buyprice * (1 - margin)



print(nike_margin)



#So the target buy price is 71.39.
#Calculation 1: Current Ratio

#Using this formula – Current Ratio = Current Assets / Current Liabilties 

#https://www.investopedia.com/terms/c/currentratio.asp



nike_BS["current ratio"] = nike_BS["Total Current Assets"] / nike_BS["Total Current Liabilities"]



nike_BS["current ratio"] 
nike_BS["current ratio"].plot()
#Calculation 2: Quick Ratio

#Using this formula – Quick Ratio = (Current Assets-Inventories) / Current Liabilties 

#https://www.investopedia.com/terms/q/quickratio.asp



nike_BS["quick ratio"] = (nike_BS["Total Current Assets"] - nike_BS["Inventories"]) / nike_BS["Total Current Liabilities"]



nike_BS["quick ratio"]
nike_BS["quick ratio"].plot()
#Calculation 3: Debt Ratio

#Using this formula – Debt Ratio = Total Liabilities / Total Assets

#https://www.investopedia.com/terms/d/debtratio.asp



nike_BS["debt ratio"] = (nike_BS["Total Liabilities"]) / nike_BS["Total Assets"]



nike_BS["debt ratio"]
nike_BS["debt ratio"].plot()
#Calculation 4: Debt to Equity ratio

#Using this formula - Debt to Equity Ratio = Total Liabilities/Total Shareholders' Equity

#https://www.investopedia.com/terms/d/debtequityratio.asp



nike_DERatio= nike_BS["Total Liabilities"]/ (nike_BS["Total Assets"] - nike_BS["Total Liabilities"])



nike_DERatio
nike_DERatio.plot()
#Calculation 5: Interest Coverage Ratio

#Using this formula - Interest Coverage Ratio = EBIT/Interest Expense

#https://www.investopedia.com/terms/i/interestcoverageratio.asp

#However, Nike does not have Interest Expense in its financial reports. So the Interest Coverage Ratio can not be calculated. 
#Calculation 6: Inventory Turnover Ratio

#Using this formula: Inventory Turnover = COGS/Inventory

#https://www.investopedia.com/ask/answers/070914/how-do-i-calculate-inventory-turnover-ratio.asp



nike_inventoryturnover = -nike_PL["Cost of revenue"]/ nike_BS["Inventories"]



nike_inventoryturnover 
nike_inventoryturnover.plot()
#Calculate days' sales in inventory.

nike_DSinventory = 365 / nike_inventoryturnover 



nike_DSinventory
nike_DSinventory.plot()
#Calculation 7: Receivables Turnover Ratio

#Using this formula: Receivables Turnover = Sales/Accounts Receivable

#https://www.investopedia.com/terms/r/receivableturnoverratio.asp



nike_receivableturnover = nike_PL["Revenue"] / nike_BS["Accounts & Notes Receivable"]



nike_receivableturnover
nike_receivableturnover.plot()
#Calculate days' sales in receivables.

nike_DSreceivable = 365 / nike_receivableturnover



nike_DSreceivable
nike_DSreceivable.plot()
#Calculation 8: Working Capital Turnover Ratio

#Using this formula: Working Capital Turnover = Sales/Working Capital

#https://www.investopedia.com/terms/w/workingcapitalturnover.asp



nike_WCturnover = nike_PL["Revenue"] / (nike_BS["Total Current Assets"] - nike_BS["Total Current Liabilities"])



nike_WCturnover
nike_WCturnover.plot()
#Calculation 9: Total Assets Turnover Ratio

#Using this formula: Total Assets Turnover = Sales/Total Assets

#https://www.investopedia.com/terms/a/assetturnover.asp



nike_assetsturnover = nike_PL["Revenue"] / nike_BS["Total Assets"]



nike_assetsturnover
nike_assetsturnover.plot()
#Calculation 10: Net Profit Margin

#Using this formula: Net Profit Margin = Net Income/Sales

#https://www.investopedia.com/terms/p/profitmargin.asp



nike_netprofitmargin = nike_PL["Net Income Available to Common Shareholders"] / nike_PL["Revenue"]



nike_netprofitmargin
nike_netprofitmargin.plot()
#Calculation 11: Return on Equity Ratio

#Using this formula - ROE = Net income/Sharehholders Equity

#https://www.investopedia.com/terms/r/returnonequity.asp



nike_ROE= nike_CF["Net Income/Starting Line"]/ (nike_BS["Total Assets"] - nike_BS["Total Liabilities"])



nike_ROE
nike_ROE.plot()
#Calculation 12: Return on Asset Ratio

#Using this formula - ROA = Net income/Total Assets

#https://www.investopedia.com/terms/r/returnonassets.asp



nike_ROA= nike_CF["Net Income/Starting Line"]/nike_BS["Total Assets"] 



nike_ROA
nike_ROA.plot()
nike_CF["Price per Share"] = [18.64,21.68,25.23,32.4,40.43,55.13,56.25,55.71,72.94,86.73]

 

nike_CF.columns
#Calculation 13: Price to Earnings Ratio

#Using this formula- PE Ratio = Price per Share/ Earnings per Share

#https://www.investopedia.com/terms/p/price-earningsratio.asp

#Nike's average price per share annually from 2010 tp 2019 is collected from https://www.macrotrends.net/stocks/charts/NKE/nike/stock-price-history

#Earning per Share = Net income/ Shares Outstanding



nike_PEratio = nike_CF["Price per Share"] / nike_EPS



nike_PEratio

nike_PEratio.plot()
#Calculation 14: Market to Book Ratio

#Using this formula - Market to Book Ratio = Market Value per Share/Book Value per Share

#



nike_MBratio = nike_CF["Price per Share"] / (nike_bookvalue / nike_CF["Annual Shares Outstanding"])



nike_MBratio
nike_MBratio.plot()
#Calculation 15: Price-to-Earnings Growth Ratio (PEG forward) of 2019 

#Using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate

#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp



# FROM YAHOO FINANCE: https://finance.yahoo.com/quote/nke/analysis/

GROWTH_RATE = 0.1678 # Five-year expected growth

PEratio_2019 = 34.829769



nike_PEGratio = PEratio_2019 / (GROWTH_RATE*100)



nike_PEGratio



print("NIKE's PEG ratio of 2019 is", nike_PEGratio)