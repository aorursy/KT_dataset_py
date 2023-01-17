#!pip install plotly==4.4.1

#!pip install chart_studio

#!pip install xlrd 
import os

import datetime

import pandas as pd

import numpy as np

import pandas_datareader.data as web





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
_, skechers_PL, skechers_BS, skechers_CF = excel_to_df("/kaggle/input/simfindataskechers/SimFin-data-Skechers.xlsx")
skechers_BS
skechers_BS["_Total Current Assets"] = skechers_BS["Cash, Cash Equivalents & Short Term Investments"] + skechers_BS["Accounts & Notes Receivable"] + skechers_BS["Inventories"] + skechers_BS["Other Short Term Assets"]
skechers_BS[["_Total Current Assets", "Total Current Assets"]]
skechers_BS["_NonCurrent Assets"] = skechers_BS["Property, Plant & Equipment, Net"] + skechers_BS["Other Long Term Assets"]
skechers_BS["_Total Assets"] = skechers_BS["_NonCurrent Assets"] + skechers_BS["_Total Current Assets"] 
skechers_BS[["_Total Assets","Total Assets"]]
skechers_BS["_Total Liabilities"] = skechers_BS["Total Current Liabilities"] + skechers_BS["Total Noncurrent Liabilities"]
skechers_BS[["_Total Liabilities", "Total Liabilities"]]
skechers_BS["_Total Equity"] = skechers_BS["Total Assets"] - skechers_BS["Total Liabilities"]
skechers_BS[["_Total Equity", "Total Equity"]]
%matplotlib inline

skechers_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

'''



asset_columns = [ x for x in good_stuff.strip().split("\n") ]

asset_columns
skechers_BS[ asset_columns ]
skechers_BS[ asset_columns ].plot()
skechers_BS[ ["Property, Plant & Equipment, Net", "Other Long Term Assets"] ].plot()
good_stufftwo = '''

Payables & Accruals

Short Term Debt

'''



liability_columns = [ x for x in good_stufftwo.strip().split("\n") ]
liability_columns
skechers_BS[ liability_columns ]
skechers_BS[ liability_columns ].plot()
skechers_BS[ ["Long Term Debt", "Other Long Term Liabilities"] ].plot()
equity_columns = '''

Preferred Equity

Share Capital & Additional Paid-In Capital

Retained Earnings

Other Equity

Equity Before Minority Interest

'''



equity_columns = [ x for x in equity_columns.strip().split("\n")]

equity_columns
skechers_BS[ equity_columns ]
skechers_BS[ equity_columns ].plot()
#Net Asset Value = Current Assets - Current Liabilities - Long Term Liabilities



skechers_NAV = skechers_BS["Total Current Assets"] - skechers_BS["Total Current Liabilities"] - skechers_BS["Long Term Debt"]
skechers_NAV
skechers_NAV.plot()
#Working Capital = Current Assets - Current Liabilities

skechers_wc = skechers_BS["Total Current Assets"] - skechers_BS["Total Current Liabilities"]
skechers_wc
skechers_wc.plot()
import chart_studio

# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 

# https://chart-studio.plot.ly/feed/#/



chart_studio.tools.set_credentials_file(username='ZhuangYichun', api_key='wLrwgIkcbNrXm9DGNtpD')
import chart_studio.plotly as py

import plotly.graph_objs as go

#from tk_library_py import combine_regexes

# Chart of Balance Sheet

assets = go.Bar(

    x=skechers_BS.index,

    y=skechers_BS["Total Assets"],

    name='Assets'

)

liabilities = go.Bar(

    x=skechers_BS.index,

    y=skechers_BS["Total Liabilities"],

    name='Liabilities'

)



shareholder_equity = go.Scatter(

    x=skechers_BS.index,

    y=skechers_BS["Total Equity"],

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

        x=skechers_BS.index,

        y=skechers_BS[ col ],

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

Long Term Debt

Other Long Term Liabilities

'''





for col in columns.strip().split("\n"):

    liability_bar = go.Bar(

        x=skechers_BS.index,

        y=skechers_BS[ col ],

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

        x=skechers_BS.index,

        y=skechers_BS[ col ],

        name=col

    )    

    equity_data.append(equity_Scatter)

    

layout_equity = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

fig_bs.show()
skechers_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
# Chart of Account Reveivables & Account Payables



PR_data = []

columns = '''

Accounts & Notes Receivable

Payables & Accruals

'''



for col in columns.strip().split("\n"):

    PR_Scatter = go.Scatter(

        x=skechers_BS.index,

        y=skechers_BS[ col ],

        name=col

    )    

    PR_data.append(PR_Scatter)

    

layout_PR = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

fig_bs.show()
# SKECHERS has no preferred stock, no intengible assets, and no goodwill



skechers_bookvalue = skechers_BS["Total Assets"] - skechers_BS["Total Liabilities"]

skechers_bookvalue
skechers_bookvalue.plot()
CF_columns = '''

Cash from Operating Activities

Cash from Investing Activities

Cash from Financing Activities

'''



CF_columns = [ x for x in CF_columns.strip().split("\n")]

skechers_CF[ CF_columns ]
skechers_CF[ CF_columns ].plot()
# Using Plotly

CF_data = []

columns = '''

Cash from Operating Activities

Cash from Investing Activities

Cash from Financing Activities

'''





for col in columns.strip().split("\n"):

    CF_Scatter = go.Scatter(

        x=skechers_CF.index,

        y=skechers_CF[ col ],

        name=col

    )    

    CF_data.append(CF_Scatter)

    

layout_CF = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

fig_bs.show()
#Calculate Earning per Share

#Using this formula - EPS = (Net Income - Preferred Dividends)/ Common Shares Outstanding

#https://www.investopedia.com/terms/e/eps.asp



skechers_EPS = skechers_CF["Net Income/Starting Line"]/skechers_CF["Annual Shares Outstanding"]
skechers_EPS
skechers_EPS.plot()
#2010-2019, 10 years of data, but only 9 years to grow, so the number of time periods is nine. 



present_value = float(skechers_EPS[0])

future_value = float(skechers_EPS[-1])

time_period= len(skechers_EPS)
print(present_value)

print(future_value)

print(time_period)
def eps_growth(present_value, future_value, time_period):

    return (future_value / present_value) ** (1 / (time_period - 1)) - 1



skechers_eps_growth = eps_growth(present_value, future_value, time_period)

print(skechers_eps_growth)
print("{:.2%}".format(skechers_eps_growth.real))
def tenyeareps_forecast (firstyear_value, eps_growth, periods):

    return firstyear_value * ((1 + eps_growth) ** periods)
firstyear_value = 2.77

eps_growth = 0.129

periods = 10

skechers_tenyeareps_forecast = tenyeareps_forecast(firstyear_value, eps_growth, periods)



print(skechers_tenyeareps_forecast)
#From https://ycharts.com/companies/SKX/pe_ratio, the average PE ratio of past five years of SKECHERS is 20.73.



def Estimate_SP (skechers_tenyeareps_forecast, Avg_PE):

    return skechers_tenyeareps_forecast * Avg_PE
Avg_PE=35.20

skechers_tenyear_SP=Estimate_SP (skechers_tenyeareps_forecast, Avg_PE)

print(skechers_tenyear_SP)
#Future Value = 217.21

#Years = 10

#Discount Rate = 0.086, from Finbox.com



import numpy as np



discount_rate = 0.086

future_value = 328.01

PMT = 0

periods = 10



skechers_buyprice = np.pv(discount_rate, periods, PMT, future_value, when='end' )

print(skechers_buyprice)
# Margin of Safety=35%

margin = 0.35

skechers_margin = -skechers_buyprice * (1 - margin)



print(skechers_margin)



#So the target buy price is 93.43.
#Calculation 1: Current Ratio

#Using this formula – Current Ratio = Current Assets / Current Liabilties 

#https://www.investopedia.com/terms/c/currentratio.asp



skechers_BS["current ratio"] = skechers_BS["Total Current Assets"] / skechers_BS["Total Current Liabilities"]



skechers_BS["current ratio"] 
skechers_BS["current ratio"].plot()
#Calculation 2: Quick Ratio

#Using this formula – Quick Ratio = (Current Assets-Inventories) / Current Liabilties 

#https://www.investopedia.com/terms/q/quickratio.asp



skechers_BS["quick ratio"] = (skechers_BS["Total Current Assets"] - skechers_BS["Inventories"]) / skechers_BS["Total Current Liabilities"]



skechers_BS["quick ratio"]
skechers_BS["quick ratio"].plot()
#Calculation 3: Debt Ratio

#Using this formula – Debt Ratio = Total Liabilities / Total Assets

#https://www.investopedia.com/terms/d/debtratio.asp



skechers_BS["debt ratio"] = (skechers_BS["Total Liabilities"]) / skechers_BS["Total Assets"]



skechers_BS["debt ratio"]
skechers_BS["debt ratio"].plot()
#Calculation 4: Debt to Equity ratio

#Using this formula - Debt to Equity Ratio = Total Liabilities/Total Shareholders' Equity

#https://www.investopedia.com/terms/d/debtequityratio.asp



skechers_DERatio= skechers_BS["Total Liabilities"]/ (skechers_BS["Total Assets"] - skechers_BS["Total Liabilities"])



skechers_DERatio
skechers_DERatio.plot()
#Calculation 5: Interest Coverage Ratio

#Using this formula - Interest Coverage Ratio = EBIT/Interest Expense

#https://www.investopedia.com/terms/i/interestcoverageratio.asp



skechers_ICRatio = skechers_PL["Annual EBIT"] / skechers_PL["Interest Expense"]



skechers_ICRatio
skechers_ICRatio.plot()
#Calculation 6: Inventory Turnover Ratio

#Using this formula: Inventory Turnover = COGS/Inventory

#https://www.investopedia.com/ask/answers/070914/how-do-i-calculate-inventory-turnover-ratio.asp



skechers_inventoryturnover = -skechers_PL["Cost of revenue"]/ skechers_BS["Inventories"]



skechers_inventoryturnover 
skechers_inventoryturnover.plot()
#Calculate days' sales in inventory.

skechers_DSinventory = 365 / skechers_inventoryturnover 



skechers_DSinventory
skechers_DSinventory.plot()
#Calculation 7: Receivables Turnover Ratio

#Using this formula: Receivables Turnover = Sales/Accounts Receivable

#https://www.investopedia.com/terms/r/receivableturnoverratio.asp



skechers_receivableturnover = skechers_PL["Revenue"] / skechers_BS["Accounts & Notes Receivable"]



skechers_receivableturnover
skechers_receivableturnover.plot()
#Calculate days' sales in receivables.

skechers_DSreceivable = 365 / skechers_receivableturnover



skechers_DSreceivable
skechers_DSreceivable.plot()
#Calculation 8: Working Capital Turnover Ratio

#Using this formula: Working Capital Turnover = Sales/Working Capital

#https://www.investopedia.com/terms/w/workingcapitalturnover.asp



skechers_WCturnover = skechers_PL["Revenue"] / (skechers_BS["Total Current Assets"] - skechers_BS["Total Current Liabilities"])



skechers_WCturnover
skechers_WCturnover.plot()
#Calculation 9: Total Assets Turnover Ratio

#Using this formula: Total Assets Turnover = Sales/Total Assets

#https://www.investopedia.com/terms/a/assetturnover.asp



skechers_assetsturnover = skechers_PL["Revenue"] / skechers_BS["Total Assets"]



skechers_assetsturnover
skechers_assetsturnover.plot()
#Calculation 10: Net Profit Margin

#Using this formula: Net Profit Margin = Net Income/Sales

#https://www.investopedia.com/terms/p/profitmargin.asp



skechers_netprofitmargin = skechers_PL["Net Income Available to Common Shareholders"] / skechers_PL["Revenue"]



skechers_netprofitmargin
skechers_netprofitmargin.plot()
#Calculation 11: Return on Equity Ratio

#Using this formula - ROE = Net income/Sharehholders Equity

#https://www.investopedia.com/terms/r/returnonequity.asp



skechers_ROE= skechers_CF["Net Income/Starting Line"]/ (skechers_BS["Total Assets"] - skechers_BS["Total Liabilities"])



skechers_ROE
skechers_ROE.plot()
#Calculation 12: Return on Asset Ratio

#Using this formula - ROA = Net income/Total Assets

#https://www.investopedia.com/terms/r/returnonassets.asp



skechers_ROA= skechers_CF["Net Income/Starting Line"]/skechers_BS["Total Assets"] 



skechers_ROA
skechers_ROA.plot()
#Calculation 13: Price to Earnings Ratio

#Using this formula- PE Ratio = Price per Share/ Earnings per Share

#https://www.investopedia.com/terms/p/price-earningsratio.asp

#SKECHER's average price per share annually from 2010 tp 2019 is collected from https://www.macrotrends.net/stocks/charts/SKX/skechers-usa/stock-price-history

#Earning per Share = Net income/ Shares Outstanding



skechers_PEratio = skechers_CF["Price Per Share"] / skechers_EPS



skechers_PEratio

skechers_PEratio.plot()
#Calculation 14: Market to Book Ratio

#Using this formula - Market to Book Ratio = Market Value per Share/Book Value per Share

#https://www.investopedia.com/terms/b/booktomarketratio.asp



skechers_MBratio = skechers_CF["Price Per Share"] / (skechers_bookvalue / skechers_CF["Annual Shares Outstanding"])



skechers_MBratio
skechers_MBratio.plot()
#Calculation 15: Price-to-Earnings Growth Ratio (PEG forward) of 2019 

#Using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate

#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp



#The five-year expected growth rate is collected from YAHOO FINANCE: https://finance.yahoo.com/quote/skx/analysis/?guccounter=1



GROWTH_RATE = 0.1288 

PEratio_2019 = 34.829769



skechers_PEGratio = PEratio_2019 / (GROWTH_RATE*100)



skechers_PEGratio



print("SKECHERS's PEG ratio of 2019 is", skechers_PEGratio)
!pip install yfinance --upgrade --no-cache-dir
!pip install quandl
#Download financial data from yahoo finance

from pandas_datareader import data as pdr





skechers = pdr.get_data_yahoo('SKX', 

                          start=datetime.datetime(2010, 1, 1), 

                          end=datetime.datetime(2020, 3, 10))

skechers.head()
skechers['Volume'].plot(figsize=(10,10))
skechers['Close'].plot()
# Assign `Adj Close` to `daily_close`

daily_close = skechers[['Adj Close']]



# Daily returns

daily_pct_change = daily_close.pct_change()



# Replace NA values with 0

daily_pct_change.fillna(0, inplace=True)



# Inspect daily returns

print(daily_pct_change.head())

daily_log_returns = np.log(daily_close.pct_change()+1)



print(daily_log_returns)
# Resampling the data of skechers by quarter and taking the mean as the value of each quarter

quarter = skechers.resample("4M").mean()



# Calculate the quarterly percentage change

quarter.pct_change()
import matplotlib.pyplot as plt

 

#Draw histogram

daily_pct_change.hist(bins=50)

 



plt.show()

 

# Print a statistical summary of daily_pct_change

print(daily_pct_change.describe())
cum_daily_return = (1 + daily_pct_change).cumprod()



print(cum_daily_return)
cum_daily_return.plot(figsize=(10,10))

 

plt.show()

# Selecting Adjusted Close price

adj_close_px = skechers['Adj Close']

 

# Calculating Moving Average

moving_avg = adj_close_px.rolling(window=40).mean()

 

# Seeing the Last Ten Results

print(moving_avg[-10:])
 # Moving Window Rolling Mean



skechers['5'] = adj_close_px.rolling(window=5).mean()

skechers['10'] = adj_close_px.rolling(window=10).mean()

skechers['30'] = adj_close_px.rolling(window=30).mean()

skechers['60'] = adj_close_px.rolling(window=60).mean()

skechers['120'] = adj_close_px.rolling(window=120).mean()

skechers['240'] = adj_close_px.rolling(window=240).mean()



# Drawing Diagram of Adjusted Closing Price

skechers[['Adj Close', '5', '10', '30', '60', '120', '240']].plot()

 

plt.show()
def get(tickers, startdate, enddate):

  def data(ticker):

    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))

  datas = map (data, tickers)

  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

 

tickers = ['NKE', 'ADDYY', 'UA']

all_data = get(tickers, datetime.datetime(2010, 1, 1), datetime.datetime(2020, 3, 10))
print(all_data.head())
#Selecting the column 'Adj Close' and transform the data box

daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')

 

#Calculating the daily percentage change for 'daily_close_px'

daily_pct_change = daily_close_px.pct_change()

 

# Drawing distribution histogram

daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))

 

plt.show()
# Drawing a scatter matrix graph with 'daily_pct_change' data

pd.plotting.scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1,figsize=(10,10))

 

plt.show()
# Define the Minumum of Periods to Consider 

min_periods = 75 



# Calculate the Volatility

vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods) 



# Plot the Volatility

vol.plot(figsize=(10, 8))



plt.show()