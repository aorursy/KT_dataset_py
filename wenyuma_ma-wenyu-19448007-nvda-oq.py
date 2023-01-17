!pip install plotly
!pip install chart_studio
!pip install xlrd
!pip install simfin
!pip install quandl
!pip install pandas_datareader
#Use simfin API to download data
import simfin as sf
sf.set_api_key('free')
sf.set_data_dir('~/input/')

us_pl = sf.load_income(dataset='income', variant='annual', market='us')
us_bs = sf.load_balance(dataset='balance', variant='annual', market='us')
us_cf = sf.load_cashflow(dataset='cashflow', variant='annual', market='us')
#download NVDA's cash flow
NVDA_cf = us_cf.loc['NVDA']
#download NVDA's income statement
NVDA_pl = us_pl.loc['NVDA']
#download NVDA's balance sheet
NVDA_bs = us_bs.loc['NVDA']
NVDA_bs
NVDA_bs.columns
#calculate NVDA's total shareholers' Equity
NVDA_bs["_Total Shareholders' Equity"] = NVDA_bs["Total Assets"] - NVDA_bs["Total Liabilities"]
#Total Shareholders' Equity for NVDA
NVDA_bs[["_Total Shareholders' Equity", "Total Equity"]]
#Total Current Asset Breakdown for NVDA
NVDA_bs["_Total Current Assets"] = NVDA_bs["Cash, Cash Equivalents & Short Term Investments"] + NVDA_bs["Accounts & Notes Receivable"] + NVDA_bs["Inventories"] 
NVDA_bs[["_Total Current Assets", "Total Current Assets"]]
#Total Current Liability Breakdown
NVDA_bs["_Total Current Liabilities"] = NVDA_bs["Payables & Accruals"] + NVDA_bs["Short Term Debt"]
NVDA_bs[["_Total Current Liabilities", "Total Current Liabilities"]]
#Net Asset Value
#Net Asset Value = Current Assets - Current Liabilities - Long Term Liability
NVDA_bs["Net Asset Value"] = NVDA_bs["Total Current Assets"] - NVDA_bs["Total Current Liabilities"] - NVDA_bs["Long Term Debt"]
NVDA_bs[["Net Asset Value"]]
#Working Capital
#Working Capital = Current Assets - Current Liabilities
NVDA_bs["Working Capital"] = NVDA_bs["Total Current Assets"] - NVDA_bs["Total Current Liabilities"]
NVDA_bs["Working Capital"]
NVDA_bs[["Total Current Assets","Total Current Liabilities"]]
NVDA_bs["Working Capital"].plot()
#Owner's Equity or Book Value
#BV = Total Assets - Intangible assets - Liabilities - Preferred Stock Value
NVDA_bs["Book value"] = NVDA_bs["Total Assets"] - NVDA_bs["Total Liabilities"]
NVDA_bs["Book value"]
NVDA_bs["Book value"].plot()
#Current Ratio
#Current Ratio = Total Current Assets / Total Current Liabilities
NVDA_bs["Current Ratio"] = NVDA_bs["Total Assets"] / NVDA_bs["Total Liabilities"]
NVDA_bs["Current Ratio"]
NVDA_bs["Current Ratio"].plot()
from IPython.display import Image
import os
Image("../input/PE.simfin.png")
from IPython.display import Image
import os
Image("../input/PEG.png")
#Price to Equity Growth (PEG forward)
#PEG = Price-to-Earnings Ratio / Earnings-Growth-Rate
PE_RATIO = 73.23  #From SIMFIN WEBSITE: https://simfin.com/data/companies/172199

#From NASDAQ WEBSITE: https://www.nasdaq.com/market-activity/stocks/nvda/price-earnings-peg-ratios, 
GROWTH_RATE = 0.171 #Forcase over the five next years

PEG_ratio = PE_RATIO / (GROWTH_RATE*100)

print("NVDA Corp's PEG Ratio is", PEG_ratio)
NVDA_pl.columns
#Earnings Per Share(EPS) Annual Compounded Growth Rate
#NVDA's basic earnings per share as follows: Profits - Dividends / Weighted Average Shares 
NVDA_EPS = NVDA_pl["Net Income"] / NVDA_bs["Shares (Basic)"]
NVDA_EPS
NVDA_EPS.plot()
#From 2010 to 2019, nine years periods
start_value = float(NVDA_EPS.iloc[6])
end_value = float(NVDA_EPS.iloc[-1])
num_periods = len(NVDA_EPS) -1
print(start_value)
print(end_value)
print(num_periods)
def cagr(start_value, end_value, num_periods):
    return (end_value / start_value) ** (1 / (num_periods - 1)) - 1
NVDA_cagr = cagr(start_value, end_value, num_periods)
print(NVDA_cagr)
format(float(NVDA_cagr.real))
print("{:.2%}".format(NVDA_cagr.real))
#Estimate EPS 10 years from now
#Present Value = $6.79
#Growth Rate = 21.67%
#Years = 10
def future (present_value, growth_rate, periods):
    return present_value * ((1 + growth_rate) ** periods)
present_value = float(NVDA_EPS.iloc[-1])
growth_rate = NVDA_cagr
periods = 10
NVDA_future = future(present_value, growth_rate, periods)
print(NVDA_future)
from IPython.display import Image
import os
Image("../input/Avg.PE.png")
#Find historical PE is from ycharts.com
#PE Ratio Range of NVDA, Past 5 Years is 37.62 from:www.ycharts.com
def Est_SP (NVDA_future, Avg_PE):
    return NVDA_future * Avg_PE
Avg_PE = 37.62
NVDA_future_SP = Est_SP (NVDA_future, Avg_PE)
print(NVDA_future_SP)
#Determine Target buy price
#Future Value = 1816
#Years = 10
#Discount Rate = 52.15, which is euqal to NVDA's cagr
import numpy as np

interest_rate = 0.1585
future_value = NVDA_future_SP
PMT = 0
periods = 10

NVDA_Est_PV = np.pv(interest_rate, periods, PMT, future_value, when='end' )
print(-NVDA_Est_PV)
#Becasue the data sample is up to 2019
#If I predict the 2020's stock value with this data
interest_rate = 0.1585
future_value = NVDA_future_SP
PMT = 0
periods = 9

NVDA_2020 = -np.pv(interest_rate, periods, PMT, future_value, when='end' )
print(NVDA_2020)
NVDA_pl.columns
NVDA_Groos_Profit_Margin = (NVDA_pl["Revenue"] - NVDA_pl["Cost of Revenue"]) / NVDA_pl["Revenue"]
NVDA_Groos_Profit_Margin
NVDA_Fixed_Expense = NVDA_pl["Selling, General & Administrative"] + NVDA_pl["Research & Development"] + NVDA_pl["Operating Expenses"] + NVDA_pl["Interest Expense, Net"]
-NVDA_Fixed_Expense
NVDA_break_even_margin = -NVDA_Fixed_Expense / NVDA_Groos_Profit_Margin
NVDA_break_even_margin
NVDA_pl["Revenue"]
NVDA_Margin_Safety = (NVDA_pl["Revenue"] - NVDA_break_even_margin) / NVDA_pl["Revenue"]
NVDA_Margin_Safety
NVDA_Margin_Safety.mean()
print("{:.2%}".format(NVDA_Margin_Safety.mean()))
#Margin of Safety 
margin = NVDA_Margin_Safety.mean()
NVDA_margin = -NVDA_Est_PV * (1 - margin)
print(NVDA_margin)
#NVDA_2020 with round to 52.93% margin， 
NVDA_2020_margin = NVDA_2020 * (1 - NVDA_Margin_Safety.mean())
NVDA_2020_margin
import pandas_datareader.data as web
from pandas_datareader import data
start_2019 = '2019-1-31'
end_2019 = '2019-1-31'

NVDA_google = data.DataReader('NVDA','yahoo', start_2019, end_2019)
print(NVDA_google)
start_now = '2020-2-14'
end_now = '2020-2-14'

NVDA_now = data.DataReader('NVDA','yahoo', start_now, end_now)
print(NVDA_now)
#Debt to Equity Ratio
NVDA_DER = NVDA_bs["Total Liabilities"] / NVDA_bs["Total Equity"]
NVDA_DER
NVDA_DER.plot()
NVDA_pl.columns
#Interest Coverage Ratio
NVDA_EBIT = NVDA_pl["Revenue"] - NVDA_pl["Cost of Revenue"] - NVDA_pl["Operating Expenses"]
NVDA_EBIT
NVDA_EBIT.plot()
NVDA_ICR = NVDA_EBIT / NVDA_pl["Interest Expense, Net"]
NVDA_ICR
NVDA_ICR.plot()
NVDA_bs.columns
#Liquidity Ratio
#Current Ratio
NVDA_CR = NVDA_bs["Total Current Assets"] / NVDA_bs["Total Current Liabilities"]
NVDA_CR
NVDA_CR.plot()
NVDA_bs.mean()
NVDA_bs.mean().Inventories
NVDA_pl.columns
#Inventories Turnover
NVDA_Inv_Turnover = NVDA_pl["Cost of Revenue"] / NVDA_bs.mean().Inventories
NVDA_Inv_Turnover
NVDA_Inv_Turnover.plot()
#Receivables Turnover
NVDA_bs.mean()
NVDA_bs["Accounts & Notes Receivable"].mean()
NVDA_Rec_Turnover = NVDA_pl["Revenue"] / NVDA_bs["Accounts & Notes Receivable"].mean()
NVDA_Rec_Turnover
#Payables turnover ratio
NVDA_bs["Payables & Accruals"].mean() 
NVDA_Pay_Turnover = NVDA_pl["Cost of Revenue"] / NVDA_bs["Payables & Accruals"].mean() 
NVDA_Pay_Turnover
#days of sales outstanding 
NVDA_days_sales = 365 / NVDA_Rec_Turnover
#days of inventory on hand
NVDA_days_Inv = 365 / NVDA_Inv_Turnover
#number of days of payables
NVDA_days_Pay = 365 / NVDA_Pay_Turnover
#Cash conversion cycle
NVDA_CCC = NVDA_days_Inv + NVDA_days_sales - NVDA_days_Pay
NVDA_CCC
NVDA_CCC.mean()
NVDA_CCC.plot()
#Return on Equity
NVDA_ROE = NVDA_pl["Net Income"] / NVDA_bs["Total Equity"]
NVDA_ROE
NVDA_ROE.plot()
#Return on Asset
NVDA_ROA = NVDA_pl["Net Income"] / NVDA_bs["Total Assets"]
NVDA_ROA.plot()
import chart_studio
chart_studio.tools.set_credentials_file(username='ma9016', api_key='v4nCqh0dvd0lRXTVrzwX')
import chart_studio.plotly as py
import plotly.graph_objs as go
assets = go.Bar(
    x=NVDA_bs.index,
    y=NVDA_bs["Total Assets"],
    name='Assets'
)
liabilities = go.Bar(
    x=NVDA_bs.index,
    y=NVDA_bs["Total Liabilities"],
    name='Liabilities'
)

shareholder_equity = go.Scatter(
    x=NVDA_bs.index,
    y=NVDA_bs["Total Equity"],
    name='Equity'
)

data = [assets, liabilities, shareholder_equity]
layout = go.Layout(
    barmode='stack'
)

fig_bs = go.Figure(data=data, layout=layout)
fig_bs.show()
py.plot(fig_bs, filename='Total Assets and Liabilities')
asset_data = []
columns = '''
Cash, Cash Equivalents & Short Term Investments
Accounts & Notes Receivable
Inventories
Property, Plant & Equipment, Net
Long Term Investments & Receivables
Other Long Term Assets
'''


for col in columns.strip().split("\n"):
    asset_bar = go.Bar(
        x=NVDA_bs.index,
        y=NVDA_bs[ col ],
        name=col
    )    
    asset_data.append(asset_bar)
    
layout_assets = go.Layout(
    barmode='stack'
)
fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)
fig_bs_assets.show()
py.plot(fig_bs_assets, filename='Total Assets Breakdown')
liability_data = []
columns = '''
Payables & Accruals
Short Term Debt
Long Term Debt
'''


for col in columns.strip().split("\n"):
    liability_bar = go.Bar(
        x=NVDA_bs.index,
        y=NVDA_bs[ col ],
        name=col
    )    
    liability_data.append(liability_bar)
    
layout_liabilitys = go.Layout(
    barmode='stack'
)

fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)
fig_bs_liabilitys.show()
py.plot(fig_bs_liabilitys, filename='Total liabilities Breakdown')
