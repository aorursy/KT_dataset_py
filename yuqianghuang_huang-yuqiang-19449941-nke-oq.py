# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Value Investing Stock Analysis with Python



### Topics Covered

# Financial Statements Analysis

# stock price volaility analysis

# Competitor Analysis
# Do the below in kaggle

!pip install plotly==4.4.1

!pip install chart_studio

!pip install xlrd # for reading_excel files with extensions .xlsx into a pandas dataframe
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
# read data from SimFIn-data

_, nike_PL, nike_BS, nike_CF = excel_to_df("/kaggle/input/SimFin-data.xlsx")
#PL data

nike_PL
#CF data

nike_CF
#BS data

nike_BS
## Balance Statments Analysis

# Using formula to calculate the total assets

nike_BS["_Total Current Assets"] = nike_BS["Cash, Cash Equivalents & Short Term Investments"] + nike_BS["Accounts & Notes Receivable"] + nike_BS["Inventories"] + nike_BS["Other Short Term Assets"]
# Vertify the the differnence between orginal data and computing data of total current assets

nike_BS[["_Total Current Assets", "Total Current Assets"]]
# Using formula to calculate nonCurrent assets

nike_BS["_NonCurrent Assets"] = nike_BS["Property, Plant & Equipment, Net"] + nike_BS["Other Long Term Assets"]
# Using formula to calculate total assets

nike_BS["_Total Assets"] = nike_BS["_NonCurrent Assets"] + nike_BS["_Total Current Assets"] 
# Using formula to calculate  total liabilities

nike_BS["_Total Liabilities"] = nike_BS["Total Current Liabilities"] + nike_BS["Total Noncurrent Liabilities"]

#Vertify the the differnence between orginal data and computing data of total liabilities

nike_BS[["_Total Liabilities", "Total Liabilities"]]
# Plot total assets, total liabilities and total equity

%matplotlib inline

nike_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
nike_BS[["Total Assets", "Total Liabilities", "Total Equity"]]
good_stuff = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

'''



current_asset_columns = [ x for x in good_stuff.strip().split("\n") ]


current_asset_columns
good_stuff = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

'''



current_asset_columns = [ x for x in good_stuff.strip().split("\n") ]
good_stuff = '''

Payables & Accruals

Short Term Debt

Other Short Term Liabilities

'''



current_liabilities_columns = [ x for x in good_stuff.strip().split("\n") ]
# Plot current_asset_columns

nike_BS[ current_asset_columns ].plot()

nike_BS[ current_asset_columns ]
# Plot current_asset_columns

nike_BS[ current_liabilities_columns ].plot()

nike_BS[ current_liabilities_columns ]
import chart_studio

# https://chart-studio.plot.ly/feed/#/



# Use my own username and own apikey of chart studio

chart_studio.tools.set_credentials_file(username='Addisonhuang', api_key='19fWtNVSrJyiPZStbLKQ')
import chart_studio.plotly as py

import plotly.graph_objs as go
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

py.plot(fig_bs, filename='Total Assets and Liabilities')

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

        x=nike_BS.index,

        y=nike_BS[ col ],

        name=col

    )    

    asset_data.append(asset_bar)

    

layout_assets = go.Layout(

    barmode='stack'

)



fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)

py.plot(fig_bs_assets, filename='Total Assets Breakdown')

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

        x=nike_BS.index,

        y=nike_BS[ col ],

        name=col

    )    

    liability_data.append(liability_bar)

    

layout_liabilities = go.Layout(

    barmode='stack'

)



fig_bs_liabilities = go.Figure(data=liability_data, layout=layout_liabilities)

py.plot(fig_bs_liabilities, filename='Total liabilities Breakdown')

fig_bs_liabilities.show()
nike_BS["working capital"] = nike_BS["Total Current Assets"] - nike_BS["Total Current Liabilities"]
nike_BS[["working capital"]].plot()

nike_BS[["working capital"]]
nike_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()

nike_BS[["Accounts & Notes Receivable", "Payables & Accruals"]]
# Using Chart Studio in Plotly 



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



fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)

py.plot(fig_bs_PR, filename='Accounts & Notes Receivable vs Payables & Accruals')

fig_bs_PR.show()
nike_BS["Inventories"].plot()

nike_BS["Inventories"]
nike_BS[ ["Property, Plant & Equipment, Net", "Other Long Term Assets"] ].plot()

nike_BS[ ["Property, Plant & Equipment, Net", "Other Long Term Assets"] ]
# Using Plotly



AAA_data = []

columns = '''

Property, Plant & Equipment, Net

Other Long Term Assets

'''





for col in columns.strip().split("\n"):

    AAA_bar = go.Bar(

        x=nike_BS.index,

        y=nike_BS[ col ],

        name=col

    )    

    AAA_data.append(AAA_bar)

    

layout_AAA = go.Layout(

    barmode='stack'

)



fig_bs_AAA = go.Figure(data=AAA_data, layout=layout_AAA)

py.plot(fig_bs_AAA, filename='Total Long Term Assets')

fig_bs_AAA.show()
equity_columns = '''

Share Capital & Additional Paid-In Capital

Retained Earnings

Other Equity

Equity Before Minority Interest

Preferred Equity

'''



equity_columns = [ x for x in equity_columns.strip().split("\n")]
equity_columns
nike_BS[ equity_columns ].plot()

nike_BS[ equity_columns ]
# Using Plotly



equity_data = []

columns = '''

Share Capital & Additional Paid-In Capital

Retained Earnings

Other Equity

Equity Before Minority Interest

Preferred Equity

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



fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)

py.plot(fig_bs_equity, filename='Total Equity')

fig_bs_equity.show()
CF_columns = '''

Cash from Operating Activities

Cash from Investing Activities

Cash from Financing Activities

'''



CF_columns = [ x for x in CF_columns.strip().split("\n")]
nike_CF[ CF_columns ].plot()

nike_CF[ CF_columns ]
# Using Plotly



CF_data = []

columns = '''

Cash from Operating Activities

Cash from Investing Activities

Cash from Financing Activities

'''





for col in columns.strip().split("\n"):

    CF_Scatter = go.Scatter(

        x=nike_CF.index,

        y=nike_CF[ col ],

        name=col

    )    

    CF_data.append(CF_Scatter)

    

layout_CF = go.Layout(

    barmode='stack'

)



fig_CF = go.Figure(data=CF_data, layout=layout_equity)

py.plot(fig_CF, filename='CF')

fig_CF.show()
# Nike  has no preferred stock, no intengible assets, and no goodwill



nike_BS["book value"] = nike_BS["Total Assets"] - nike_BS["Total Liabilities"]

# nike  has no preferred stock, no intengible assets, and no goodwill



nike_BS[ "book value"].plot()

nike_BS[ "book value"]
# liquidation value = company real estate+ fixtures + equipment + inventory

nike_BS["liquidation value"] =nike_BS["Inventories"] + nike_BS["Property, Plant & Equipment, Net"]

nike_BS["liquidation value"].plot()

nike_BS["liquidation value"]
nike_PL["Revenue"]
nike_PL["Revenue"].plot()
nike_Revenue = nike_PL["Revenue"]

nike_Revenue
revenue_growth = nike_Revenue[-1]/ nike_Revenue[0]

revenue_growth

revenue_compounded =pow(revenue_growth,1/9)-1

revenue_compounded

print('%.2f%%' % (revenue_compounded* 100))

revenue2020=  nike_Revenue[-1]*(1+revenue_compounded)

revenue2020
!pip install yfinance --upgrade --no-cache-dir
!pip install quandl
#download financial data from yahoo finance

from pandas_datareader import data as pdr





NKE = pdr.get_data_yahoo('NKE', 

                          start=datetime.datetime(2006, 10, 1), 

                          end=datetime.datetime(2020, 1, 18))

NKE.head()
NKE['Volume'].plot(figsize=(12,8))
NKE['Close'].plot()
# calculate volatility

# source fromm:https://github.com/datacamp/datacamp-community-tutorials/blob/master/Python%20Finance%20Tutorial%20For%20Beginners/Python%20For%20Finance%20Beginners%20Tutorial.ipynb

# Assign `Adj Close` to `daily_close`

daily_close = NKE[['Adj Close']]



# Daily returns

daily_pct_c = daily_close.pct_change()



# Replace NA values with 0

daily_pct_c.fillna(0, inplace=True)



# Inspect daily returns

print(daily_pct_c)



# Daily log returns

daily_log_returns = np.log(daily_close.pct_change()+1)



# Print daily log returns

print(daily_log_returns)
# Resample `NKE` to business months, take last observation as value 

monthly = NKE.resample('BM').apply(lambda x: x[-1])



# Calculate the monthly percentage change

monthly.pct_change()



# Resample `NKE` to quarters, take the mean as value per quarter

quarter = NKE.resample("4M").mean()



# Calculate the quarterly percentage change

quarter.pct_change()
# Daily returns

daily_pct_c = daily_close / daily_close.shift(1) - 1



# Print `daily_pct_c`

print(daily_pct_c)
# Import matplotlib

import matplotlib.pyplot as plt



# Plot the distribution of `daily_pct_c`

daily_pct_c.hist(bins=50)



# Show the plot

plt.show()



# Pull up summary statistics

print(daily_pct_c.describe())
# Calculate the cumulative daily returns

cum_daily_return = (1 + daily_pct_c).cumprod()



# Print `cum_daily_return`

print(cum_daily_return)
# Import matplotlib

import matplotlib.pyplot as plt 



# Plot the cumulative daily returns

cum_daily_return.plot(figsize=(12,8))



# Show the plot

plt.show()
# Resample the cumulative daily return to cumulative monthly return 

cum_monthly_return = cum_daily_return.resample("M").mean()



# Print the `cum_monthly_return`

print(cum_monthly_return)
# Isolate the adjusted closing prices 

adj_close_px = NKE['Adj Close']



# Calculate the moving average

moving_avg = adj_close_px.rolling(window=40).mean()



# Inspect the result

moving_avg[-10:]
# moving window rolling mean



NKE['5'] = adj_close_px.rolling(window=5).mean()

NKE['20'] = adj_close_px.rolling(window=20).mean()

NKE['60'] = adj_close_px.rolling(window=60).mean()

NKE['120'] = adj_close_px.rolling(window=120).mean()

NKE['250'] = adj_close_px.rolling(window=250).mean()



# Plot the adjusted closing price, the short and long windows of rolling means

NKE[['Adj Close', "5","20",'60',"120",'250']].plot(figsize=(18,10))

plt.show()



from pandas_datareader import data as pdr





def get(tickers, startdate, enddate):

    def data(ticker):

        return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))

    datas = map (data, tickers)

    return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))



tickers = ['NKE',"UA","SKX","CROX"]

all_data = get(tickers, datetime.datetime(2010, 1, 1), datetime.datetime(2020, 2, 18))

all_data.head()
daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')



# Calculate the daily percentage change for `daily_close_px`

daily_pct_change = daily_close_px.pct_change()



# Plot the distributions

daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))



# Show the resulting plot

plt.show()
# Define the minumum of periods to consider 

min_periods = 75 



# Calculate the volatility

vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods) 



# Plot the volatility

vol.plot(figsize=(10, 8))



# Show the plot

plt.show()
#Short Term Solvency Ratio
#current ratio

nike_BS["current ratio"] = nike_BS["Total Current Assets"] / nike_BS["Total Current Liabilities"]

nike_BS["current ratio"].plot()

nike_BS["current ratio"]
#quick ratio

nike_BS["quick ratio"] = (nike_BS["Total Current Assets"] -nike_BS["Inventories"])/ nike_BS["Total Current Liabilities"]

nike_BS["quick ratio"].plot()

nike_BS["quick ratio"]
# cash ratio

nike_BS["cash ratio"] = nike_BS["Cash, Cash Equivalents & Short Term Investments"] / nike_BS["Total Current Liabilities"]

nike_BS["cash ratio"].plot()

nike_BS["cash ratio"]
#Long Term Solvency Ratio
#Liablities to Asset Ratio

nike_BS["Liablities to Asset Ratio"] = nike_BS["Total Liabilities"] / nike_BS["Total Assets"]

nike_BS["Liablities to Asset Ratio"].plot()

nike_BS["Liablities to Asset Ratio"]
#Debt to Equity Ratio
nike_BS["Debt to Equity Ratio"] = nike_BS["Total Liabilities"] / nike_BS["Total Equity"]

nike_BS["Debt to Equity Ratio"].plot()

nike_BS["Debt to Equity Ratio"]
#profitability ratio
#gross profit ratio

nike_PL["Gross Profit Ratio"] = (nike_PL["Revenue"]+nike_PL["Cost of revenue"])/ nike_PL["Revenue"]

nike_PL["Gross Profit Ratio"].plot()

nike_PL["Gross Profit Ratio"]
# Net Income Ratio

nike_PL["Net Income Ratio"] = nike_PL["Net Income Available to Common Shareholders"] / nike_PL["Revenue"]

nike_PL["Net Income Ratio"].plot()

nike_PL["Net Income Ratio"]
# Return on Equity

Return_on_Equity = nike_PL["Net Income Available to Common Shareholders"] / nike_BS["Total Equity"]

Return_on_Equity.plot()

Return_on_Equity
# Return on Asset

Return_on_Asset = nike_PL["Net Income Available to Common Shareholders"] / nike_BS["Total Assets"]

Return_on_Asset.plot()

Return_on_Asset
# caculate NI / Long Term Debt

ni_ltd_ratio = nike_PL["Net Income Available to Common Shareholders"] / nike_BS["Long Term Debt"]

ni_ltd_ratio.plot()

ni_ltd_ratio
#using simfin to down load nike financial data

!pip install simfin

# Import the main functionality from the SimFin Python API.

import simfin as sf



# Import names used for easy access to SimFin's data-columns.

from simfin.names import *


sf.set_data_dir('~/simfin_data/')


sf.set_api_key(api_key='free')
# sf.load_api_key(path='~/simfin_api_key.txt', default_key='free')
df_companies = sf.load_companies(index=TICKER, market='us')
df_companies.loc['NKE']
df_industries = sf.load_industries()
industry_id = df_companies.loc['NKE'][INDUSTRY_ID]

industry_id
df_industries.loc[industry_id]
df1 = sf.load(dataset='income', variant='annual', market='us',

              index=[TICKER, REPORT_DATE],

              parse_dates=[REPORT_DATE, PUBLISH_DATE])
df1.columns
df1.loc['NKE']["Selling, General & Administrative"]

df1.loc['NKE']["Cost of Revenue"]
variable_cost = -df1.loc['NKE']["Cost of Revenue"]

variable_cost 
contribution_margin=(df1.loc['NKE']["Revenue"]-variable_cost)/ df1.loc['NKE']["Revenue"]

contribution_margin
df1.columns
df1.loc['NKE']["Operating Expenses"]
df1.loc['NKE']["Interest Expense, Net"]
fixed_cost=-(df1.loc['NKE']["Operating Expenses"]+df1.loc['NKE']["Interest Expense, Net"])

fixed_cost
breakeven_point = fixed_cost/contribution_margin

breakeven_point
margin_of_safety = (df1.loc['NKE']["Revenue"]-breakeven_point)/df1.loc['NKE']["Revenue"]

margin_of_safety
margin_of_safety.mean()
net_income = df1.loc['NKE']["Net Income"]

net_income
Revenue= df1.loc['NKE']["Revenue"]

Revenue
Revenue.plot()
df1.loc['NKE']["Shares (Basic)"]

df1.loc['NKE']["Shares (Diluted)"]
(df1.loc['NKE']["Net Income"] /df1.loc['NKE']["Shares (Diluted)" ]).plot()
# Caculate Basic EPS

Eps = df1.loc['NKE']["Net Income"] /df1.loc['NKE']["Shares (Basic)" ]

Eps
# Caculate Diluted EPS

Eps_diluted = df1.loc['NKE']["Net Income"] /df1.loc['NKE']["Shares (Diluted)" ]

Eps_diluted
a= Eps[0]

EPS2019= 2.55 #FROM MSN WEBSITE: https://www.msn.com/en-us/money/stockdetails/financials/nys-nke/fi-a1yjxm

print(a)

print(EPS2019)
growth = (EPS2019/a)

compounded = pow(growth,1/10) - 1

print('%.2f%%' % (compounded * 100))

EPS_anticipation = []

for i in range(1,11):

    anticipate = EPS2019 *( compounded + 1)**i

    EPS_anticipation.append(anticipate)

print(EPS_anticipation)
EPS_anticipation

from pandas.core.frame import DataFrame

index1 = np.arange(2020,2030)



EPS_predicted={ "Year":index1,"EPS_anticipation": EPS_anticipation}

EPS_predicted= DataFrame(EPS_predicted)

EPS_predicted
eps2029 = EPS_anticipation[-1]

eps2029
# average p/e ratio from YCHARTS website: https://ycharts.com/companies/NKE/pe_ratio

Average_PE_Ratio = 35.20

Average_PE_Ratio
estimated_stock_price= Average_PE_Ratio* eps2029

estimated_stock_price

# WACC = 8.03%  from gurufocus website:https://www.gurufocus.com/term/wacc/NYSE:NKE/WACC-/Nike

def get_present_value(total_year,future_price,annual_discount_rate):

        present_value=future_price/(1+annual_discount_rate)**total_year      

        return present_value

annual_discount_rate = 0.083

target_buy_price2020 = get_present_value(9,estimated_stock_price,annual_discount_rate)

round(target_buy_price2020,2)

target_buy_price2020
# (Current Sales Level – Breakeven Point) ÷ Current Sales Level = Margin of safety

After_margin_of_safety = target_buy_price2020 *(1-0.2940)

After_margin_of_safety
#Calculate 1.	Price-to-Earnings Growth Ratio (PEG forward)  

#using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate

#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp



PE_RATIO = 35.20# FROM SIMFIN WEBSITE: https://simfin.com/data/companies/62747



# FROM NASDAQ WEBSITE: https://finance.yahoo.com/quote/nke/analysis/

GROWTH_RATE = 0.1678 # Forcast over the five next years



PEG_ratio = PE_RATIO / (GROWTH_RATE*100)



print("nike's PEG Ratio is", PEG_ratio)
df1.loc['NKE']["Pretax Income (Loss)"]
df1.loc['NKE']["Interest Expense, Net"]
#calculate interest coverage ratio

df1.loc['NKE']["Pretax Income (Loss)"]/-df1.loc['NKE']["Interest Expense, Net"]
#calculate D/E ratio

nike_BS["Debt to Equity Ratio"] = nike_BS["Total Liabilities"] / nike_BS["Total Equity"]

nike_BS["Debt to Equity Ratio"].plot()

nike_BS["Debt to Equity Ratio"]
# Using SimFin api  to download detailed balance statement

df2 = sf.load(dataset='balance', variant='annual', market='us',

              index=[TICKER, REPORT_DATE],

              parse_dates=[REPORT_DATE, PUBLISH_DATE])
df2.loc['NKE']
df2.columns
df2.loc['NKE']["Cash, Cash Equivalents & Short Term Investments"]
#caluculate net asset value

NAV = df2.loc['NKE']["Total Assets"]-df2.loc['NKE']["Total Liabilities"]

NAV.plot()

NAV
# Using SimFin api  to download detailed balance statement

df3 = sf.load(dataset='cashflow', variant='annual', market='us',

              index=[TICKER, REPORT_DATE],

              parse_dates=[REPORT_DATE, PUBLISH_DATE])
df3.loc['NKE']
# Doing the competitor analysis/Correlation

# FROM Github WEBSITE : https://github.com/liyangbit/PyDataRoad/blob/master/projects/Stock-prediction-with-Python/Lesson%203%20%20Basic%20Python%20for%20Data%20Analytics%20(Stocks%20Prediction).ipynb
start = datetime.datetime(2009, 1, 1)

end = datetime.datetime.now()



df = web.DataReader("NKE", 'yahoo', start, end)

df.tail()



dfcomp = web.DataReader(['NKE', 'UA', 'SKX', 'CROX'],'yahoo',

                               start=start, 

                               end=end)['Adj Close']

dfcomp.tail()
dfcomp.shape
retscomp = dfcomp.pct_change()



corr = retscomp.corr()

corr
import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib import style

from pandas import Series, DataFrame

plt.scatter(retscomp.NKE, retscomp.UA)

plt.xlabel('Returns NKE')

plt.ylabel('Returns UA')
pd.plotting.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10));
plt.imshow(corr, cmap='hot', interpolation='none')

plt.colorbar()

plt.xticks(range(len(corr)), corr.columns)

plt.yticks(range(len(corr)), corr.columns);
plt.scatter(retscomp.mean(), retscomp.std())

plt.xlabel('Expected returns')

plt.ylabel('Risk')

for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):

    plt.annotate(

        label, 

        xy = (x, y), xytext = (20, -20),

        textcoords = 'offset points', ha = 'right', va = 'bottom',

        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),

        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))