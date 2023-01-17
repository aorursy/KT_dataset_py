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
!pip install plotly

!pip install chart_studio

!pip install numpy_financial

!pip install simfin

#!pip install cufflinks
import pandas as pd

import chart_studio

import plotly

#import cufflinks as cf

chart_studio.tools.set_credentials_file(username='ma9016', api_key='v4nCqh0dvd0lRXTVrzwX')

#plotly.tools.set_credentials_file(username='ma9016', api_key='v4nCqh0dvd0lRXTVrzwX')

#import plotly.tools as tls

#tls.embed('https://plot.ly/~cufflinks/8')

import chart_studio.plotly as py

import plotly.graph_objs as go

import numpy as np

import numpy_financial as npf

from keras.models import Sequential 

from keras.layers import Dense 

from keras.layers import Dropout

import statsmodels.api as sm

import itertools

import seaborn as sns

import simfin as sf



width = 6

height = 3

import matplotlib

matplotlib.rcParams['figure.figsize'] = [width, height]

import matplotlib.pyplot as plt



plt.style.use('ggplot')
df = pd.read_excel("/kaggle/input/intel-q/Intel_Q.xlsx")

df2 = pd.read_excel("/kaggle/input/intel-data/intel_data.xlsx")
# organize the intel's general information from simfin

index_GI = df.loc[df2["Data provided by SimFin"] == "General Information"].index[0]

index_SI = df.loc[df2["Data provided by SimFin"] == "Stock Information"].index[0]

index_FR = df.loc[df2["Data provided by SimFin"] == "Key Financials & Ratios"].index[0]

#General Information

df2_GI = df2.iloc[index_GI:index_SI-1,1:]

df2_GI.columns = list(df2_GI.iloc[1])

df2_GI.set_index("Common name of company",inplace=True)

df2_GI = df2_GI.iloc[1:,]

df2_GI = df2_GI[1:]

#Stock Information

df2_SI = df2.iloc[index_SI:index_FR-1,1:]

df2_SI.columns = list(df2_SI.iloc[1])

df2_SI.set_index("Share price (USD; as per 2020-03-13)",inplace=True)

df2_SI = df2_SI.iloc[1:,]

df2_SI = df2_SI[1:]

#Key Financials&Ratios

df2_FR = df2.iloc[index_FR:,1:]

df2_FR.columns = list(df2_FR.iloc[1])

df2_FR.set_index("Revenues (in million USD, TTM)",inplace=True)

df2_FR = df2_FR.iloc[1:,]

df2_FR = df2_FR[1:]

#define name

intel_GI = df2_GI.T

intel_SI = df2_SI.T

intel_FR = df2_FR.T

#my "tk-library", to organize financial statement from simfin

index_PL = df.loc[df["Data provided by SimFin"] == 'Profit & Loss statement'].index[0]

index_BS = df.loc[df["Data provided by SimFin"] == 'Balance Sheet'].index[0]

index_CF = df.loc[df["Data provided by SimFin"] == 'Cash Flow statement'].index[0]

#income statement

df_PL = df.iloc[index_PL:index_BS-1,1:]

df_PL.columns = list(df_PL.iloc[1])

df_PL.set_index("in million USD",inplace=True)

df_PL.fillna(0, inplace=True)

df_PL = df_PL.iloc[1:,]

df_PL = df_PL[1:]

#balance sheet

df_BS = df.iloc[index_BS:index_CF-1,1:]

df_BS.columns = list(df_BS.iloc[1])

df_BS.set_index("in million USD",inplace=True)

df_BS.fillna(0, inplace=True)

df_BS = df_BS.iloc[1:,]

df_BS = df_BS[1:]

#cash flow

df_CF = df.iloc[index_CF:,1:]

df_CF.columns = list(df_CF.iloc[1])

df_CF.set_index("in million USD",inplace=True)

df_CF.fillna(0, inplace=True)

df_CF = df_CF.iloc[1:,]

df_CF = df_CF[1:]
intel_PL = df_PL.T

intel_BS = df_BS.T

intel_CF = df_CF.T
%matplotlib inline

intel_BS["Total Shareholers' Equity"] = intel_BS["Total Assets"] - intel_BS["Total Liabilities"]

intel_BS[["Total Shareholers' Equity", "Total Equity"]].plot(title = "Total Shareholder's Equity") #they are same 
intel_BS[["Long Term Debt","Other Long Term Liabilities"]].plot(title = "Long-term debt")
intel_PL["Operating Expenses"].plot(title = "Operating Expenses")
intel_BS["Cash, Cash Equivalents & Short Term Investments"].plot(title = "Cash, Cash Equivalents & Short Term Investments")
intel_PL[["Cost of revenue","Revenue"]].plot(title = "Cost of revenue and Revenue")
#Intel's Current Asset Breakdown

intel_BS["Minority Interest"] = intel_BS["Total Equity"] - intel_BS["Equity Before Minority Interest"] 

asset_data = []

columns = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

'''





for col in columns.strip().split("\n"):

    asset_bar = go.Bar(

        x=intel_BS.index,

        y=intel_BS[ col ],

        name=col

    )    

    asset_data.append(asset_bar)

    

layout_assets = go.Layout(

    barmode='stack'

)



fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)

fig_bs_assets.show()

py.plot(fig_bs_assets, filename= 'Intel Total Current Assets Breakdown')
#Total Current Liability Breakdown

liability_data = []

columns = '''

Payables & Accruals

Short Term Debt

Other Short Term Liabilities

'''





for col in columns.strip().split("\n"):

    liability_bar = go.Bar(

        x=intel_BS.index,

        y=intel_BS[ col ],

        name=col

    )    

    liability_data.append(liability_bar)

    

layout_liabilitys = go.Layout(

    barmode='stack'

)



fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)

fig_bs_liabilitys.show()

py.plot(fig_bs_liabilitys, filename='Total Current liabilities Breakdown')
#Net Asset Value

intel_BS["Net Asset Value"] = intel_BS["Total Current Assets"] - intel_BS["Total Current Liabilities"] - intel_BS["Total Noncurrent Liabilities"] 

intel_BS["Net Asset Value"].plot(title = "Net Asset Value")
#Working Capital

intel_BS["Working Capital"] = intel_BS["Total Current Assets"] - intel_BS["Total Current Liabilities"]

intel_BS["Working Capital"].plot(title = "Working Capital")
#Owner's Equity or Book Value

intel_BS["Book Value"] = intel_BS["Inventories"] + intel_BS["Cash, Cash Equivalents & Short Term Investments"]+ intel_BS["Property, Plant & Equipment, Net"] - intel_BS["Total Liabilities"] 

intel_BS[["Book Value","Total Equity"]].plot(title = "Book Value")
#Current Ratio

intel_BS["Current Ratio"] = intel_BS["Total Assets"] / intel_BS["Total Current Liabilities"]

intel_BS["Current Ratio"].plot(title = "Current Ratio")
#Price to Equity Growth (PEG forward)

PE_Ratio = intel_FR.loc[71965, "Price to Earnings Ratio (TTM)"] #price to earning ratio

Growth_Rate = 1.46 #From NASDAQ Website: https://www.nasdaq.com/market-activity/stocks/intc/price-earnings-peg-ratios

PEG_Ratio = PE_Ratio / (Growth_Rate * 100)

print("Intel Corp's PEG Ratio is", PEG_Ratio)
#Earnings Per Shares(EPS) Annual Compounded Growth Rate

intel_WAS = intel_SI.loc[54.43, "Average Basic Shares Outstanding (in million USD)"] / 1000000 #Weighted Average Shares

intel_EPS = intel_PL["Net Income Available to Common Shareholders"] / intel_WAS

intel_EPS.plot(title = "Intel's earning per shares (EPS) for 10 years")
intel_EPS.mean()
#from 2010 to 2019, EPS Annual Compunded Growth Rate

start_value = float(intel_EPS.iloc[0])

end_value = float(intel_EPS.iloc[-1])

num_periods = len(intel_EPS) -1

def cagr(start_value, end_value, num_periods):

    return (end_value / start_value) ** (1 / (num_periods - 1)) - 1

INTC_cagr = cagr(start_value, end_value, num_periods)

format(float(INTC_cagr.real))

print("INTC's EPS Annual Compounded Growth Rate:","{:.2%}".format(INTC_cagr.real))
#Estimate EPS 10 years from now

def future (present_value, growth_rate, periods):

    return present_value * ((1 + growth_rate) ** periods)

present_value = float(intel_EPS.iloc[-1])

growth_rate = INTC_cagr

periods = 10

INTC_future = future(present_value, growth_rate, periods)

print("INTC's estimated 10years EPS:","{:.4}".format(INTC_future.real))
#Determine Current Target Buy Price

def Est_SP (INTC_future, PE_Ratio):

    return INTC_future * PE_Ratio

intel_PE = 54.43 / intel_EPS.mean() #Share Price from simfin data (Share price (USD; as per 2020-03-13)

INTC_future_SP = Est_SP (INTC_future, intel_PE)

INTC_Est_PV = -npf.pv(0.1, 10, 0, INTC_future_SP, when='end' ) #intel discount rate from https://finbox.com/NASDAQGS:INTC/models/dcf-growth-exit-5yr

print("INTC's Current Target Buy Price:","{:.4}".format(INTC_Est_PV))
#Margin of Safety

intel_Gross_Profit_Margin= (intel_PL["Revenue"] - intel_PL["Cost of revenue"]) / intel_PL["Revenue"]

intel_Fixed_Expense = -intel_PL["Operating Expenses"] 

intel_breakeven = intel_Fixed_Expense / intel_Gross_Profit_Margin

intel_Marginsafety = (intel_PL["Revenue"] - intel_breakeven) / intel_PL["Revenue"]

print(intel_Marginsafety.mean())
margin =1 - intel_Marginsafety.mean()

intel_margin = -INTC_Est_PV * (1 - margin)

print("INTC's margin of safety:",margin)
#Add margin of safety

Target_price_intel = INTC_Est_PV * intel_Marginsafety.mean()

print(Target_price_intel)
#Debt to Equity Ratio

intel_DER = intel_BS["Total Liabilities"] / intel_BS["Total Equity"]

intel_DER.plot(title = "Intel's Debt to Equity Ratio")
#Interest Coverage Ratio

intel_EBIT = intel_PL["Revenue"] - intel_PL["Cost of revenue"] - intel_PL["Operating Expenses"]

intel_interestExp = intel_EBIT - intel_PL["Pretax Income (Loss), Adjusted"]

intel_ICR = intel_EBIT / intel_interestExp

intel_ICR.plot(title = "intel's interest coverage ratio")
#  neural network model diagnostics

INTC = pd.read_csv("/kaggle/input/intc-stockprice/INTC.csv", index_col = 'Date',parse_dates = ['Date'])

INTC = INTC.dropna() 

#INTC = INTC['Close']

INTC = INTC[['Open', 'High', 'Low', 'Close']]

INTC.plot(title = "INTC's stock price")
INTC['H-L'] = INTC['High'] - INTC['Low']

INTC['O-C'] = INTC['Close'] - INTC['Open']

INTC['3day MA'] = INTC['Close'].shift(1).rolling(window = 3).mean()

INTC['10day MA'] = INTC['Close'].shift(1).rolling(window = 10).mean() 

INTC['30day MA'] = INTC['Close'].shift(1).rolling(window = 30).mean() 

INTC['Std_dev']= INTC['Close'].rolling(5).std() 

#INTC['RSI'] = talib.RSI(dataset['Close'].values, timeperiod = 9)

#INTC['Williams %R'] = talib.WILLR(INTC['High'].values, INTC['Low'].values, INTC['Close'].values, 7)
INTC['3day MA'].plot(title = "INTC 3day MA")
INTC['10day MA'].plot(title = "INTC 10 day MA")
INTC['30day MA'].plot(title = "INTC 30 day MA")
INTC['Price_Rise'] = np.where(INTC['Close'].shift(-1) > INTC['Close'], 1, 0)
X = INTC.iloc[:, 4:-1] 

y = INTC.iloc[:, -1]
split = int(len(INTC)*0.8) 

X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
from sklearn.preprocessing import StandardScaler 

sc = StandardScaler() 

X_train = sc.fit_transform(X_train) 

X_test = sc.transform(X_test)
classifier = Sequential()

classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))

classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
y_pred = classifier.predict(X_test) 

y_pred = (y_pred > 0.5)

INTC['y_pred'] = np.NaN 

INTC.iloc[(len(INTC) - len(y_pred)):,-1:] = y_pred 

trade_INTC = INTC.dropna()
trade_INTC['Strategy Returns'] = 0. 

trade_INTC['Strategy Returns'] = np.log(trade_INTC['Close']/trade_INTC['Close'].shift(1)) 

trade_INTC['Strategy Returns'] = trade_INTC['Strategy Returns'].shift(-1)
trade_INTC['Tomorrows Returns'] = 0. 

trade_INTC['Tomorrows Returns'] = np.where(trade_INTC['y_pred'] == True, trade_INTC['Tomorrows Returns'], - trade_INTC['Tomorrows Returns'])
trade_INTC['Cumulative Market Returns'] = np.cumsum(trade_INTC['Tomorrows Returns']) 

trade_INTC['Cumulative Strategy Returns'] = np.cumsum(trade_INTC['Strategy Returns'])
plt.figure(figsize=(10,5)) 

plt.plot(trade_INTC['Cumulative Market Returns'], color='r', label='Market Returns') 

plt.plot(trade_INTC['Cumulative Strategy Returns'], color='g', label='Strategy Returns') 

plt.legend() 

plt.title('Market Return vs. Strategy Return')

plt.show()
#ARIMA prediction and forcaset

INTC.index = pd.to_datetime(INTC.index)

sub = INTC[ '2009/1/2': '2018/12/31']['Close']

train = sub.loc[ '2009/1/2': '2014/12/31']

test = sub.loc[ '2015/1/1': '2018/12/31']

plt.figure(figsize=( 10, 10))

print(train)

plt.plot(train)

plt.title('Train data set')

plt.show()
INTC[ 'Close_diff_1'] = INTC[ 'Close'].diff( 1)

INTC[ 'Close_diff_2'] = INTC[ 'Close_diff_1'].diff( 1)

fig = plt.figure(figsize=( 20, 6))

ax1 = fig.add_subplot( 131)

ax1.plot(INTC[ 'Close'])

ax2 = fig.add_subplot( 132)

ax2.plot(INTC[ 'Close_diff_1'])

ax3 = fig.add_subplot( 133)

ax3.plot(INTC[ 'Close_diff_2'])

plt.show()
fig = plt.figure(figsize=( 12, 8))

ax1 = fig.add_subplot( 211)

fig = sm.graphics.tsa.plot_acf(train, lags= 20,ax=ax1)

ax1.xaxis.set_ticks_position( 'bottom')

fig.tight_layout()

ax2 = fig.add_subplot( 212)

fig = sm.graphics.tsa.plot_pacf(train, lags= 20, ax=ax2)

ax2.xaxis.set_ticks_position( 'bottom')

fig.tight_layout()

plt.show()
model = sm.tsa.ARIMA(train, order=( 1, 0, 0))

results = model.fit()

resid = results.resid 

fig = plt.figure(figsize=( 12, 8))

fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags= 40)

plt.show()
#2015/1/1': '2018/12/31'

model = sm.tsa.ARIMA(sub, order=( 1, 0, 0))

results = model.fit()

predict_sunspots = results.predict(start=str('2015-01'),end=str('2018-12'),dynamic=False)

print(predict_sunspots)

fig, ax = plt.subplots(figsize=(12, 8))

ax = sub.plot(ax=ax)

predict_sunspots.plot(ax=ax)

plt.title('Prediction')

plt.show()
results.forecast() [0]

print("The forcecase stock price", results.forecast() [0])
#Use simfin API to download data

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
NVDA_bs.columns
#NVDA's Current Asset Breakdown

asset_data = []

columns = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

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



fig_bs_nvda_assets = go.Figure(data=asset_data, layout=layout_assets)

fig_bs_nvda_assets.show()

py.plot(fig_bs_assets, filename= 'NVDA Total Current Assets Breakdown')
NVDA_pl.columns
intel_PL.columns
NVDA_bs["Total Current Liabilities"].plot(title = "NVDA Total Current Liabilities")
plt.figure(figsize=(30,12),dpi=80)

plt.plot(intel_PL["Revenue"],label="Revenue",color='r')

plt.plot(intel_PL["Cost of revenue"],label="Cost of Sales",color='g')

plt.title("Revenue vs. Cost of Sales for Intel")

plt.legend()