#Do the below in kaggle

# !pip install plotly==4.4.1

!pip install chart_studio

# !pip install xlrd # for reading_excel files with extensions .xlsx into a pandas dataframe

import os

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
import pandas as pd

import numpy as np
# Show the files and their pathnames

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Add your simfin-data.xlsx using the '+ Add Data' top right button

_, xray_PL, xray_BS, xray_CF = excel_to_df("/kaggle/input/SimFin-data-xray.xlsx")

# _, xray_PL, xray_BS, xray_CF = excel_to_df(r"C:\Users\Admin\SimFin-data.xlsx")
xray_BS
xray_BS["_Total Current Assets"] = xray_BS["Cash, Cash Equivalents & Short Term Investments"] + xray_BS["Accounts & Notes Receivable"] + xray_BS["Inventories"] + xray_BS["Other Short Term Assets"]
xray_BS[["_Total Current Assets", "Total Current Assets"]]
xray_BS["_Total NonCurrent Assets"] = xray_BS["Property, Plant & Equipment, Net"] + xray_BS["Other Long Term Assets"]
xray_BS["_Total Assets"] = xray_BS["_Total NonCurrent Assets"] + xray_BS["_Total Current Assets"] 
xray_BS["_Total Liabilities"] = xray_BS["Total Current Liabilities"] + xray_BS["Total Noncurrent Liabilities"]
xray_BS[["_Total Liabilities", "Total Liabilities"]]
%matplotlib inline

xray_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

'''



asset_columns = [ x for x in good_stuff.strip().split("\n") ]

asset_columns
xray_BS[ asset_columns ].plot()
!pip install chart_studio

import chart_studio

chart_studio.tools.set_credentials_file(username='ZHUQINQIN',api_key='ILHp23uNgGXJB22ZPIXG')
import chart_studio.plotly as py

import plotly.graph_objs as go

assets = go.Bar(

    x=xray_BS.index,

    y=xray_BS["Total Assets"],

    name='Assets'

)

liabilities = go.Bar(

    x=xray_BS.index,

    y=xray_BS["Total Liabilities"],

    name='Liabilities'

)



shareholder_equity = go.Scatter(

    x=xray_BS.index,

    y=xray_BS["Total Equity"],

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

        x=xray_BS.index,

        y=xray_BS[ col ],

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

Long Term Debt

Other Long Term Liabilities

Total Noncurrent Liabilities

'''





for col in columns.strip().split("\n"):

    liability_bar = go.Bar(

        x=xray_BS.index,

        y=xray_BS[ col ],

        name=col

    )    

    liability_data.append(liability_bar)

    

layout_liabilitys = go.Layout(

    barmode='stack'

)



fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)

#py.iplot(fig_bs_liabilitys, filename='Total liabilities Breakdown')

fig_bs_liabilitys.show()
xray_BS["working capital"] = xray_BS["Total Current Assets"] - xray_BS["Total Current Liabilities"]
xray_BS[["working capital"]].plot()
xray_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
# Using Chart Studio in Plotly 



PR_data = []

columns = '''

Accounts & Notes Receivable

Payables & Accruals

'''



for col in columns.strip().split("\n"):

    PR_Scatter = go.Scatter(

        x=xray_BS.index,

        y=xray_BS[ col ],

        name=col

    )    

    PR_data.append(PR_Scatter)

    

layout_PR = go.Layout(

    barmode='stack'

)



fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)

#py.iplot(fig_bs_PR, filename='Accounts & Notes Receivable vs Payables & Accruals')

fig_bs_PR.show()
xray_BS["Inventories"].plot()
xray_BS[ ["Property, Plant & Equipment, Net", "Other Long Term Assets"] ].plot()
# Using Plotly



AAA_data = []

columns = '''

Property, Plant & Equipment, Net

Other Long Term Assets

'''





for col in columns.strip().split("\n"):

    AAA_bar = go.Bar(

        x=xray_BS.index,

        y=xray_BS[ col ],

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

Preferred Equity

Share Capital & Additional Paid-In Capital

Treasury Stock

Retained Earnings

Other Equity

Equity Before Minority Interest

Minority Interest

'''



equity_columns = [ x for x in equity_columns.strip().split("\n")]

equity_columns
xray_BS[ equity_columns ].plot()
# Using Plotly



equity_data = []

columns = '''

Preferred Equity

Share Capital & Additional Paid-In Capital

Treasury Stock

Retained Earnings

Other Equity

Equity Before Minority Interest

Minority Interest

'''





for col in columns.strip().split("\n"):

    equity_Scatter = go.Scatter(

        x=xray_BS.index,

        y=xray_BS[ col ],

        name=col

    )    

    equity_data.append(equity_Scatter)

    

layout_equity = go.Layout(

    barmode='stack'

)



fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)

#py.iplot(fig_bs_equity, filename='Total Equity')

fig_bs_equity.show()
# xray has no preferred stock, no intangible assets, and no goodwill



xray_BS["book value"] = xray_BS["Total Assets"] - xray_BS["Total Liabilities"]

xray_BS["book value"].plot()
xray_BS["current ratio"] = xray_BS["Total Current Assets"] / xray_BS["Total Current Liabilities"]
xray_BS["current ratio"].plot()
#Calculate 1.	Price-to-Earnings Growth Ratio (PEG forward)  

#using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate

#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp



PE_RATIO = 16.38 # FROM SIMFIN WEBSITE: https://simfin.com/data/companies/121214



# FROM NASDAQ WEBSITE: https://www.nasdaq.com/symbol/xom/earnings-growth

GROWTH_RATE = 0.1394 # Forcast over the five next years



PEG_ratio = PE_RATIO / (GROWTH_RATE*100)



print("Dentsply Sirona's PEG Ratio is", PEG_ratio)
!pip install simfin

import pandas as pd



# Import the main functionality from the SimFin Python API.

import simfin as sf



# Import names used for easy access to SimFin's data-columns.

from simfin.names import *

import math

import datetime

import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

from sklearn.linear_model import LinearRegression

from scipy import sparse
sf.set_data_dir('~/simfin_data/')

sf.set_api_key(api_key='free')
df1 = sf.load(dataset='income', variant='annual', market='us')
df_PL = sf.load(dataset='income', variant='annual', market='us',

              index=[TICKER, REPORT_DATE],

              parse_dates=[REPORT_DATE, PUBLISH_DATE])

df_BS = sf.load(dataset='balance', variant='annual', market='us',

              index=[TICKER, REPORT_DATE],

              parse_dates=[REPORT_DATE, PUBLISH_DATE])

df_CF = sf.load(dataset='cashflow', variant='annual', market='us',

              index=[TICKER, REPORT_DATE],

              parse_dates=[REPORT_DATE, PUBLISH_DATE])
XRAY_PL = df_PL.loc['XRAY']

XRAY_BS = df_BS.loc['XRAY']

XRAY_CF = df_CF.loc['XRAY']

#.columns

XRAY_PL.to_csv('XRAY_PL')

XRAY_BS.to_csv('XRAY_BS')

XRAY_CF.to_csv('XRAY_CF')
#calculate EPS

NetIncome = XRAY_PL['Net Income']

Shares  = XRAY_PL['Shares (Basic)']

EPS = NetIncome / Shares

EPS_rate = (EPS[-1]/EPS[1])**(1/9) -1 

EPS.plot(grid = True)

EPS
#predict EPS in 10 years

model = LinearRegression()

x = [[i] for i in range(2008,2019)]

x_pre = [[i] for i in range(2019,2028)]

y = list(EPS[0:])

y1 = [[i] for i in y]

model.fit(x,y)

y_pred = model.predict(x_pre)

plt.plot(x, y, 'k.')

plt.plot(x_pre, y_pred, 'r.')

a = model.coef_

b = model.intercept_

plt.plot(x, a*x+b, '-')

plt.grid()
#Interest_coverage_ratio = XRAY_PL['Pretax Income (Loss), Adj.']/ (-XRAY_PL['Interest Expense, Net']), since the Interest Expense is counted as negative value in balance sheet

Interest_coverage_ratio = (XRAY_PL['Operating Income (Loss)']+XRAY_PL['Non-Operating Income (Loss)'])/(-XRAY_PL['Interest Expense, Net'])

Interest_coverage_ratio.plot(grid = True)

Interest_coverage_ratio
#calculate annual revenue growth rate

Revenue = XRAY_PL['Revenue']

Annual_Revenue_Growth_Rate = Revenue.copy()[-5:]

for i in range(len(Annual_Revenue_Growth_Rate)):

    Annual_Revenue_Growth_Rate[i] = (Annual_Revenue_Growth_Rate[i]-Revenue[i+4])/Revenue[i+4]

Annual_Revenue_Growth_Rate.plot(grid = True)

Annual_Revenue_Growth_Rate
#Calculate annual gross profit growth rate

Gross_Profit = XRAY_PL['Gross Profit']

Annual_Profit_Growth_Rate = Gross_Profit.copy()[-5:]

for i in range(len(Annual_Profit_Growth_Rate)):

    Annual_Profit_Growth_Rate[i] = (Annual_Profit_Growth_Rate[i]-Gross_Profit[i+4])/Gross_Profit[i+4]

Annual_Profit_Growth_Rate.plot(grid = True)

Annual_Profit_Growth_Rate
Net_Cash_from_Operating_Activities = XRAY_CF['Net Cash from Operating Activities'].copy()[-5:]

Net_Cash_from_Operating_Activities.plot(grid =True)

Net_Cash_from_Operating_Activities
Total_Current_Assets = XRAY_BS['Total Current Assets']

Total_Current_Liabilities = XRAY_BS['Total Current Liabilities']

Current_Ratio = Total_Current_Assets/Total_Current_Liabilities

Current_Ratio.plot(grid = True)

Current_Ratio
Total_Liabilities = XRAY_BS['Total Liabilities']

Total_Equity = XRAY_BS['Total Equity']

Debt_to_Equity_Ratio = Total_Liabilities/Total_Equity

Debt_to_Equity_Ratio.plot(grid = True)

Debt_to_Equity_Ratio
df_prices = sf.load_shareprices(variant='daily', market='us')
#predict margin on safety price

XRAY_stock = df_prices.loc['XRAY'].copy()

Margin_of_Safety = XRAY_stock['Close']*(1-0.25)

Margin_of_Safety.plot(grid = True)