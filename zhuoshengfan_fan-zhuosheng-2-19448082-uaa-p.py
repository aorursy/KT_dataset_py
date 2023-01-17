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
#!pip install plotly==4.4.1

#!pip install chart_studio

#!pip install xlrd
import os

#from tk_library_py import excel_to_df
for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
_, UA_PL, UA_BS, UA_CF = excel_to_df("/kaggle/input/SimFin-data.xlsx")
UA_BS
del(UA_BS["Assets"])
UA_BS
UA_BS["_Total Current Assets"] = UA_BS["Cash, Cash Equivalents & Short Term Investments"] + UA_BS["Accounts & Notes Receivable"] + UA_BS["Inventories"] + UA_BS["Other Short Term Assets"]
UA_BS[["_Total Current Assets", "Total Current Assets"]]
UA_BS["_NonCurrent Assets"] = UA_BS["Property, Plant & Equipment, Net"] + UA_BS["Other Long Term Assets"]
UA_BS["_Total Assets"] = UA_BS["_NonCurrent Assets"] + UA_BS["_Total Current Assets"]
UA_BS["_Total Liabilities"] = UA_BS["Total Current Liabilities"] + UA_BS["Total Noncurrent Liabilities"]
UA_BS[["_Total Liabilities", "Total Liabilities"]]
#UA_TE = total equity

UA_TE=UA_BS["Total Assets"]-UA_BS["Total Liabilities"]
UA_TE
bookvalue_pershare = UA_TE.values/np.asarray([397,407,412,418,439,442,445,441,446,454])

bookvalue_pershare
pb_ratio=np.asarray([4.7322,8.7803,12.3991,16.2984,29.9224,42.6721,38.1589,19.0986,19.2185,21.4640])/np.asarray([1.25180353, 1.56371499, 1.98282039, 2.51998565, 3.07585421,

       3.77425792, 4.56382022, 4.5774195 , 4.52213229, 4.73587445])

pb_ratio
print("UA historical PB ratio is",pb_ratio)
%matplotlib inline

UA_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

'''



asset_columns = [ x for x in good_stuff.strip().split("\n") ]
asset_columns
UA_BS[ asset_columns ].plot()
!pip install chart_studio
import chart_studio

#every time runs go to plotly generate API key.SETTING->API KEY

chart_studio.tools.set_credentials_file(username='zhuosheng03', api_key="BZu9NaS0WqNtZ0qsqB9W")
import chart_studio.plotly as py

import plotly.graph_objs as go

#from tk_library_py import combine_regexes
assets = go.Bar(

    x=UA_BS.index,

    y=UA_BS["Total Assets"],

    name='Assets'

)

liabilities = go.Bar(

    x=UA_BS.index,

    y=UA_BS["Total Liabilities"],

    name='Liabilities'

)



shareholder_equity = go.Scatter(

    x=UA_BS.index,

    y=UA_BS["Total Equity"],

    name='Equity'

)



data = [assets, liabilities, shareholder_equity]

layout = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

fig_bs.update_layout(yaxis_title="In Million USD", xaxis_title="Year", title="Balance sheet")

fig_bs.show()

# py.iplot(fig_bs, filename='Total Liabilities & Equity')
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

        x=UA_BS.index,

        y=UA_BS[ col ],

        name=col

    )    

    asset_data.append(asset_bar)

    

layout_assets = go.Layout(

    barmode='stack'

)



fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)

fig_bs_assets.update_layout(yaxis_title="In Million USD", xaxis_title="Year", title="Assets")

fig_bs_assets.show()

#py.iplot(fig_bs_assets, filename='Total Assets Breakdown')
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

        x=UA_BS.index,

        y=UA_BS[ col ],

        name=col

    )    

    liability_data.append(liability_bar)

    

layout_liabilitys = go.Layout(

    barmode='stack'

)



fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)

fig_bs_liabilitys.update_layout(yaxis_title="In Million USD", xaxis_title="Year", title="Total Liabilities")

fig_bs_liabilitys.show()

#py.iplot(fig_bs_liabilitys, filename='Total liabilities Breakdown')
UA_BS["working capital"] = UA_BS["Total Current Assets"] - UA_BS["Total Current Liabilities"]
UA_BS[["working capital"]].plot()
UA_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
PR_data = []

columns = '''

Accounts & Notes Receivable

Payables & Accruals

'''



for col in columns.strip().split("\n"):

    PR_Scatter = go.Scatter(

        x=UA_BS.index,

        y=UA_BS[ col ],

        name=col

    )    

    PR_data.append(PR_Scatter)

    

layout_PR = go.Layout(

    barmode='stack'

)



fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)

fig_bs_PR.update_layout(yaxis_title="In Million USD", xaxis_title="Year", title="PR")

fig_bs_PR.show()

#py.iplot(fig_bs_PR, filename='Accounts & Notes Receivable vs Payables & Accruals')
UA_BS["Inventories"].plot()
UA_BS[ ["Property, Plant & Equipment, Net", "Other Long Term Assets"] ].plot()
AAA_data = []

columns = '''

Property, Plant & Equipment, Net

Other Long Term Assets

'''





for col in columns.strip().split("\n"):

    AAA_bar = go.Bar(

        x=UA_BS.index,

        y=UA_BS[ col ],

        name=col

    )    

    AAA_data.append(AAA_bar)

    

layout_AAA = go.Layout(

    barmode='stack'

)



fig_bs_AAA = go.Figure(data=AAA_data, layout=layout_AAA)

fig_bs_AAA.update_layout(yaxis_title="In Million USD", xaxis_title="Year", title="Fix Asset")

fig_bs_AAA.show()

#py.iplot(fig_bs_AAA, filename='Total Long Term Assets')
equity_columns = '''

Share Capital & Additional Paid-In Capital

Retained Earnings

Other Equity

Equity Before Minority Interest

'''



equity_columns = [ x for x in equity_columns.strip().split("\n")]
equity_columns
UA_BS[ equity_columns ].plot()
equity_data = []

columns = '''

Share Capital & Additional Paid-In Capital

Retained Earnings

Other Equity

Equity Before Minority Interest

'''





for col in columns.strip().split("\n"):

    equity_Scatter = go.Scatter(

        x=UA_BS.index,

        y=UA_BS[ col ],

        name=col

    )    

    equity_data.append(equity_Scatter)

    

layout_equity = go.Layout(

    barmode='stack'

)



fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)

fig_bs_equity.update_layout(yaxis_title="In Million USD", xaxis_title="Year", title="Total Equity")

fig_bs_equity.show()

#py.iplot(fig_bs_equity, filename='Total Equity')
UA_BS["book value"] = UA_BS["Total Assets"] - UA_BS["Total Liabilities"]
UA_BS["book value"].plot()
UA_BS["current ratio"] = UA_BS["Total Current Assets"] / UA_BS["Total Current Liabilities"]
UA_BS["current ratio"].plot()
#additional calculation 1#

#EPS=Net Income/share outstanding#

#UA share outstanding from year 2010 to year 2019 is [397,407,412,418,439,442,445,441,446,454], from website:https://www.macrotrends.net/stocks/charts/UAA/under-armour/shares-outstanding# 

eps = UA_PL["Net Income Available to Common Shareholders"]/[397,407,412,418,439,442,445,441,446,454]

cagr = (eps.loc["FY '19"] / eps["FY '10"]) **(1/10)-1

print("The EPS Annual Compounded Growth Rate of UA is",(cagr))
#additional calculation 2#

#Estimate EPS 10 years from now#

#fv is estimate EPS 10 years from now#

eps = UA_PL["Net Income Available to Common Shareholders"]/[397,407,412,418,439,442,445,441,446,454]

cagr = (eps.loc["FY '19"] / eps["FY '10"]) **(1/10)-1

fv = eps.loc["FY '19"]*((1+cagr)**10)

print("Estimate EPS 10 years from now is",fv) 
#additional calculation 3#

#PE ratio = share price / EPS #

#Share price of UA from year 2010 to year 2019 is [4.7322,8.7803,12.3991,16.2984,29.9224,42.6721,38.1589,19.0986,19.2185,21.4640] from website :https://www.macrotrends.net/stocks/charts/UAA/under-armour/stock-price-history#

#After calculate the estimate future price and then discount back to present at a proper discount rate#

#discount rate can calculated by using CAPM model#

import numpy as np

eps = UA_PL["Net Income Available to Common Shareholders"]/[397,407,412,418,439,442,445,441,446,454]

peratio = [4.7322,8.7803,12.3991,16.2984,29.9224,42.6721,38.1589,19.0986,19.2185,21.4640]/eps

np. sum(peratio)

avgperatio = np. sum(peratio)/10

avgperatio

stockpricefuture = avgperatio * fv

stockpricefuture

print("The future price of UA is",stockpricefuture)
#discount rate, using WACC model#

#discount rate is 7.72% from website:https://www.gurufocus.com/term/wacc/NYSE:UA/WACC-/Under-Armour#

desiredrate=0.072

print("The discount rate is", desiredrate)
todayprice = stockpricefuture / (1 + desiredrate) ** 10

todayprice

print("Current Target price of UA is",todayprice)
#additional calculation 4#

#margin of safty 15% of target price#

marginofsafety = todayprice * 0.15

marginofsafety

print("Margin of safety is",marginofsafety)
#additional calculation 5#

#debt to equity ratio = total debt / total equity#

dte = UA_BS["Total Liabilities"] / UA_BS["Total Equity"]

dte

print(dte)
#additional calculation 6#

#interest coverage ratio = EBIT / interest#

#interest expense of UA is 4.238m in year 2019, from website: https://finbox.com/NYSE:UA/explorer/interest_exp#

#EBIT of UA is 237m in year 2019# from website:https://www.macrotrends.net/stocks/charts/UAA/under-armour/ebit#

EBIT = 237

interest = 4.238

icr = EBIT / interest

icr

print("The interest coverage ratio of UA is",icr)

# !pip install pandas_datareader

# !pip install statsmodels

!pip install TA-Lib
#try to do a time series model, some code copy from Internet#

import pandas_datareader as web

from statsmodels.tsa.seasonal import seasonal_decompose

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
stock_price = web.get_data_yahoo("UA", "2017-03-01")

stock_price
def decomposing(timeseries):

    decomposition = seasonal_decompose(timeseries)

    trend = decomposition.trend

    seasonal = decomposition.seasonal

    residual = decomposition.resid

    

    ts_std = np.std(timeseries)

    ts_mean = np.mean(timeseries)

    ts = (timeseries-ts_mean) / ts_std

    

    trend_std = np.std(trend)

    trend_mean = np.mean(trend)

    tr = (trend-trend_mean) / trend_std



    sea_std = np.std(seasonal)

    sea_mean = np.mean(seasonal)

    

    sea = (seasonal-sea_mean) / sea_std

    

    sma = ts.rolling(14).mean()

    

    plt.figure(figsize=(21,21))

    plt.subplot(311)

    plt.plot(ts, label='Original')

    plt.plot(tr, label='Trend')

    plt.legend(loc='best')

    plt.subplot(312)

    plt.plot(ts, label='Original')

    plt.plot(sea, label='Seasonarity')

    plt.plot(sma, label="SMA")

    plt.legend(loc='best')

    plt.subplot(313)

    plt.plot(residual, label='Residual')

    plt.legend(loc='best')

    plt.show()

    # problem: when to dropna and when to fillna

    trend = trend.fillna(0)

    seasonal = seasonal.fillna(0)

    residual = residual.fillna(0)

    # trend.dropna(inplace=True)

    # seasonal.dropna(inplace=True)

    # residual.dropna(inplace=True)

    # return timeseries, trend, seasonal, residual
decomposing(stock_price.resample("w").mean()["Close"])