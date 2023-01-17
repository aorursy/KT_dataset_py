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

#import tk_library_py

#from tk_library_py import excel_to_df
for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
_, nike_PL, nike_BS, nike_CF = excel_to_df("/kaggle/input/simfin/SimFin-data.xlsx")
nike_BS
del(nike_BS["Assets"])

nike_BS
nike_BS["_Total Current Assets"] = nike_BS["Cash, Cash Equivalents & Short Term Investments"] + nike_BS["Accounts & Notes Receivable"] + nike_BS["Inventories"] + nike_BS["Other Short Term Assets"]
nike_BS[["_Total Current Assets", "Total Current Assets"]]
nike_BS["_NonCurrent Assets"] = nike_BS["Property, Plant & Equipment, Net"] + nike_BS["Other Long Term Assets"]
nike_BS["_Total Assets"] = nike_BS["_NonCurrent Assets"] + nike_BS["_Total Current Assets"] 
nike_BS["_Total Liabilities"] = nike_BS["Total Current Liabilities"] + nike_BS["Total Noncurrent Liabilities"]
nike_BS[["_Total Liabilities", "Total Liabilities"]]
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
nike_BS[ asset_columns ].plot()
!pip install chart_studio
import chart_studio

#every time runs go to plotly generate API key.SETTING->API KEY

chart_studio.tools.set_credentials_file(username='zhuosheng03', api_key="BZu9NaS0WqNtZ0qsqB9W")
import chart_studio.plotly as py

import plotly.graph_objs as go

#from tk_library_py import combine_regexes

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

py.iplot(fig_bs, filename='Total Liabilities & Equity')
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

py.iplot(fig_bs_assets, filename='Total Assets Breakdown')
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



fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)

py.iplot(fig_bs_liabilitys, filename='Total liabilities Breakdown')
nike_BS["working capital"] = nike_BS["Total Current Assets"] - nike_BS["Total Current Liabilities"]
nike_BS[["working capital"]].plot()
nike_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
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

py.iplot(fig_bs_PR, filename='Accounts & Notes Receivable vs Payables & Accruals')
nike_BS["Inventories"].plot()
nike_BS[ ["Property, Plant & Equipment, Net", "Other Long Term Assets"] ].plot()
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

py.iplot(fig_bs_AAA, filename='Total Long Term Assets')
equity_columns = '''

Share Capital & Additional Paid-In Capital

Retained Earnings

Other Equity

Equity Before Minority Interest

'''



equity_columns = [ x for x in equity_columns.strip().split("\n")]

equity_columns
nike_BS[ equity_columns ].plot()
equity_data = []

columns = '''

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



fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)

py.iplot(fig_bs_equity, filename='Total Equity')
nike_BS["book value"] = nike_BS["Total Assets"] - nike_BS["Total Liabilities"]

nike_BS["book value"].plot()
nike_BS["current ratio"] = nike_BS["Total Current Assets"] / nike_BS["Total Current Liabilities"]
nike_BS["current ratio"].plot()
#additional calculation 1#

#EPS=Net Income/share outstanding#

#Nike share outstanding from year 2010 to year 2019 is [1976,1943,1879,1833,1812,1769,1743,1692,1659,1618], from website:https://www.macrotrends.net/stocks/charts/NKE/nike/shares-outstanding# 

eps = nike_PL["Net Income Available to Common Shareholders"]/[1976,1943,1879,1833,1812,1769,1743,1692,1659,1618]

cagr = (eps.loc["FY '19"] / eps["FY '10"]) **(1/10)-1

print("The EPS Annual Compounded Growth Rate of Nike is",(cagr))
#additional calculation 2#

#Estimate EPS 10 years from now#

#fv is estimate EPS 10 years from now#

eps = nike_PL["Net Income Available to Common Shareholders"]/[1976,1943,1879,1833,1812,1769,1743,1692,1659,1618]

cagr = (eps.loc["FY '19"] / eps["FY '10"]) **(1/10)-1

fv = eps.loc["FY '19"]*((1+cagr)**10)

print("Estimate EPS 10 years from now is",fv) 
#additional calculation 3#

#PE ratio = share price / EPS #

#Share price of Nike from year 2010 to year 2019 is [18.6400,21.6823,25.2291,32.3958,40.4344,55.1266,56.2473,55.7060,72.9388,86.7334] from website :https://www.macrotrends.net/stocks/charts/NKE/nike/stock-price-history#

#After calculate the estimate future price and then discount back to present at a proper discount rate#

#discount rate can calculated by using CAPM model#

import numpy as np

eps = nike_PL["Net Income Available to Common Shareholders"]/[1976,1943,1879,1833,1812,1769,1743,1692,1659,1618]

peratio = [18.6400,21.6823,25.2291,32.3958,40.4344,55.1266,56.2473,55.7060,72.9388,86.7334]/eps

np. sum(peratio)

avgperatio = np. sum(peratio)/10

avgperatio

stockpricefuture = avgperatio * fv

stockpricefuture

print("The future price of Nike is",stockpricefuture)
#discount rate, using CAPM model#

#US 10-year bond yield as risk-free rate from website: https://www.marketwatch.com/investing/bond/tmubmusd10y?countrycode=bx#

#market return using average of ten year Dow Jones index, from website:https://www.macrotrends.net/1358/dow-jones-industrial-average-last-10-years#

#beta is from yahoo finance website, https://finance.yahoo.com/quote/NKE?p=NKE&.tsrc=fin-srch#

#rf is risk-free rate, mktrtn is market return in 10 years, avgmktrtn is average market return#

rf = 0.01623

mktrtn = [0.1102,0.0553,0.0726,0.265,0.0752,-0.0223,0.1342,0.2508,-0.0563,0.2234,0.0198]

np.sum(mktrtn)

avgmktrtn = np.sum(mktrtn)/10

beta = 0.84

desiredrate = rf + beta*(avgmktrtn-rf)

desiredrate

print("The discount rate is", desiredrate)
todayprice = stockpricefuture / (1 + desiredrate) ** 10

todayprice

print("Current Target price of Nike is",todayprice)
#additional calculation 4#

#margin of safty 10% of target price#

marginofsafety = todayprice * 0.10

marginofsafety

print("Margin of safety is",marginofsafety)
#additional calculation 5#

#debt to equity ratio = total debt / total equity#

dte = nike_BS["Total Liabilities"] / nike_BS["Total Equity"]

dte

print(dte)
#additional calculation 6#

#interest coverage ratio = EBIT / interest#

#interest expense of nike is 131m in year 2019, from website: https://finbox.com/NYSE:NKE/explorer/interest_exp#

#EBIT of nike is 5238m in year 2019# from website:https://finbox.com/NYSE:NKE/explorer/ebit#

EBIT = 5238

interest = 131

icr = EBIT / interest

icr

print("The interest coverage ratio of Nike is",icr)