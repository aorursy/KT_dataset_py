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
# !pip install plotly==4.4.1
# !pip install chart_studio
# !pip install xlrd # for reading_excel files with extensions .xlsx into a pandas dataframe
# Add 'tk_library.py' file given by your tutor, as a utility script under 'File'
# Look for it under usr/bin on the right drawer

# import excel_to_df function
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


import os
# import tk_library_py
# from tk_library_py import excel_to_df
# Show the files and their pathnames
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Add your simfin-data.xlsx using the '+ Add Data' top right button
_, NIKE_PL, NIKE_BS, NIKE_CF = excel_to_df("/kaggle/input/simfindata-nike/SimFin-data NIKE2.xlsx")

NIKE_BS
NIKE_BS["_Total Current Assets"] = NIKE_BS["Cash, Cash Equivalents & Short Term Investments"] + NIKE_BS["Accounts & Notes Receivable"] + NIKE_BS["Inventories"] +NIKE_BS["Other Short Term Assets"]
NIKE_BS[["_Total Current Assets", "Total Current Assets"]]
NIKE_BS["_NonCurrent Assets"] = NIKE_BS["Property, Plant & Equipment, Net"]
NIKE_BS["_Total Assets"] = NIKE_BS["_Total Current Assets"] 
NIKE_BS["_Total Liabilities"] = NIKE_BS["Total Current Liabilities"] + NIKE_BS["Total Noncurrent Liabilities"]
NIKE_BS[["_Total Liabilities", "Total Liabilities"]]
%matplotlib inline
NIKE_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''
Cash, Cash Equivalents & Short Term Investments
Accounts & Notes Receivable
Inventories
Other Short Term Assets
'''

asset_columns = [ x for x in good_stuff.strip().split("\n") ]

asset_columns
NIKE_BS[ asset_columns ].plot()
import chart_studio
# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 
# https://chart-studio.plot.ly/feed/#/

# Un-remark the code below and add your own your_username and own your_apikey
chart_studio.tools.set_credentials_file(username='999', api_key='iLc7J9lfdvWDmLid9rbU')
import chart_studio.plotly as py
import plotly.graph_objs as go
#from tk_library_py import combine_regexes
assets = go.Bar(
    x=NIKE_BS.index,
    y=NIKE_BS["Total Assets"],
    name='Assets'
)
liabilities = go.Bar(
    x=NIKE_BS.index,
    y=NIKE_BS["Total Liabilities"],
    name='Liabilities'
)

shareholder_equity = go.Scatter(
    x=NIKE_BS.index,
    y=NIKE_BS["Total Equity"],
    name='Equity'
)

data = [assets, liabilities, shareholder_equity]
layout = go.Layout(
    barmode='stack'
)

fig_bs = go.Figure(data=data, layout=layout)
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
        x=NIKE_BS.index,
        y=NIKE_BS[ col ],
        name=col
    )    
    asset_data.append(asset_bar)
    
layout_assets = go.Layout(
    barmode='stack'
)

fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)
fig_bs.show()
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
        x=NIKE_BS.index,
        y=NIKE_BS[ col ],
        name=col
    )    
    liability_data.append(liability_bar)
    
layout_liabilitys = go.Layout(
    barmode='stack'
)

fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)
fig_bs.show()
NIKE_BS["working capital"] = NIKE_BS["Total Current Assets"] - NIKE_BS["Total Current Liabilities"]
NIKE_BS[["working capital"]].plot()
NIKE_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()

PR_data = []
columns = '''
Accounts & Notes Receivable
Payables & Accruals
'''

for col in columns.strip().split("\n"):
    PR_Scatter = go.Scatter(
        x=NIKE_BS.index,
        y=NIKE_BS[ col ],
        name=col
    )    
    PR_data.append(PR_Scatter)
    
layout_PR = go.Layout(
    barmode='stack'
)

fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)
fig_bs.show()
NIKE_BS["Inventories"].plot()
NIKE_BS[ ["Property, Plant & Equipment, Net", "Other Long Term Assets"] ].plot()
AAA_data = []
columns = '''
Property, Plant & Equipment, Net
Other Long Term Assets
'''


for col in columns.strip().split("\n"):
    AAA_bar = go.Bar(
        x=NIKE_BS.index,
        y=NIKE_BS[ col ],
        name=col
    )    
    AAA_data.append(AAA_bar)
    
layout_AAA = go.Layout(
    barmode='stack'
)

fig_bs_AAA = go.Figure(data=AAA_data, layout=layout_AAA)
fig_bs.show()
equity_columns = '''
Share Capital & Additional Paid-In Capital
Retained Earnings
Other Equity
Equity Before Minority Interest
'''

equity_columns = [ x for x in equity_columns.strip().split("\n")]

equity_columns
NIKE_BS[ equity_columns ].plot()
equity_data = []
columns = '''
Share Capital & Additional Paid-In Capital
Retained Earnings
Other Equity
Equity Before Minority Interest
'''


for col in columns.strip().split("\n"):
    equity_Scatter = go.Scatter(
        x=NIKE_BS.index,
        y=NIKE_BS[ col ],
        name=col
    )    
    equity_data.append(equity_Scatter)
    
layout_equity = go.Layout(
    barmode='stack'
)

fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)
fig_bs.show()
NIKE_BS["book value"] = NIKE_BS["Total Assets"] - NIKE_BS["Total Liabilities"]

NIKE_BS["book value"].plot()
NIKE_BS["current ratio"] = NIKE_BS["Total Current Assets"] / NIKE_BS["Total Current Liabilities"]
NIKE_BS["current ratio"].plot()
#Calculate 1.	Price-to-Earnings Growth Ratio (PEG forward)  
#using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate
#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp

PE_RATIO = 35.66 # FROM SIMFIN WEBSITE: https://simfin.com/data/companies/121214

# FROM NASDAQ WEBSITE: https://www.nasdaq.com/symbol/xom/earnings-growth
GROWTH_RATE = 0.2069 # Forcast over the five next years

PEG_ratio = PE_RATIO / (GROWTH_RATE*100)

print("NIKE' s PEG Ratio is", 1.7235)