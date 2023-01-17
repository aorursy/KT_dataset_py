# Do the below in kaggle

!pip install plotly==4.4.1

!pip install chart_studio

!pip install xlrd # for reading_excel files with extensions .xlsx into a pandas dataframe
# Add 'tk_library.py' file given by your tutor, as a utility script under 'File'

# Look for it under usr/bin on the right drawer



# import excel_to_df function

import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import tk_library_py

#from tk_library_py import excel_to_df

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
# Show the files and their pathnames

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Add your simfin-data.xlsx using the '+ Add Data' top right button

_, Columbia_PL, Columbia_BS, Columbia_CF = excel_to_df("/kaggle/input/simfindata-columbia/SimFin-data  Columbia.xlsx")
Columbia_BS
Columbia_BS["_Total Current Assets"] = Columbia_BS["Cash, Cash Equivalents & Short Term Investments"] + Columbia_BS["Accounts & Notes Receivable"] + Columbia_BS["Inventories"] + Columbia_BS["Other Short Term Assets"]
Columbia_BS[["_Total Current Assets", "Total Current Assets"]]
Columbia_BS["_NonCurrent Assets"] = Columbia_BS["Property, Plant & Equipment, Net"] + Columbia_BS["Other Long Term Assets"]
Columbia_BS["_Total Assets"] = Columbia_BS["_NonCurrent Assets"] + Columbia_BS["_Total Current Assets"] 
Columbia_BS["_Total Liabilities"] = Columbia_BS["Total Current Liabilities"] + Columbia_BS["Total Noncurrent Liabilities"]
Columbia_BS[["_Total Liabilities", "Total Liabilities"]]
%matplotlib inline

Columbia_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

'''



asset_columns = [ x for x in good_stuff.strip().split("\n") ]

asset_columns
Columbia_BS[ asset_columns ].plot()
import chart_studio

# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 

# https://chart-studio.plot.ly/feed/#/



# Un-remark the code below and add your own your_username and own your_apikey

chart_studio.tools.set_credentials_file(username='999', api_key='PwRbafUPnVPslR9zDFFy')
import chart_studio.plotly as py

import plotly.graph_objs as go

#from tk_library_py import combine_regexes

assets = go.Bar(

    x=Columbia_BS.index,

    y=Columbia_BS["Total Assets"],

    name='Assets'

)

liabilities = go.Bar(

    x=Columbia_BS.index,

    y=Columbia_BS["Total Liabilities"],

    name='Liabilities'

)



shareholder_equity = go.Scatter(

    x=Columbia_BS.index,

    y=Columbia_BS["Total Equity"],

    name='Equity'

)



data = [assets, liabilities, shareholder_equity]

layout = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

#py.plot(fig_bs_assets, filename='Total Assets and Liabilities')

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

        x=Columbia_BS.index,

        y=Columbia_BS[ col ],

        name=col

    )    

    asset_data.append(asset_bar)

    

layout_assets = go.Layout(

    barmode='stack'

)



fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)

#py.plot(fig_bs_assets, filename='Total Assets Breakdown')

fig_bs.show()
liability_data = []

columns = '''

Payables & Accruals

Short Term Debt

Other Short Term Liabilities

Other Long Term Liabilities

'''





for col in columns.strip().split("\n"):

    liability_bar = go.Bar(

        x=Columbia_BS.index,

        y=Columbia_BS[ col ],

        name=col

    )    

    liability_data.append(liability_bar)

    

layout_liabilitys = go.Layout(

    barmode='stack'

)



fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)

#py.plot(fig_bs_liabilitys, filename='Total liabilities Breakdown')

fig_bs.show()
Columbia_BS["working capital"] = Columbia_BS["Total Current Assets"] - Columbia_BS["Total Current Liabilities"]
Columbia_BS[["working capital"]].plot()
Columbia_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
# Using Chart Studio in Plotly 



PR_data = []

columns = '''

Accounts & Notes Receivable

Payables & Accruals

'''



for col in columns.strip().split("\n"):

    PR_Scatter = go.Scatter(

        x=Columbia_BS.index,

        y=Columbia_BS[ col ],

        name=col

    )    

    PR_data.append(PR_Scatter)

    

layout_PR = go.Layout(

    barmode='stack'

)



fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)

#py.plot(fig_bs_PR, filename='Accounts & Notes Receivable vs Payables & Accruals')

fig_bs.show()
Columbia_BS["Inventories"].plot()
Columbia_BS[ ["Property, Plant & Equipment, Net",  "Other Long Term Assets"] ].plot()
# Using Plotly



AAA_data = []

columns = '''

Property, Plant & Equipment, Net

Other Long Term Assets

'''





for col in columns.strip().split("\n"):

    AAA_bar = go.Bar(

        x=Columbia_BS.index,

        y=Columbia_BS[ col ],

        name=col

    )    

    AAA_data.append(AAA_bar)

    

layout_AAA = go.Layout(

    barmode='stack'

)



fig_bs_AAA = go.Figure(data=AAA_data, layout=layout_AAA)

#py.plot(fig_bs_AAA, filename='Total Long Term Assets')

fig_bs.show()
equity_columns = '''

Share Capital & Additional Paid-In Capital

Treasury Stock

Retained Earnings

Other Equity

Equity Before Minority Interest

Minority Interest

'''



equity_columns = [ x for x in equity_columns.strip().split("\n")]

equity_columns
# Using Plotly



equity_data = []

columns = '''

Share Capital & Additional Paid-In Capital

Retained Earnings

Other Equity

Equity Before Minority Interest

Minority Interest

'''





for col in columns.strip().split("\n"):

    equity_Scatter = go.Scatter(

        x=Columbia_BS.index,

        y=Columbia_BS[ col ],

        name=col

    )    

    equity_data.append(equity_Scatter)

    

layout_equity = go.Layout(

    barmode='stack'

)



fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)

#py.plot(fig_bs_equity, filename='Total Equity')

fig_bs.show()
# Exxon mobil has no preferred stock, no intengible assets, and no goodwill



Columbia_BS["book value"] = Columbia_BS["Total Assets"] - Columbia_BS["Total Liabilities"]

Columbia_BS["current ratio"] = Columbia_BS["Total Current Assets"] / Columbia_BS["Total Current Liabilities"]
Columbia_BS["current ratio"].plot()
#Calculate 1.	Price-to-Earnings Growth Ratio (PEG forward)  

#using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate

#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp



PE_RATIO = 20.64 # FROM SIMFIN WEBSITE: https://simfin.com/data/companies/121214



# FROM NASDAQ WEBSITE: https://www.nasdaq.com/symbol/xom/earnings-growth

GROWTH_RATE = 0.0857 # Forcast over the five next years



PEG_ratio = PE_RATIO / (GROWTH_RATE*100)



print("Columbia Mobil Corp's PEG Ratio is", PEG_ratio)