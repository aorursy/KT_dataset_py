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
#type code within tk_library.py into front



import pandas as pd

import numpy as np





def excel_to_df(excel_sheet): 

    df=pd.read_excel(excel_sheet) 

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



def combine_regexes(regexes): return "(" + ")|(".join(regexes) + ")"
import os
#Get excel data downloaded from simfin.com

_, CTSH_PL, CTSH_BS, CTSH_CF=excel_to_df("/kaggle/input/assignment2/CTSH.xlsx")
#The directory access of excel data

!ls ../input/assignment2/CTSH.xlsx
#CTSH's Income Statement

CTSH_PL
#Install chart studio

!pip install chart_studio

!pip install plotly==4.4.1
#Type chart studio's username and api key

import chart_studio

chart_studio.tools.set_credentials_file(username='lisaaawu', api_key='196bSPpDdUz9GzijMJk5')
import chart_studio.plotly as py

import plotly.graph_objs as go
#Draw the graph of revenue

revenues=go.Scatter(

   x=CTSH_PL.index,

   y=CTSH_PL['Revenues'],

   name='Revenues')



net_income=go.Scatter(

   x=CTSH_PL.index,

   y=CTSH_PL["Net income"],

   name='Net Income')





data=[revenues, net_income]

layout=go.Layout(barmode='stack')

fig_revenue=go.Figure(data=data,layout=layout)

#py.iplot(fig_revenue, filename='Revenue')

fig_revenue.show()
import chart_studio.plotly as py

import plotly.graph_objs as go
#Draw the graph of expenses

revenues=go.Scatter(

   x=CTSH_PL.index,

   y=CTSH_PL['Cost of revenues (exclusive of depreciation and amortization expense shown separately below)'],

   name='Cost of Revenues')



general_expense=go.Scatter(

   x=CTSH_PL.index,

   y=CTSH_PL["Selling, general and administrative expenses"],

   name='Selling, General and Administrative Expenses')



depreciation_expense=go.Scatter(

   x=CTSH_PL.index,

   y=CTSH_PL["Depreciation and amortization expense"],

   name='Depreciation and Amortization Expenses')





data=[revenues, general_expense,depreciation_expense ]

layout=go.Layout(barmode='stack')

fig_overhead=go.Figure(data=data,layout=layout)

#py.iplot(fig_overhead, filename='Overhead')

fig_overhead.show()
#CTSH's Statement of Financial Position

CTSH_BS
#Calculate CTSH's Book Value

#Book Value=Total assets-Intangible assets-Total liabilities

CTSH_BS['Book Value']=CTSH_BS['Total assets']-CTSH_BS['Intangible assets, net']-CTSH_BS['Total liabilities']

CTSH_BS
#Calculate CTSH'S Liquidation Value

#Liquidation Value=Total assets-Intangible assets-Goodwill-Total liabilities

CTSH_BS['Liquidation Value']=CTSH_BS['Total assets']-CTSH_BS['Intangible assets, net']-CTSH_BS['Goodwill']-CTSH_BS['Total liabilities']

del CTSH_BS['Assets']

CTSH_BS
#Type chart studio's username and api key

import chart_studio

chart_studio.tools.set_credentials_file(username='lisaaawu', api_key='196bSPpDdUz9GzijMJk5')
import chart_studio.plotly as py

import plotly.graph_objs as go
#Draw the graph of both book value and liquidation value

liquidation_value=go.Scatter(

   x=CTSH_BS.index,

   y=CTSH_BS['Liquidation Value'],

   name='Liquidation Value')



book_value=go.Scatter(

   x=CTSH_BS.index,

   y=CTSH_BS["Book Value"],

   name='Book Value')





data=[liquidation_value, book_value, ]

layout=go.Layout(barmode='stack')

fig_values=go.Figure(data=data,layout=layout)

#py.iplot(fig_values, filename='Different Value')

fig_values.show()
#Install simfin api

!pip install simfin
#Get US market's whole annual income statement

#by following the instructions from https://github.com/SimFin/simfin-tutorials/blob/master/01_Basics.ipynb

import simfin as sf

from simfin.names import *

sf.set_api_key='jDd9vWCMgL3SjymLWEAcchPRzPD6ZrNY'

sf.set_data_dir('~/simfin_data/')

df_income=sf.load(dataset='income', variant='annual', market='us', 

           index=[TICKER,REPORT_DATE], 

           parse_dates=[REPORT_DATE, PUBLISH_DATE])
df_income
#Find CTSH's Income Statement

df_income.loc['CTSH']
#Calculate CTSH's Gross Margin & Net Profit Margin

from pandas import Series, DataFrame

import pandas as pd



data1={'Revenue': Series(df_income.loc['CTSH']['Revenue']), 

      'Gross Profit': Series(df_income.loc['CTSH']['Gross Profit']),

      'Net Profit': Series(df_income.loc['CTSH']['Net Income']),

      'Gross Margin': Series(df_income.loc['CTSH']['Gross Profit'])/Series(df_income.loc['CTSH']['Revenue']),

      'Net Profit Margin':Series(df_income.loc['CTSH']['Net Income'])/Series(df_income.loc['CTSH']['Revenue']) }

df11=pd.DataFrame(data1)

df11
import chart_studio

chart_studio.tools.set_credentials_file(username='lisaaawu', api_key='196bSPpDdUz9GzijMJk5')
import chart_studio.plotly as py

import plotly.graph_objs as go
#Draw the graph of Revenue, Gross Profit, and Net Profit

revenue=go.Scatter(

   x=df11.index,

   y=df11['Revenue'],

   name='Revenue')



gross_profit=go.Scatter(

   x=df11.index,

   y=df11["Gross Profit"],

   name='Gross Profit')



net_profit=go.Scatter(

   x=df11.index,

   y=df11["Net Profit"],

   name='Net Profit')





data=[revenue, gross_profit,net_profit ]

layout=go.Layout(barmode='stack')

fig_revenue=go.Figure(data=data,layout=layout)

#py.iplot(fig_revenue, filename='Revenue')

fig_revenue.show()
import chart_studio

chart_studio.tools.set_credentials_file(username='lisaaawu', api_key='196bSPpDdUz9GzijMJk5')
import chart_studio.plotly as py

import plotly.graph_objs as go
#Draw the graph of Gross Margin and Net Profit Margin

gross_margin=go.Scatter(

   x=df11.index,

   y=df11['Gross Margin'],

   name='Gross Margin')



net_profit_margin=go.Scatter(

   x=df11.index,

   y=df11["Net Profit Margin"],

   name='Net Profit Margin')



data=[gross_margin, net_profit_margin]

layout=go.Layout(barmode='stack')

fig_margin=go.Figure(data=data,layout=layout)

#py.iplot(fig_margin, filename='Margin')

fig_margin.show()
#Get CTSH's number of shares (diluted) from 2009-2018

df_income.loc['CTSH']['Shares (Diluted)']
#Install yahoo finance

!pip install yfinance
#Get CTSH's sotck price of year 2009

import yfinance as yf

df1 = yf.download('CTSH','2009-01-01','2009-12-31')

df1=df1['Close'] 

df1



#Calculate CTSH's 2009 Market Value

#Market Value=number of shares (diluted)*average stock price within 1 year

MV_2009=df_income.loc['CTSH']['Shares (Diluted)']['2009-12-31']*np.average(df1)/1000000

print(MV_2009)
#Get CTSH's stock price of year 2010

df2 = yf.download('CTSH','2010-01-01','2010-12-31')

df2=df2['Close'] 

df2



#Calculate CTSH's 2010 Market Value

#Market Value=number of shares (diluted)*average stock price within 1 year

MV_2010=df_income.loc['CTSH']['Shares (Diluted)']['2010-12-31']*np.average(df2)/1000000

print(MV_2010)
#Get CTSH's sotck price of year 2011

df3 = yf.download('CTSH','2011-01-01','2011-12-31')

df3=df3['Close'] 

df3



#Calculate CTSH's 2011 Market Value

#Market Value=number of shares (diluted)*average stock price within 1 year

MV_2011=df_income.loc['CTSH']['Shares (Diluted)']['2011-12-31']*np.average(df3)/1000000

print(MV_2011)
#Get CTSH's stock price of year 2012

df4 = yf.download('CTSH','2012-01-01','2012-12-31')

df4=df4['Close'] 

df4



#Calculate CTSH's 2012 Market Value

#Market Value=number of shares (diluted)*average stock price within 1 year

MV_2012=df_income.loc['CTSH']['Shares (Diluted)']['2012-12-31']*np.average(df4)/1000000

print(MV_2012)
#Get CTSH's stock price of year 2013

df5 = yf.download('CTSH','2013-01-01','2013-12-31')

df5=df5['Close'] 



#Calculate CTSH's 2013 Market Value

#Market Value=number of shares (diluted)*average stock price within 1 year

df5

MV_2013=df_income.loc['CTSH']['Shares (Diluted)']['2013-12-31']*np.average(df5)/1000000

print(MV_2013)
#Get CTSH's stock price of year 2014

df6 = yf.download('CTSH','2014-01-01','2014-12-31')

df6=df6['Close'] 

df6



#Calculate CTSH's 2014 Market Value

#Market Value=number of shares (diluted)*average stock price within 1 year

MV_2014=df_income.loc['CTSH']['Shares (Diluted)']['2014-12-31']*np.average(df6)/1000000

print(MV_2014)
#Get CTSH's stock price of year 2015

df7 = yf.download('CTSH','2015-01-01','2015-12-31')

df7=df7['Close'] 

df7



#Calculate CTSH's 2015 Market Value

#Market Value=number of shares (diluted)*average stock price within 1 year

MV_2015=df_income.loc['CTSH']['Shares (Diluted)']['2015-12-31']*np.average(df7)/1000000

print(MV_2015)
#Get CTSH's stock price of year 2016

df8 = yf.download('CTSH','2016-01-01','2016-12-31')

df8=df8['Close'] 

df8



#Calculate CTSH's 2016 Market Value

#Market Value=number of shares (diluted)*average stock price within 1 year

MV_2016=df_income.loc['CTSH']['Shares (Diluted)']['2016-12-31']*np.average(df8)/1000000

print(MV_2016)
#Get CTSH's stock price of year 2017

df9 = yf.download('CTSH','2017-01-01','2017-12-31')

df9=df9['Close'] 

df9



#Calculate CTSH's 2017 Market Value

#Market Value=number of shares (diluted)*average stock price within 1 year

MV_2017=df_income.loc['CTSH']['Shares (Diluted)']['2017-12-31']*np.average(df9)/1000000

print(MV_2017)
#Get CTSH's stock price of year 2018

df10 = yf.download('CTSH','2018-01-01','2018-12-31')

df10=df10['Close'] 

df10



#Calculate CTSH's 2018 Market Value

#Market Value=number of shares (diluted)*average stock price within 1 year

MV_2018=df_income.loc['CTSH']['Shares (Diluted)']['2018-12-31']*np.average(df10)/1000000

print(MV_2018)
#Intergrate Market Value Results within one form

from pandas import DataFrame

mv=DataFrame(index=['2009','2010','2011','2012','2013','2014','2015','2016','2017','2018'],

            columns=['Market Value'],

            data=['9103.894139087191','17425.05511115292','22182.00522054126',

                 '20536.932482038887','23529.148879921293','29953.836613917254',

                 '38200.9664734221','34921.04755810329','39340.4516962515','44620.52'])

mv
import chart_studio

chart_studio.tools.set_credentials_file(username='lisaaawu', api_key='196bSPpDdUz9GzijMJk5')
import chart_studio.plotly as py

import plotly.graph_objs as go
#Draw the graph of CTSH's Market Value

market_value=go.Scatter(

   x=mv.index,

   y=mv["Market Value"],

   name='Market Value')

data=[market_value]





layout=go.Layout(barmode='stack',width=900,height=400)

fig_mv=go.Figure(data=data,layout=layout)

#py.iplot(fig_mv, filename='Market Value')

fig_mv.show()

import chart_studio

chart_studio.tools.set_credentials_file(username='lisaaawu', api_key='196bSPpDdUz9GzijMJk5')
import chart_studio.plotly as py

import plotly.graph_objs as go
#Draw graph of CTSH's Book Value

book_value=go.Scatter(

   x=CTSH_BS.index,

   y=CTSH_BS["Book Value"],

   name='Book Value')





data=[book_value]

layout=go.Layout(barmode='stack')

fig_bv=go.Figure(data=data,layout=layout)

#py.iplot(fig_bv, filename='Book Value')

fig_bv.show()
#Draw graph of CTSH's Liquidation Value

liquidation_value=go.Scatter(

   x=CTSH_BS.index,

   y=CTSH_BS["Liquidation Value"],

   name='Liquidation Value')





data=[book_value]

layout=go.Layout(barmode='stack',width=900,height=400)

fig_lv=go.Figure(data=data,layout=layout)

#py.iplot(fig_lv, filename='Liquidation Value')

fig_lv.show()
#Calculate CTSH's Interest Coverage Ratio

#CTSH's EBIT=CTSH's Pretax Income + Interest Expense

#CTSH's Interest Coverage Ratio= EBIT/Interest Expense

#The formula is retrieved from https://www.investopedia.com/terms/i/interestcoverageratio.asp

from pandas import Series, DataFrame

import pandas as pd



data2={'Pretax Income': Series(df_income.loc['CTSH']['Pretax Income (Loss)']),

      'Interest Expense': Series(df_income.loc['CTSH']['Interest Expense, Net']),

       'EBIT': Series(df_income.loc['CTSH']['Pretax Income (Loss)']+df_income.loc['CTSH']['Interest Expense, Net']),

      'Interest Coverage Ratio': Series(df_income.loc['CTSH']['Pretax Income (Loss)']+df_income.loc['CTSH']['Interest Expense, Net'])/Series(df_income.loc['CTSH']['Interest Expense, Net'])

      }

df12=pd.DataFrame(data2)

df12
import chart_studio.plotly as py

import plotly.graph_objs as go
#Draw the graph of CTSH's PreTax income and EBIT

pretax_income=go.Scatter(

   x=df12.index,

   y=df12['Pretax Income'],

   name='Pretax Income')



ebit=go.Scatter(

   x=df12.index,

   y=df12["EBIT"],

   name='EBIT')



data=[pretax_income, ebit]

layout=go.Layout(barmode='stack')

fig_income=go.Figure(data=data,layout=layout)

#py.iplot(fig_income, filename='income')

fig_income.show()
#Draw the graph of CTSH's interest coverage ratio

interest_coverage_ratio=go.Scatter(

   x=df12.index,

   y=df12["Interest Coverage Ratio"],

   name='Interest Coverage Ratio')





data=[interest_coverage_ratio]

layout=go.Layout(barmode='stack',width=900,height=400)

fig_icr=go.Figure(data=data,layout=layout)

#py.iplot(fig_icr, filename='Interest Coverage Ratio')

fig_icr.show()
#Get US market's whole annual balance sheet

import simfin as sf

from simfin.names import *

sf.set_api_key='jDd9vWCMgL3SjymLWEAcchPRzPD6ZrNY'

sf.set_data_dir('~/simfin_data/')

df_balance=sf.load(dataset='balance', variant='annual', market='us', 

           index=[TICKER,REPORT_DATE], 

           parse_dates=[REPORT_DATE, PUBLISH_DATE])
df_balance
#Find CTSH's Balance Sheet

df_balance.loc['CTSH']
import chart_studio

chart_studio.tools.set_credentials_file(username='lisaaawu', api_key='196bSPpDdUz9GzijMJk5')
import chart_studio.plotly as py

import plotly.graph_objs as go
#Draw the graph of CTSH's short-term debt, long-term debt, and total liabilities

short_term_debt=go.Scatter(

   x=df_balance.loc['CTSH'].index,

   y=df_balance.loc['CTSH']["Short Term Debt"],

   name='Short Term Debt')



long_term_debt=go.Scatter(

   x=df_balance.loc['CTSH'].index,

   y=df_balance.loc['CTSH']["Long Term Debt"],

   name='Long Term Debt')



total_liabilities=go.Scatter(

   x=df_balance.loc['CTSH'].index,

   y=df_balance.loc['CTSH']["Total Liabilities"],

   name='Total Liabilities')



data=[short_term_debt,long_term_debt,total_liabilities]

layout=go.Layout(barmode='stack',width=900,height=400)

fig_liabilities=go.Figure(data=data,layout=layout)

#py.iplot(fig_liabilities, filename='Liabilities')

fig_liabilities.show()
#Calculate CTSH's EPS

from pandas import Series, DataFrame

import pandas as pd

data={'Net Income': Series(df_income.loc['CTSH']['Net Income']), 

      'Shares': Series(df_balance.loc['CTSH']['Shares (Diluted)']),

      'EPS': Series(df_income.loc['CTSH']['Net Income']/df_balance.loc['CTSH']['Shares (Diluted)'])

      }

df13=pd.DataFrame(data)

df13
#Calculate CTSH's EPS annunal compound rate

Multiplier=df13['EPS']['2018-12-31']/df13['EPS']['2009-12-31']

print(Multiplier)
#EPS annual compound rate=(EPS at 2018/EPS at 2009)^(1/10)-1

EPS_annual_compound_rate=4.118173160578548**(1/10)-1

print(EPS_annual_compound_rate)
#EPS in 10 years from now (i.e. EPS in 2027): FV=PV(1+annual compound rate)^10

EPS_2027=df13['EPS']['2018-12-31']*(1+EPS_annual_compound_rate)**10

print(EPS_2027)
#Calculate CTSH's P/E Ratio from 2009 to 2019

#P/E Ratio=stock price per share/EPS

#The formula is retrieved from https://www.investopedia.com/terms/p/price-earningsratio.asp

#Here we use the average of 1 year's stock price as stock price per share



PE_2009=np.average(df1)/df13['EPS']['2009-12-31']

PE_2010=np.average(df2)/df13['EPS']['2010-12-31']

PE_2011=np.average(df3)/df13['EPS']['2011-12-31']

PE_2012=np.average(df4)/df13['EPS']['2012-12-31']

PE_2013=np.average(df5)/df13['EPS']['2013-12-31']

PE_2014=np.average(df6)/df13['EPS']['2014-12-31']

PE_2015=np.average(df7)/df13['EPS']['2015-12-31']

PE_2016=np.average(df8)/df13['EPS']['2016-12-31']

PE_2017=np.average(df9)/df13['EPS']['2017-12-31']

PE_2018=np.average(df10)/df13['EPS']['2018-12-31']

print(PE_2009,PE_2010,PE_2011,PE_2012,

      PE_2013,PE_2014,PE_2015,PE_2016,PE_2017,PE_2018)
#Intergrate P/E Results within one form

from pandas import DataFrame

PE=DataFrame(index=['2009','2010','2011','2012','2013','2014','2015','2016','2017','2018'],

            columns=['P/E'],

            data=['17.215494298227068','23.957684323553362','24.98034099953526',

                 '19.40244273581416','19.192209986152285','20.82592625492022',

                 '23.49974995789728','22.519362791472137','25.937406692280955','21.128655402189434'])

PE
#Estimate CTSH's Stock Price 10 Years from now



average_PE=(PE_2016+PE_2017+PE_2018)/3

print(average_PE)

Stock_Price_2027=EPS_2027*average_PE

print(Stock_Price_2027)
#Calculate CTSH's Target Buy Price today

Discount_Rate=0.08

#this data is retrieved from https://finbox.com/NASDAQGS:CTSH/models/wacc

Year=10 

Stock_Price_2027=345.42323924721387

Target_Price=160

#this data is calculated by http://www.moneychimp.com/calculator/present_value_calculator.htm
#Add margin of safety,due to the spread of coronavirus, use a medium margin of safety

margin_safety=0.20

Buy_Price=Target_Price*(1-margin_safety)

print(Buy_Price)
#Prepare data for ARIMA model

df = pd.read_csv('../input/supplementary/arima.csv')



df['Date'] = pd.to_datetime(df['Date'])

df.sort_values('Date', inplace=True)

df.set_index('Date', inplace=True)



print(df.shape)



df.head()
# Reorganize the data and make CTSH's closing prices are aggregated on a weekly basis

df_week = df.resample('w').mean()

df_week = df_week[['Close']]

df_week.head()
#Create a column for weekly returns

#Take the log to do normalization 

df_week['weekly_ret'] = np.log(df_week['Close']).diff()

df_week.head()
# Drop null rows

df_week.dropna(inplace=True)
#draw the graph

df_week.weekly_ret.plot(kind='line', figsize=(12, 6));
udiff = df_week.drop(['Close'], axis=1)

udiff.head()
#Prepare tools for ARIMA model

import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller
#Test stationarity for udiff series

rolmean = udiff.rolling(20).mean()

rolstd = udiff.rolling(20).std()
#Draw the graph

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

orig = plt.plot(udiff, color='blue', label='Original')

mean = plt.plot(rolmean, color='red', label='Rolling Mean')

std = plt.plot(rolstd, color='black', label = 'Rolling Std Deviation')

plt.title('Rolling Mean & Standard Deviation')

plt.legend(loc='best')

plt.show(block=False)
# Perform Dickey-Fuller test

df_test = sm.tsa.adfuller(udiff.weekly_ret, autolag='AIC')

df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

for key, value in df_test[4].items():

    df_output['Critical Value ({0})'.format(key)] = value

    

df_output
#Draw the graph of auto correlation

from statsmodels.graphics.tsaplots import plot_acf



# the autocorrelation chart provides just the correlation at increasing lags

fig, ax = plt.subplots(figsize=(12,5))

plot_acf(udiff.values, lags=10, ax=ax)

plt.show()
#Draw the graph of partial auto correlation

from statsmodels.graphics.tsaplots import plot_pacf



fig, ax = plt.subplots(figsize=(12,5))

plot_pacf(udiff.values, lags=10, ax=ax)

plt.show()
#Build ARIMA Model

from statsmodels.tsa.arima_model import ARMA



# Notice that you have to use udiff - the differenced data rather than the original data. 

ar1 = ARMA(tuple(udiff.values), (3, 1)).fit()

ar1.summary()
#Draw the graph

#The blue line refers to the actual price

#The red line refers to the prediction price

plt.figure(figsize=(12, 8))

plt.plot(udiff.values, color='blue')

preds = ar1.fittedvalues

plt.plot(preds, color='red')

plt.show()