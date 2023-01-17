#plotly

import pandas as pd

import numpy as np



import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

from plotly import tools

import plotly.figure_factory as ff



import warnings

warnings.filterwarnings('ignore')
#load information

df = pd.read_csv('../input/all_stocks_5yr.csv')

df['date'] = pd.to_datetime(df['date'])

df.head()
#Explore info

df.info
#read data of ADM and AMZN

ADM = df[df.Name=='ADM']

AMZN = df[df.Name=='AMZN']

#We need to calculate mean and standard deviation in periods of 20 days (It represents monthly info)

#But with the format we have, we can not because 'date' and 'Name' columns do not have proper format.

ADM.head()
ADM.index = ADM['date']

AMZN.index = AMZN['date']
ADM.head()
#now we remove 'date' and 'Name' columns

ADM = ADM.drop(['Name','date'],axis=1)
AMZN = AMZN.drop(['Name','date'],axis=1)
AMZN.head()
#Now, we create dataframes with sample moving average of 20 days

ADM_SMA = ADM.rolling(window=20).mean()

AMZN_SMA = AMZN.rolling(window=20).mean()

#Then, we see that during first 20 days we do not have values because the size of each sample

ADM_SMA.head(21)
#Now, we create plots that reflects behaviour of the mean from each sample in close prices

trace_ADM_cl = go.Scatter(x=ADM.index,y=ADM.close,name='ADM close')

trace_ADM_cl_mean = go.Scatter(x=ADM_SMA.index,y=ADM_SMA.close,name='ADM 20 days MA')

py.iplot([trace_ADM_cl,trace_ADM_cl_mean])

#It seems to me a pretty nice looking graph. You can visualize the evolution of prices along the curve

#And you can zoom any part you want examine
#We do the same for AMZN

trace_AMZN_cl = go.Scatter(x=AMZN.index,y=AMZN.close,name='AMZN close')

trace_AMZN_cl_mean = go.Scatter(x=AMZN_SMA.index,y=AMZN_SMA.close,name='AMZN 20 days MA')

py.iplot([trace_AMZN_cl,trace_AMZN_cl_mean])
#Here we can observe an upward trend of prices of AMAZON while the prices of ADM are tending to be

#more stable.

#Let's create two bands with standard deviation:

up_std_ADM = ADM_SMA[['close']] + 2*ADM_SMA[['close']].rolling(window=20).std()

lo_std_ADM = ADM_SMA[['close']] - 2*ADM_SMA[['close']].rolling(window=20).std()



#Now we plot Bollinger bands



drw_std_up = go.Scatter(x=up_std_ADM.index,y=up_std_ADM.close,name='20 MA upper band ADM')

drw_std_lo = go.Scatter(x=lo_std_ADM.index,y=lo_std_ADM.close,name='20 MA lower band ADM')



py.iplot([trace_ADM_cl,trace_ADM_cl_mean,drw_std_up,drw_std_lo])







#And, for AMZN we obtain

up_std_AMZN = AMZN_SMA[['close']] + 2*AMZN_SMA[['close']].rolling(window=20).std()

lo_std_AMZN = AMZN_SMA[['close']] - 2*AMZN_SMA[['close']].rolling(window=20).std()



#Now we plot Bollinger bands



drw_std_up_amzn = go.Scatter(x=up_std_AMZN.index,y=up_std_AMZN.close,name='20 MA upper band AMZN')

drw_std_lo_amzn = go.Scatter(x=lo_std_AMZN.index,y=lo_std_AMZN.close,name='20 MA lower band AMZN')



py.iplot([trace_AMZN_cl,trace_AMZN_cl_mean,drw_std_up_amzn,drw_std_lo_amzn])
#We calculate proportion of prices that are above and below of Bollinger band

above_adm = ADM[ADM['close'] > up_std_ADM['close']]

below_adm = ADM[ADM['close'] < lo_std_ADM['close']]

print(above_adm.shape)

print(below_adm.shape)
#And the proportion of data out of the band is

prop_adm = (above_adm.shape[0]+below_adm.shape[0])/ADM.shape[0]

print(prop_adm)
#For AMZN we have

above_amzn = AMZN[AMZN['close'] > up_std_AMZN['close']]

below_amzn = AMZN[AMZN['close'] < lo_std_AMZN['close']]

prop_amzn = (above_amzn.shape[0]+below_amzn.shape[0])/AMZN.shape[0]

print(prop_amzn)