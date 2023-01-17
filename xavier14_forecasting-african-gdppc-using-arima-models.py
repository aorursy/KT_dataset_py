import pandas as pd

from pandas import DataFrame

import matplotlib as plt

import matplotlib.pyplot as plt

from matplotlib import pyplot

import seaborn

import warnings

import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.preprocessing import scale

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.figure_factory as ff

init_notebook_mode(connected=True)

import statsmodels.formula.api as sm

from statsmodels.compat import lzip

import numpy as np

import statsmodels.api as sm

from plotly.plotly import plot_mpl

from plotly.offline import init_notebook_mode, iplot_mpl
df2=pd.read_csv('../input/WDIData.csv')
df2=df2[(df2['Country Name'].str.contains('African|Benin|Burkina|Burundi|Cabo Verde|Cameroon|CAR|Chad|Comoros|Congo|Ivoire|Guinea|Eritrea|Eswatini|Ethiopia|Gambia|Ghana|Guinea|Guinea-Bissau|Kenya|Lesotho|Liberia|Madagascar|Malawi|Mali|Mauritania|Mauritius|Mozambique|Niger|Nigeria|Rwanda|Sao Tome|Senegal|Sierra Leone|South Sudan|Sudan|Tanzania|Togo|Uganda|Zambia|Zimbabwe')==True)]

df2['Indicator Name']=df2['Indicator Name'].str.replace('(',':').str.replace(')',':')

df2=df2[df2['Indicator Name'].str.contains("GDP per capita, PPP :current international")==True]

df2.index=df2['Country Name']

df2=df2[df2.index.str.contains('Sao Tome and Principe|South Sudan|Papua New Guinea|Equatorial Guinea|Mauritius')==False]

df2=df2.fillna(method='bfill',axis=1)

df2=df2.fillna(method='ffill',axis=1)

df2=df2.iloc[:,34:-2]

ts=df2.mean()

ts=pd.DataFrame(ts)

ts.columns=['GDP per capita']
X=pd.to_datetime(ts.index)

y=ts['GDP per capita']

layout = go.Layout(title= 'Sub-Saharan African GDP per capita PPP 1990-2017 current international $',

    xaxis = dict(ticks='', nticks=43),

    yaxis = dict(nticks=20))

data = [go.Scatter(x=X,y=y)]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='heatmap',show_link=False)
ts_2=ts

ts_2['log_GDP_pc']=np.log(ts['GDP per capita'])

X=ts['log_GDP_pc'].index

y_1=ts['log_GDP_pc']

layout = go.Layout(title= 'Sub-Saharan African LOG GDP per capita PPP 1990-2017 current international $',

    xaxis = dict(ticks='', nticks=43),

    yaxis = dict(nticks=20),legend=dict(x=0.7, y=1))

trace0 = go.Scatter(x=X,y=y_1, mode = 'lines',name="SSA GDPpc PPP $ growth 1990-2017", marker = dict(size=12, color='red'))

data = [trace0]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='heatmap',show_link=False)
f, ax = plt.subplots(figsize=(15, 5)); plt.title('Distribution')

sns.distplot(ts['GDP per capita'],color='blue', bins=18,rug=True,hist_kws={"density":True})

plt.rc('xtick', labelsize=14)    # fontsize of the tick X labels 

plt.rc('ytick', labelsize=14)    # fontsize of the tick Y labels

plt.title('TIME SERIES - Histogram and probability density function (PDF)', size=18)

plt.xlabel('GDP per capita annual growth', size=16)

plt.show()
ts_1=ts

ts_1['mean']=ts['GDP per capita'].pct_change().fillna(0).mean()

X=ts['GDP per capita'].index

y_2=ts['GDP per capita'].pct_change().fillna(0)

y1=ts_1['mean']

layout = go.Layout(title= 'Sub-Saharan African GDP per capita PPP growth dynamic 1990-2017 current international $',

    xaxis = dict(ticks='', nticks=43),

    yaxis = dict(nticks=20),legend=dict(x=0.7, y=1))

trace0 = go.Scatter(x=X,y=y_2, mode = 'lines',name="SSA GDPpc PPP $ growth 1990-2017", marker = dict(size=12, color='red'))

trace1 = go.Scatter(x=X,y=y1, mode = 'lines', name="TS Mean ", marker = dict(size=12, color='darkcyan'))

data = [trace0,trace1]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='heatmap',show_link=False)
returns1 = ts['GDP per capita'].pct_change().fillna(0)

ret_index_1 = (1+returns1).cumprod()

X_1=ret_index_1.index

y_1=ret_index_1.values

layout = go.Layout(title= 'Sub-Saharan African GDP per capita PPP 1990-2017 accumulative growth (Index 1=1990)',

    xaxis = dict(ticks='', nticks=43),

    yaxis = dict(nticks=20))

data = [go.Bar(x=X_1,y=y_1,marker =dict(color='lightsalmon'))]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='heatmap',show_link=False)
trace1 = go.Box(

    y=ts['GDP per capita'].pct_change().fillna(0),

    name='Time series growth dynamic with Mean & SD',boxpoints='all',

    marker=dict(

        color='darkcyan',opacity=0.7),boxmean='sd')

data = [trace1]

layout = go.Layout(title='')

fig =go.Figure(data=data, layout=layout)

iplot(data)
fig = {"data": [{"type": 'violin', "y": y, "box": {"visible": True}, "line": {"color": 'black'},"meanline": 

                 {"visible": True },"fillcolor": 'darkcyan',"opacity": 0.7,"x0": 'Violin'}],

    "layout" : { "title": "Violin Plot - Time series growth dynamic", "yaxis": { "zeroline": False,}}}



fig =go.Figure(data=fig, layout=layout)

iplot(fig)
f, ax = plt.subplots(figsize=(15, 5)); plt.title('Distribution')

sns.distplot(ts['GDP per capita'].pct_change().fillna(0),bins=15, hist_kws={'cumulative': True},color='darkcyan', 

             kde_kws={'cumulative': True},rug=True)

plt.rc('xtick', labelsize=14)    # fontsize of the tick X labels 

plt.rc('ytick', labelsize=14)    # fontsize of the tick Y labels

plt.title('Time series growth dynamic - Cumulative Distribution Function (CDF)', 

          size=18)

SIZE2=12

plt.xlabel('GDPpc PP anual growth', size=14)

plt.ylabel('Accumulated probability [0,1]', size=14)

plt.show()
f, ax = plt.subplots(figsize=(16, 5)); plt.title('Distribution')

sns.distplot(ts['GDP per capita'].pct_change().fillna(0),color='darkcyan', bins=20,rug=True,hist_kws={"density":True})

plt.rc('xtick', labelsize=SIZE2)    # fontsize of the tick X labels 

plt.rc('ytick', labelsize=SIZE2)    # fontsize of the tick Y labels

plt.title('Time series growth dynamic - Histogram and probability density function (PDF)', size=18)

plt.xlabel('GDP per capita annual growth', size=14)

plt.show()
import statsmodels.api as sm

fig, ax = plt.subplots(figsize=(15,5))

ax.set_facecolor('silver')

sm.graphics.tsa.plot_acf(y,ax=ax,lags=27)

plt.show()
import statsmodels.api as sm

fig, ax = plt.subplots(figsize=(15,5))

ax.set_facecolor('silver')

sm.graphics.tsa.plot_pacf(y,ax=ax)

plt.show()
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):



    #Determing rolling statistics

    rolmean = timeseries.rolling(window=12).mean()

    rolstd = timeseries.rolling(window=12).std()



    #Plot rolling statistics:

    fig = plt.figure(figsize=(12, 8))

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.rc('xtick', labelsize=8); plt.rc('ytick', labelsize=16) 

    plt.xticks([])

    

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    

    plt.show()

    

    #Perform Dickey-Fuller test:

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)
test_stationarity(y)
ts.index = pd.to_datetime(ts.index)

from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = 13, 7

SIZE2=15  

plt.rc('xtick', labelsize=SIZE2)    # fontsize of the tick X labels 

plt.rc('ytick', labelsize=SIZE2)    # fontsize of the tick Y labels

decomposition = seasonal_decompose(ts, model='additive')

fig = decomposition.plot()

plt.show()
# first order difference of the time series

y_diff = (y).diff().dropna()

y_diff = pd.Series(y_diff )
layout = go.Layout(title= 'Sub-Saharan African GDP First order differencing',

xaxis = dict(ticks='', nticks=43),

yaxis = dict(nticks=20),legend=dict(x=0.7, y=1))



trace0 = go.Scatter(x=y_diff.index, y=y_diff.values, mode = 'lines+markers',name="SSA GDPpc PPP $ growth 1990-2017", marker = dict(size=12, color='black'))

data = [trace0]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='heatmap',show_link=False)
test_stationarity(y_diff)
# log difference time series

y_diff2 = (y).diff().diff().dropna()

y_diff2 = pd.Series(y_diff2)

test_stationarity(y_diff2)
layout = go.Layout(title= 'Sub-Saharan African GDP Second order differencing', xaxis = dict(ticks='', nticks=43),yaxis = dict(nticks=20),legend=dict(x=0.7, y=1))

trace0 = go.Scatter(x=y_diff2.index, y=y_diff2.values, mode = 'lines+markers',name="SSA GDPpc PPP $ growth 1990-2017", marker = dict(size=12, color='black'))

data = [trace0]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='heatmap',show_link=False)
fig, ax = plt.subplots(figsize=(15,5))

ax.set_facecolor('silver')

plt.rc('xtick', labelsize=16)

sm.graphics.tsa.plot_acf(y_diff2,ax=ax,lags=25)

plt.show()
fig, ax = plt.subplots(figsize=(15,5))

ax.set_facecolor('silver')

sm.graphics.tsa.plot_pacf(y_diff2,ax=ax, lags=25)

plt.show()
mod = sm.tsa.statespace.SARIMAX(ts['GDP per capita'].values,order=(3,2,3),enforce_stationarity=True,enforce_invertibility=True, maxiter=1000, method='css')

results1 = mod.fit()

print(results1.summary())
results1.plot_diagnostics(figsize=(15, 12),lags=25)

plt.show()
prediction_summary=results1.get_prediction(start=0, end=40).summary_frame()

prediction_summary.index=['1990-01-01', '1991-01-01','1992-01-01','1993-01-01', '1994-01-01', '1995-01-01',

               '1996-01-01', '1997-01-01', '1998-01-01', '1999-01-01',

               '2000-01-01', '2001-01-01', '2002-01-01', '2003-01-01',

               '2004-01-01', '2005-01-01', '2006-01-01', '2007-01-01',

               '2008-01-01', '2009-01-01', '2010-01-01', '2011-01-01',

               '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01',

               '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01',

               '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01',

               '2024-01-01', '2025-01-01', '2026-01-01', '2027-01-01',

               '2028-01-01', '2029-01-01', '2030-01-01']
forecast=results1.predict(start=0,end=40)

forecast=pd.DataFrame(forecast, columns = ['projection'])

forecast.index=['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998',

       '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007',

       '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016',

       '2017','2018','2019','2020','2021','2022','2023','2024','2025','2026','2027','2028','2029','2030']

# Create traces

trace0 = go.Scatter(

    x = ts.index,

    y = ts['GDP per capita'],

    mode = 'markers',

    name = 'Actual GDP PPP',marker = dict(size=7))

trace1 = go.Scatter(

    x = forecast.iloc[2:].index,

    y = forecast.iloc[2:].projection,

    mode = 'lines+markers',

    name = 'Arima model (5,2,0) in sample model & 2030 projection',marker = dict(size=7),opacity = 0.6)





layout = go.Layout(title= 'In-sample prediction and out-of-sample forecasting to 2030 per capita - GDP PPP US$ SSA',

    xaxis = dict(ticks='', nticks=43),

    yaxis = dict(nticks=20), legend=dict(x=0.1, y=1))

data = [trace0, trace1]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='scatter-mode')
# Create traces

trace0 = go.Scatter(

    x = ts.index,

    y = ts['GDP per capita'],

    mode = 'markers',

    name = 'Actual GDP PPP',marker = dict(size=7))





trace2 =go.Scatter(x=prediction_summary.iloc[2:].index, y=prediction_summary.iloc[2:].mean_ci_lower.values, name='ARIMA model 95% Lower CI', mode = 'lines',

                   marker = dict(size=10, color='red'),opacity = 0.3)



trace3 =go.Scatter(x=prediction_summary.iloc[2:].index, y=prediction_summary.iloc[2:].mean_ci_upper.values,name='ARIMA model 95% Upper CI', mode = 'lines',

                   marker = dict(size=10, color='red'),opacity = 0.3)



trace4 =go.Scatter(x=prediction_summary.iloc[28:].index, y=prediction_summary.iloc[28:]['mean'].values,name='ARIMA model mean projected values', mode = 'markers',

                   marker = dict(size=10, color='red'),opacity = 0.3)



layout = go.Layout(title= 'In-sample prediction and out-of-sample forecasting 95% CI - GDPpc PPP US$ SSA',

    xaxis = dict(ticks='', nticks=43),

    yaxis = dict(nticks=20), legend=dict(x=0.1, y=1))

data = [trace0,trace2,trace3,trace4]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='scatter-mode')
df3=pd.read_csv('../input/WDIData.csv')

df3=df3[df3['Country Name']=='World']

df3['Indicator Name']=df3['Indicator Name'].str.replace('(',':').str.replace(')',':')

df3=df3[df3['Indicator Name'].str.contains("GDP per capita, PPP :current international")==True]

df3.index=df3['Country Name']

df3=df3.dropna(axis=1)

df3.drop(['Country Name','Country Code','Indicator Name','Indicator Code'], axis=1, inplace=True)

ts2=df3 ; ts2=ts2.T

ts2.index = pd.to_datetime(ts2.index)

ts2.columns=['GDP per capita']

ts2['world_percentatge']=ts['GDP per capita']/ts2['GDP per capita']
layout = go.Layout(title= 'World & SSA GDP per capita PPP constant $ 1990-2017',

xaxis = dict(ticks='', nticks=43),

yaxis = dict(nticks=20), legend=dict(x=0.1, y=1))

trace0 =go.Scatter(x=ts2.index, y=ts2['GDP per capita'],

                   name='World GDP per capita PPP TS 1990-2017', marker = dict(size=12, color='green'))

trace1 =go.Scatter(x=ts.index,y=ts['GDP per capita'],

                   name='SSA GDP per capita PPP TS 1990-2017',marker = dict(size=12, color='blue'))

data = [trace0]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='heatmap',show_link=False)
trace1 = go.Box(

    y=ts['GDP per capita'].pct_change().fillna(0),

    name='SSA GDPpc growth TS',

    marker=dict(

        color='blue'),boxmean='sd')



trace2 = go.Box(

    y=ts2['GDP per capita'].pct_change().fillna(0),

    name='World GDPpc growth TS',

    marker=dict(

        color='green'),boxmean='sd')

data = [trace1,trace2]

fig =go.Figure(data=data, layout=layout)

iplot(data)
SSA_growth=ts['GDP per capita'].pct_change().fillna(0)

World_growth=ts2['GDP per capita'].pct_change().fillna(0)
layout = go.Layout(title= 'SSA vs World GDP per capita PPP growth % 1990-2017',

    xaxis = dict(ticks='', nticks=43), yaxis = dict(nticks=20),legend=dict(x=0.1, y=1))



trace0 =go.Scatter(x=ts.index, y=SSA_growth*100, name='SSA GDP per capita PPP anual growth', mode = 'markers+lines',

                   marker = dict(size=12, color='blue'))



trace1 =go.Scatter(x=ts2.index, y=World_growth*100,name='World GDP per capita PPP anual growth', mode = 'markers+lines',

                   marker = dict(size=12, color='green'))

data = [trace0, trace1]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='heatmap',show_link=False)
ts2['world_percentatge']=ts/ts2

layout = go.Layout(title= 'SSA GDP per capita PPP annual ratio with the world average 1990-2017',

    xaxis = dict(ticks='', nticks=43),

    yaxis = dict(nticks=20))

trace0 =go.Scatter(x=ts2.index, y=ts2['world_percentatge'],

                   mode = 'markers+lines',marker = dict(size=12, color='coral'))

data = [trace0]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='heatmap',show_link=False)
ts2.index = pd.to_datetime(ts2.index)
# log difference time series

ts2_diff2 = ((ts2['GDP per capita']).diff().diff().fillna(0))

ts2_diff2 = pd.Series(ts2_diff2)

test_stationarity(ts2_diff2)
fig, ax = plt.subplots(figsize=(15,5))

ax.set_facecolor('silver')

sm.graphics.tsa.plot_acf(ts2_diff2,ax=ax, lags=25)

plt.show()
mod = sm.tsa.statespace.SARIMAX((ts2['GDP per capita']), order=(0,2,2),

                                enforce_stationarity=True, enforce_invertibility=True,)

results2 = mod.fit()

print(results2.summary())
results2.plot_diagnostics(figsize=(15, 12),lags=20)

plt.show()
prediction_summary2=results2.get_prediction(start=0, end=40).summary_frame()

prediction_summary2.index=['1990-01-01', '1991-01-01','1992-01-01','1993-01-01', '1994-01-01', '1995-01-01',

               '1996-01-01', '1997-01-01', '1998-01-01', '1999-01-01',

               '2000-01-01', '2001-01-01', '2002-01-01', '2003-01-01',

               '2004-01-01', '2005-01-01', '2006-01-01', '2007-01-01',

               '2008-01-01', '2009-01-01', '2010-01-01', '2011-01-01',

               '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01',

               '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01',

               '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01',

               '2024-01-01', '2025-01-01', '2026-01-01', '2027-01-01',

               '2028-01-01', '2029-01-01', '2030-01-01']
forecast2=results2.predict(start=2,end=40)



forecast1=results1.predict(start=0,end=40)



# Create traces

trace0 = go.Scatter(

    x = ts.index,

    y = ts['GDP per capita'],

    mode = 'markers',

    name = 'SSA Actual GDP PPP',marker = dict(size=7))



trace1 = go.Scatter(

    x = forecast.iloc[2:].index,

    y = forecast.iloc[2:].projection,

    mode = 'lines',

    name = 'SSA Arima model (5,2,0) prediction',marker = dict(size=7),opacity = 0.5)



# Create traces

trace2 = go.Scatter(

    x = ts2.index,

    y = ts2['GDP per capita'],

    mode = 'markers',

    name = 'World Actual GDP PPP',marker = dict(size=7))





trace3 = go.Scatter(

    x = forecast2.index,

    y = forecast2.values,

    mode = 'lines',

    name = 'World Arima model (0,2,1) prediction',marker = dict(size=7),opacity = 0.5)





layout = go.Layout(title= 'In-sample prediction and out-of-sample projection to 2030 (GDPpc PPP US$) SSA & WA',

    xaxis = dict(ticks='', nticks=43),

    yaxis = dict(nticks=20), legend=dict(x=0.1, y=1))



data = [trace0, trace1,trace2,trace3]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='scatter-mode')
forecast

# Create traces

trace0 = go.Scatter(

    x = ts.index,

    y = ts['GDP per capita'],

    mode = 'markers',

    name = 'Actual GDP PPP',marker = dict(size=7))





trace2 =go.Scatter(x=prediction_summary.iloc[2:].index, y=prediction_summary.iloc[2:].mean_ci_lower.values, name='95% CI', mode = 'lines',

                   marker = dict(size=10, color='red'),opacity = 0.3)



trace3 =go.Scatter(x=prediction_summary.iloc[2:].index, y=prediction_summary.iloc[2:].mean_ci_upper.values, name='95% CI', mode = 'lines',

   marker = dict(size=10, color='red'),opacity = 0.3)



# Create traces

trace4 = go.Scatter(

    x = ts2.index,

    y = ts2['GDP per capita'],

    mode = 'markers',

    name = 'World Actual GDP PPP',marker = dict(size=7,color='green'))



trace5 =go.Scatter(x=prediction_summary2.iloc[2:].index, y=prediction_summary2.iloc[2:].mean_ci_lower.values, name='95% CI', mode = 'lines',

                   marker = dict(size=10, color='red'),opacity = 0.3)



trace6 =go.Scatter(x=prediction_summary2.iloc[2:].index, y=prediction_summary2.iloc[2:].mean_ci_upper.values, name='95% CI', mode = 'lines',

                   marker = dict(size=10, color='red'),opacity = 0.3)





layout = go.Layout(title= 'In-sample prediction and out-of-sample projection to 2030 95% CI (GDPpc PPP US$) SSA & WA',

    xaxis = dict(ticks='', nticks=43),

    yaxis = dict(nticks=20), legend=dict(x=0.1, y=1))

data = [trace0,trace2,trace3,trace4,trace5,trace6]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='scatter-mode')
trace5 =go.Scatter(x=prediction_summary2.iloc[2:].index, y=prediction_summary2.iloc[2:].mean_ci_lower.values, name='95% CI', mode = 'lines',

                   marker = dict(size=10, color='red'),opacity = 0.3)



trace6 =go.Scatter(x=prediction_summary2.iloc[2:].index, y=prediction_summary2.iloc[2:].mean_ci_upper.values, name='95% CI', mode = 'lines',

                   marker = dict(size=10, color='red'),opacity = 0.3)

ts2['world_percentatge']=ts/ts2

future_ratio=results1.predict(start=27,end=40)/results2.predict(start=27,end=40)



ratio_l_CI=prediction_summary.iloc[2:28].mean_ci_lower/prediction_summary2.iloc[2:28].mean_ci_lower

ratio_h_CI=prediction_summary.iloc[2:28].mean_ci_upper/prediction_summary2.iloc[2:28].mean_ci_upper

future_ratio_l_CI=prediction_summary.iloc[27:].mean_ci_lower/prediction_summary2.iloc[27:].mean_ci_lower

future_ratio_h_CI=prediction_summary.iloc[27:].mean_ci_upper/prediction_summary2.iloc[27:].mean_ci_upper

layout = go.Layout(title= 'SSA GDPpc PPP as a ratio of the World average future prediction to 2030',

    xaxis = dict(ticks='', nticks=43),

    yaxis = dict(nticks=20),legend=dict(x=0.7, y=1))



trace0 =go.Scatter(x=ts2.index, y=ts2['world_percentatge'],

                   mode = 'markers',name = '1990-2017 actual ratio', marker = dict(size=12, color='coral'),opacity=0.8)

trace10 =go.Scatter(x=future_ratio.index, y=future_ratio,

                   mode = 'markers',name = 'predicted ratio', marker = dict(size=12, color='red'),opacity=0.5)

trace5 =go.Scatter(x=ratio_l_CI.index, y=ratio_l_CI.values,

                   mode = 'lines',name = '2017-2030 actual ratio 95% CI', marker = dict(size=10, color='black'),opacity = 0.8)



trace6 =go.Scatter(x=ratio_h_CI.index, y=ratio_h_CI.values,

                   mode = 'lines',name = '2017-2030 actual ratio 95% CI', marker = dict(size=10, color='black'),opacity = 0.8)

trace2 =go.Scatter(x=future_ratio_l_CI.index, y=future_ratio_l_CI.values,

                   mode = 'lines',name = '2017-2030 projected ratio 95% CI', marker = dict(size=10, color='red'),opacity = 0.5)

trace3 =go.Scatter(x=future_ratio_h_CI.index, y=future_ratio_h_CI.values,

                   mode = 'lines',name = '2017-2030 projected ratio 95% CI', marker = dict(size=10, color='red'),opacity = 0.5)

data = [trace0,trace2,trace3,trace10,trace5,trace6]

fig =go.Figure(data=data, layout=layout)

iplot(fig, filename='heatmap',show_link=False)