import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import warnings

import datetime as dt

import matplotlib.dates as mdates

warnings.filterwarnings('ignore')
gen_1=pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

gen_1.drop('PLANT_ID',1,inplace=True)

sens_1= pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

sens_1.drop('PLANT_ID',1,inplace=True)

#format datetime

gen_1['DATE_TIME']= pd.to_datetime(gen_1['DATE_TIME'],format='%d-%m-%Y %H:%M')

sens_1['DATE_TIME']= pd.to_datetime(sens_1['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')
df_gen=gen_1.groupby('DATE_TIME').sum().reset_index()

df_gen['time']=df_gen['DATE_TIME'].dt.time



fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,5))

# daily yield plot

df_gen.plot(x='DATE_TIME',y='DAILY_YIELD',color='navy',ax=ax[0])

# AC & DC power plot

df_gen.set_index('time').drop('DATE_TIME',1)[['AC_POWER','DC_POWER']].plot(style='o',ax=ax[1])



ax[0].set_title('Daily yield',)

ax[1].set_title('AC power & DC power during day hours')

ax[0].set_ylabel('kW',color='navy',fontsize=17)

plt.show()
daily_gen=df_gen.copy()

daily_gen['date']=daily_gen['DATE_TIME'].dt.date



daily_gen=daily_gen.groupby('date').sum()



fig,ax= plt.subplots(ncols=2,dpi=100,figsize=(20,5))

daily_gen['DAILY_YIELD'].plot(ax=ax[0],color='navy')

daily_gen['TOTAL_YIELD'].plot(kind='bar',ax=ax[1],color='navy')

fig.autofmt_xdate(rotation=45)

ax[0].set_title('Daily Yield')

ax[1].set_title('Total Yield')

ax[0].set_ylabel('kW',color='navy',fontsize=17)

plt.show()
df_sens=sens_1.groupby('DATE_TIME').sum().reset_index()

df_sens['time']=df_sens['DATE_TIME'].dt.time



fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,5))

# daily yield plot

df_sens.plot(x='time',y='IRRADIATION',ax=ax[0],style='o')

# AC & DC power plot

df_sens.set_index('DATE_TIME').drop('time',1)[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']].plot(ax=ax[1])



ax[0].set_title('Irradiation during day hours',)

ax[1].set_title('Ambient and Module temperature')

ax[0].set_ylabel('W/m',color='navy',fontsize=17)

ax[1].set_ylabel('°C',color='navy',fontsize=17)





plt.show()
losses=gen_1.copy()

losses['day']=losses['DATE_TIME'].dt.date

losses=losses.groupby('day').sum()

losses['losses']=losses['AC_POWER']/losses['DC_POWER']*100



losses['losses'].plot(style='o--',figsize=(17,5),label='Real Power')



plt.title('% of DC power converted in AC power',size=17)

plt.ylabel('DC power converted (%)',fontsize=14,color='red')

plt.axhline(losses['losses'].mean(),linestyle='--',color='gray',label='mean')

plt.legend()

plt.show()
sources=gen_1.copy()

sources['time']=sources['DATE_TIME'].dt.time

sources.set_index('time').groupby('SOURCE_KEY')['DC_POWER'].plot(style='o',legend=True,figsize=(20,10))

plt.title('DC Power during day for all sources',size=17)

plt.ylabel('DC POWER ( kW )',color='navy',fontsize=17)

plt.show()
dc_gen=gen_1.copy()

dc_gen['time']=dc_gen['DATE_TIME'].dt.time

dc_gen=dc_gen.groupby(['time','SOURCE_KEY'])['DC_POWER'].mean().unstack()



cmap = sns.color_palette("Spectral", n_colors=12)



fig,ax=plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,6))

dc_gen.iloc[:,0:11].plot(ax=ax[0],color=cmap)

dc_gen.iloc[:,11:22].plot(ax=ax[1],color=cmap)



ax[0].set_title('First 11 sources')

ax[0].set_ylabel('DC POWER ( kW )',fontsize=17,color='navy')

ax[1].set_title('Last 11 sources')

plt.show()
temp1_gen=gen_1.copy()



temp1_gen['time']=temp1_gen['DATE_TIME'].dt.time

temp1_gen['day']=temp1_gen['DATE_TIME'].dt.date





temp1_sens=sens_1.copy()



temp1_sens['time']=temp1_sens['DATE_TIME'].dt.time

temp1_sens['day']=temp1_sens['DATE_TIME'].dt.date



# just for columns

cols=temp1_gen.groupby(['time','day'])['DC_POWER'].mean().unstack()
ax =temp1_gen.groupby(['time','day'])['DC_POWER'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30))

temp1_gen.groupby(['time','day'])['DAILY_YIELD'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,20),style='-.',ax=ax)



i=0

for a in range(len(ax)):

    for b in range(len(ax[a])):

        ax[a,b].set_title(cols.columns[i],size=15)

        ax[a,b].legend(['DC_POWER','DAILY_YIELD'])

        i=i+1



plt.tight_layout()

plt.show()
ax= temp1_sens.groupby(['time','day'])['MODULE_TEMPERATURE'].mean().unstack().plot(subplots=True,layout=(17,2),figsize=(20,30))

temp1_sens.groupby(['time','day'])['AMBIENT_TEMPERATURE'].mean().unstack().plot(subplots=True,layout=(17,2),figsize=(20,40),style='-.',ax=ax)



i=0

for a in range(len(ax)):

    for b in range(len(ax[a])):

        ax[a,b].axhline(50)

        ax[a,b].set_title(cols.columns[i],size=15)

        ax[a,b].legend(['Module Temperature','Ambient Temperature'])

        i=i+1



plt.tight_layout()

plt.show()
worst_source=gen_1[gen_1['SOURCE_KEY']=='bvBOhCH3iADSZry']

worst_source['time']=worst_source['DATE_TIME'].dt.time

worst_source['day']=worst_source['DATE_TIME'].dt.date



ax=worst_source.groupby(['time','day'])['DC_POWER'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30))

worst_source.groupby(['time','day'])['DAILY_YIELD'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30),ax=ax,style='-.')



i=0

for a in range(len(ax)):

    for b in range(len(ax[a])):

        ax[a,b].set_title(cols.columns[i],size=15)

        ax[a,b].legend(['DC_POWER','DAILY_YIELD'])

        i=i+1



plt.tight_layout()

plt.show()
from pandas.tseries.offsets import DateOffset

! pip install pmdarima

from pmdarima.arima import auto_arima

from statsmodels.tsa.stattools import adfuller
pred_gen=gen_1.copy()

pred_gen=pred_gen.groupby('DATE_TIME').sum()

pred_gen=pred_gen['DAILY_YIELD'][-288:].reset_index()

pred_gen.set_index('DATE_TIME',inplace=True)

pred_gen.head()
result = adfuller(pred_gen['DAILY_YIELD'])

print('Augmented Dickey-Fuller Test:')

labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']



for value,label in zip(result,labels):

    print(label+' : '+str(value) )

    

if result[1] <= 0.05:

    print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")

else:

    print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
train=pred_gen[:192]

test=pred_gen[-96:]

plt.figure(figsize=(15,5))

plt.plot(train,label='Train',color='navy')

plt.plot(test,label='Test',color='darkorange')

plt.title('Last 4 days of daily yield',fontsize=17)

plt.legend()

plt.show()
arima_model = auto_arima(train,

                         start_p=0,d=1,start_q=0,

                         max_p=4,max_d=4,max_q=4,

                         start_P=0,D=1,start_Q=0,

                         max_P=1,max_D=1,max_Q=1,m=96,

                         seasonal=True,

                         error_action='warn',trace=True,

                         supress_warning=True,stepwise=True,

                         random_state=20,n_fits=1)
future_dates = [test.index[-1] + DateOffset(minutes=x) for x in range(0,2910,15) ]
prediction=pd.DataFrame(arima_model.predict(n_periods=96),index=test.index)

prediction.columns=['predicted_yield']



fig,ax= plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(17,5))

ax[0].plot(train,label='Train',color='navy')

ax[0].plot(test,label='Test',color='darkorange')

ax[0].plot(prediction,label='Prediction',color='green')

ax[0].legend()

ax[0].set_title('Forecast on test set',size=17)

ax[0].set_ylabel('kW',color='navy',fontsize=17)





f_prediction=pd.DataFrame(arima_model.predict(n_periods=194),index=future_dates)

f_prediction.columns=['predicted_yield']

ax[1].plot(pred_gen,label='Original data',color='navy')

ax[1].plot(f_prediction,label='18th & 19th June',color='green')

ax[1].legend()

ax[1].set_title('Next days forecast',size=17)

plt.show()
arima_model.summary()
from fbprophet import Prophet

pred_gen2=gen_1.copy()

pred_gen2=pred_gen2.groupby('DATE_TIME')['DAILY_YIELD'].sum().reset_index()

pred_gen2.rename(columns={'DATE_TIME':'ds','DAILY_YIELD':'y'},inplace=True)

pred_gen2.plot(x='ds',y='y',figsize=(17,5))

plt.legend('')

plt.title('DAILY_YIELD',size=17)

plt.show()
m = Prophet()

m.fit(pred_gen2)
future = [pred_gen2['ds'].iloc[-1:] + DateOffset(minutes=x) for x in range(0,2910,15) ]

time1=pd.DataFrame(future).reset_index().drop('index',1)

time1.rename(columns={3157:'ds'},inplace=True)
timeline=pd.DataFrame(pred_gen2['ds'])

fut=timeline.append(time1,ignore_index=True)

fut.tail()
forecast = m.predict(fut)
m.plot(forecast,figsize=(15,7))

plt.title('ok')

plt.legend(labels=['Original data','Prophet Forecast'])

plt.title('Prophet Forecast')

plt.show()
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

test2=pd.DataFrame(test.index)

test2.rename(columns={'DATE_TIME':'ds'},inplace=True)

test_prophet=m.predict(test2)
print('SARIMAX R2 Score: %f' % (r2_score(prediction['predicted_yield'],test['DAILY_YIELD'])))

print('Prophet R2 Score: %f' % (r2_score(test['DAILY_YIELD'],test_prophet['yhat'])))

print('-'*15)

print('SARIMAX MAE Score: %f' % (mean_absolute_error(prediction['predicted_yield'],test['DAILY_YIELD'])))

print('Prophet MAE Score: %f' % (mean_absolute_error(test['DAILY_YIELD'],test_prophet['yhat'])))

print('-'*15)

print('SARIMAX RMSE Score: %f' % (mean_squared_error(prediction['predicted_yield'],test['DAILY_YIELD'],squared=False)))

print('Prophet RMSE Score: %f' % (mean_squared_error(test['DAILY_YIELD'],test_prophet['yhat'],squared=False)))