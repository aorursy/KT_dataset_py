# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
calls_dir = '../input/sf-police-calls-for-service-and-incidents/police-department-calls-for-service.csv'
calls_df = pd.read_csv(calls_dir)
calls_df.head()
incidents_dir = '../input/sf-police-calls-for-service-and-incidents/police-department-incidents.csv'
incidents_df = pd.read_csv(incidents_dir, parse_dates = [['Date', 'Time']])
incidents_df.head()
incidents_df.iloc[0:5, 0:10]
incidents_df.iloc[0:5, 11:20]
incidents_df.iloc[0:5, 21:]
incidents_df.info()
for col_name in incidents_df.columns:
    print (col_name, end="---->")
    print (sum(incidents_df[col_name].isnull()))
    print (sum(incidents_df[col_name].isnull()) / 2160953)
#radio_dir = '../input/sf-police-calls-for-service-and-incidents/Radio Codes 2016.xlsx'
#radio_df = pd.read_excel(radio_dir)
#print(radio_df)
incidents_df['Date_year'] =pd.DatetimeIndex(pd.to_datetime(incidents_df['Date_Time'])).year
incidents_df['Date_month'] =pd.DatetimeIndex(pd.to_datetime(incidents_df['Date_Time'])).month
incidents_df['Date_quarter'] = pd.PeriodIndex(pd.to_datetime(incidents_df['Date_Time']), freq='Q')
incidents_df['Date_dayofweek']=pd.DatetimeIndex(pd.to_datetime(incidents_df['Date_Time'])).dayofweek
#incidents_df.head()

incidents_df.groupby(['Date_year']).agg({'PdId':'count'})
incidents_df.query('Date_year==2015').groupby(['Date_month']).agg({'PdId':'count'})
incidents_df.query('Date_year>=2014').groupby(['DayOfWeek']).agg({'PdId':'count'})


incidents_df.columns
incidents_df.groupby(['Category']).agg({'PdId':'count'}).sort_values(by='PdId') 
                                    
incidents_df.query("Category in ('BURGLARY','WARRANTS','VANDALISM','DRUG/NARCOTIC','VEHICLE THEFT','ASSAULT','NON-CRIMINAL','OTHER OFFENSES','LARCENY/THEFT') & Date_year!=2018 ").groupby(['Category','Descript']).agg({'PdId':'count'}).reset_index() 
import plotly.express as px
import plotly as plotly
plotly.offline.init_notebook_mode (connected = True)

td=incidents_df.query("Category in ('LARCENY/THEFT') & Date_year!=2018 & Descript not in ('PETTY THEFT FROM LOCKED AUTO','GRAND THEFT FROM LOCKED AUTO')").groupby(['Category','Date_year','Descript']).agg({'PdId':'count'}).reset_index()

fig = px.line(td, x="Date_year", y="PdId", color='Descript')
fig.show()

import plotly.express as px
import plotly as plotly
plotly.offline.init_notebook_mode (connected = True)

td=incidents_df.query("Category in ('LARCENY/THEFT') & Date_year!=2018 & Descript not in ('PETTY THEFT FROM LOCKED AUTO','GRAND THEFT FROM LOCKED AUTO')").groupby(['Category','Date_year']).agg({'PdId':'count'}).reset_index()

fig = px.line(td, x="Date_year", y="PdId")
fig.show()

incidents_df.query("Category in ('LARCENY/THEFT') & Date_year>2014 ").groupby(['Category','Descript']).agg({'PdId':'count'}).reset_index() .sort_values(by='PdId')

import plotly.express as px
import plotly as plotly
plotly.offline.init_notebook_mode (connected = True)
td=incidents_df.query("Category not in ('NON-CRIMINAL') & Date_year!=2018 ").groupby(['Date_year']).agg({'PdId':'count'}).reset_index()  
fig = px.line(td, x="Date_year", y="PdId")
fig.show()
import plotly.express as px
import plotly as plotly
plotly.offline.init_notebook_mode (connected = True)
td=incidents_df.query("Category in ('BURGLARY','WARRANTS','VANDALISM','DRUG/NARCOTIC','VEHICLE THEFT','ASSAULT','NON-CRIMINAL','OTHER OFFENSES','LARCENY/THEFT') & Date_year!=2018 ").groupby(['Category','Date_year']).agg({'PdId':'count'}).reset_index()  
fig = px.line(td, x="Date_year", y="PdId", color='Category')
fig.show()
def flag_cat(series):
    if series['Descript'] in ('GRAND THEFT FROM LOCKED AUTO',
'PETTY THEFT FROM LOCKED AUTO',
'STOLEN AUTOMOBILE',
'GRAND THEFT FROM UNLOCKED AUTO',
'PETTY THEFT FROM UNLOCKED AUTO'):
        return 'auto'
    elif series['Descript'] in ('PETTY THEFT OF PROPERTY',
'GRAND THEFT OF PROPERTY',
'LOST PROPERTY, PETTY THEFT',
'LOST PROPERTY, GRAND THEFT',
'ATTEMPTED PETTY THEFT OF PROPERTY'
    ):
        return 'property'
    elif series['Category'] in ('BURGLARY','WARRANTS','VANDALISM'
                                ,'DRUG/NARCOTIC','VEHICLE THEFT',
                                'ASSAULT','NON-CRIMINAL','OTHER OFFENSES',
                                'LARCENY/THEFT'
    ):
        return series['Category']  
    else:
        return 'other'



    
incidents_df['new_category']=incidents_df.apply(flag_cat,axis=1)   

incidents_df.query("Date_year<2018").groupby(['Date_year','new_category']).agg({'PdId':'count'}).reset_index()
import plotly.express as px
import plotly as plotly
plotly.offline.init_notebook_mode (connected = True)
td=incidents_df.query("Date_year<2018").groupby(['Date_year','new_category']).agg({'PdId':'count'}).reset_index()

fig = px.line(td, x="Date_year", y="PdId", color='new_category')
fig.show()
def get_time_of_day(x):
    hour = x.hour
    
    if 4 < hour <= 10:
        return 'Morning'
    elif 11 <= hour < 16:
        return 'Afternoon'
    elif 16 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'
    
incidents_df['period_of_day'] = incidents_df['Time'].apply(get_time_of_day)
categories = ['auto','property']
cat_vol_time = incidents_df.groupby(['period_of_day','new_category']).count().loc[:,'IncidntNum']
cat_vol_time.unstack()[categories].plot.bar(legend = True)
import pkg_resources
#pkg_resources.require("numpy==1.15.4")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#most common incidents by time of day
incidents_auto = incidents_df.query("new_category=='auto'")
incidents_property = incidents_df.query("new_category=='property'")
incidents_other = incidents_df.query("new_category=='other'")

#visualize distribution of incident by time of day 
fig, (ax3,ax4) = plt.subplots(ncols=2, nrows=1,figsize=(20,10))

ax3.set_xlim(37.65, 37.85)
ax3.set_ylim(-122.53,-122.35)
ax3.set_title('Auto')
ax3.scatter(incidents_auto['Y'],incidents_auto['X'], s=0.01, alpha=1)

ax4.set_xlim(37.65, 37.85)
ax4.set_ylim(-122.53,-122.35)
ax4.set_title('Property')
ax4.scatter(incidents_property['Y'],incidents_property['X'], s=0.01, alpha=1)
incidents_df.query("new_category=='auto'").groupby(['Current Police Districts 2 2']).agg({'PdId':'count'}).reset_index()
incidents_df.query("new_category=='property'").groupby(['Current Police Districts 2 2']).agg({'PdId':'count'}).reset_index()
import plotly.express as px
import plotly as plotly
#plotly.offline.init_notebook_mode (connected = True)


t1=incidents_df.loc[ incidents_df['new_category']=='auto']
incidents_auto=t1.loc[incidents_df['Current Police Districts 2 2'] > 4.0]
incidents_property = incidents_df.query("new_category=='property'")
incidents_other = incidents_df.query("new_category=='other'")
incidents_auto = incidents_df.query("new_category=='auto'")

t1=incidents_df.loc[ incidents_df['new_category']=='auto']

incidents_auto_10=t1.loc[incidents_df['Current Police Districts 2 2'] ==10.0]
incidents_auto_8=t1.loc[incidents_df['Current Police Districts 2 2'] ==2.0]

incidents_property = incidents_df.query("new_category=='property'")
incidents_other = incidents_df.query("new_category=='other'")


#visualize distribution of incident by time of day 
fig, (ax3) = plt.subplots(ncols=1, nrows=1,figsize=(10,10))

ax3.set_xlim(37.65, 37.85)
ax3.set_ylim(-122.53,-122.35)
ax3.set_title('Auto')
ax3.scatter(incidents_auto['Y'],incidents_auto['X'], s=0.01, alpha=1,c='tab:blue')
ax3.scatter(incidents_auto_8['Y'],incidents_auto_8['X'], s=0.01, alpha=1,c='tab:orange')
ax3.scatter(incidents_auto_10['Y'],incidents_auto_10['X'], s=0.01, alpha=1,c='tab:green')
import plotly.express as px
import plotly as plotly
plotly.offline.init_notebook_mode (connected = True)
td=incidents_df.query("Category not in ('NON-CRIMINAL') & Date_year<2018").groupby(['Date_year','Descript']).agg({'PdId':'count'}).reset_index()

fig = px.line(td.query('PdId>2000 & PdId<6000'), x="Date_year", y="PdId", color='Descript')
fig.show()
import plotly.express as px
import plotly as plotly
plotly.offline.init_notebook_mode (connected = True)
td=incidents_df.query("Category in ('OTHER OFFENSES') & Date_year<2018").groupby(['Category','Date_year','Descript']).agg({'PdId':'count'}).reset_index()

fig = px.line(td.query('PdId>500'), x="Date_year", y="PdId", color='Descript')
fig.show()

import plotly.express as px
import plotly as plotly
plotly.offline.init_notebook_mode (connected = True)
td=incidents_df.query("Category in ('ASSAULT') & Date_year<2018").groupby(['Category','Date_year','Descript']).agg({'PdId':'count'}).reset_index()

fig = px.line(td.query('PdId>500'), x="Date_year", y="PdId", color='Descript')
fig.show()

import plotly.express as px
import plotly as plotly
plotly.offline.init_notebook_mode (connected = True)
td=incidents_df.query("Category in ('VANDALISM') & Date_year<2018").groupby(['Category','Date_year','Descript']).agg({'PdId':'count'}).reset_index()

fig = px.line(td.query('PdId>500'), x="Date_year", y="PdId", color='Descript')
fig.show()
import plotly.express as px
import plotly as plotly
plotly.offline.init_notebook_mode (connected = True)
td=incidents_df.query("Category in ('ROBBERY') & Date_year<2018").groupby(['Category','Date_year','Descript']).agg({'PdId':'count'}).reset_index()

fig = px.line(td.query('PdId>500'), x="Date_year", y="PdId", color='Descript')
fig.show()

import plotly.express as px
import plotly as plotly
plotly.offline.init_notebook_mode (connected = True)
td=incidents_df.query("Category in ('LARCENY/THEFT') & Date_year<2018").groupby(['Category','Date_year','Descript']).agg({'PdId':'count'}).reset_index()

fig = px.line(td.query('PdId>800'), x="Date_year", y="PdId", color='Descript')
fig.show()


import plotly.express as px
import plotly as plotly
plotly.offline.init_notebook_mode (connected = True)
td=incidents_df.query("Category in ('VEHICLE THEFT') & Date_year<2018").groupby(['Category','Date_year','Descript']).agg({'PdId':'count'}).reset_index()

fig = px.line(td.query('PdId>100'), x="Date_year", y="PdId", color='Descript')
fig.show()




import plotly.express as px
import plotly as plotly
plotly.offline.init_notebook_mode (connected = True)
td=incidents_df.query("Category in ('RECOVERED VEHICLE') & Date_year<2018").groupby(['Category','Date_year','Descript']).agg({'PdId':'count'}).reset_index()

fig = px.line(td.query('PdId>0'), x="Date_year", y="PdId", color='Descript')
fig.show()
import plotly.express as px

td=incidents_df.query("Category in ('BURGLARY','WARRANTS','VANDALISM','DRUG/NARCOTIC','VEHICLE THEFT','ASSAULT','NON-CRIMINAL','OTHER OFFENSES','LARCENY/THEFT') & Date_year>2014 & Date_year<2018 ").groupby(['Category','Date_dayofweek']).agg({'PdId':'count'}).reset_index() 
fig = px.line(td, x="Date_dayofweek", y="PdId", color='Category')
fig.show()

#The day of the week with Monday=0, Sunday=6.
#Analysis Neighborhoods
import plotly.express as px

td=incidents_df.query("Category in ('LARCENY/THEFT') & Date_year<2018 ").groupby(['PdDistrict','Date_year']).agg({'PdId':'count'}).reset_index() 
fig = px.line(td, x="Date_year", y="PdId", color='PdDistrict')

fig.update_layout(title='',
                  yaxis_zeroline=False, xaxis_zeroline=False)

fig.show()


incidents_df.info()

incidents_df.query("Category in ('LARCENY/THEFT') & Date_year>2014 & Date_year<2018 ").groupby(['Category','Descript']).agg({'PdId':'count'}).reset_index().sort_values(by='PdId') 

import plotly.express as px

td=incidents_df.query("Category in ('BURGLARY','WARRANTS','VANDALISM','DRUG/NARCOTIC','VEHICLE THEFT','ASSAULT','NON-CRIMINAL','OTHER OFFENSES','LARCENY/THEFT') & Date_year>2014 & Date_year<2018 ").groupby(['Category','DayOfWeek']).agg({'PdId':'count'}).reset_index() 
fig = px.line(td, x="DayOfWeek", y="PdId", color='Category')
fig.show()
import fbprophet

incidents_df['Time'] = pd.to_datetime(incidents_df['Date_Time'], format='%H:%M')
incidents_df['Date'] = pd.to_datetime(incidents_df['Date_Time'])
incidents_df['hour'] = incidents_df['Time'].dt.hour
incidents_df['wday'] = incidents_df['Date'].dt.dayofweek
incidents_df.head()
incidents_sort = incidents_df.sort_values('Date')
weekly = incidents_df.groupby(['new_category', pd.Grouper(key='Date',freq='W')]).count()['IncidntNum'].reset_index()
weekly_train = weekly[weekly['Date'].dt.year < 2018]
weekly_train = weekly_train.rename(columns={"Date": "ds", "IncidntNum": "y"})
weekly_vali = weekly[weekly['Date'].dt.year >= 2012]
weekly_vali = weekly_vali.rename(columns={"Date": "ds", "IncidntNum": "y"})
categories = weekly_train['new_category'].unique()
def prophet_model(df,cat):
    train = df[df['new_category'] == cat]
    in_prophet = fbprophet.Prophet(changepoint_prior_scale=0.1)
    in_prophet.fit(train)
    
    return in_prophet
model_dict = {cat: prophet_model(weekly_train, cat) for cat in categories}

cat = "auto"
in_forecast = model_dict[cat].make_future_dataframe(periods=52, freq='W')
pred = model_dict[cat].predict(in_forecast)
def get_pred(preds, date):
    return pred['trend'].loc[pred['ds'] == date]
get_pred(pred, '2018-11-25')
pred.head()
fig, ax = plt.subplots(1, 1, figsize = (15, 9))
model_dict[cat].plot(pred, xlabel = 'Date', ylabel = 'Incidents', ax=ax)


weekly_vali.head()


plot=weekly_vali.query("new_category=='auto'").merge(pred, left_on='ds', right_on='ds',

          suffixes=('_left', '_right'))

 

import plotly.graph_objects as go

fig=go.Figure(data=go.Scatter(x=plot['ds'],y=plot['y'],name="Actual"))

fig.add_trace(go.Scatter(x=plot['ds'],y=plot['yhat']  ,name="Prediction"))

fig.show()