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
        
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
co = pd.read_csv("/kaggle/input/south-korea-covid19-daily-confirmation/worldwide_covid19.csv",parse_dates=['date'])
ct = pd.read_csv("/kaggle/input/south-korea-covid19-daily-confirmation/worldwide_country.csv")
co.shape, ct.shape, co.info(), ct.info()
print("Start date: ",co['date'].min())
print("End date: ",co['date'].max())
print("Time Period: ",co['date'].max() - co['date'].min())

co['patient'] = co['patient'].str.replace(',','').astype(int)
co['date'] = pd.to_datetime(co['date']) 
co['Year'] = co['date'].dt.year
co['Month'] = co['date'].dt.month
co['Week'] = co['date'].dt.isocalendar().week
co['Day'] = co['date'].dt.day
co['dayofweek'] = co['date'].dt.dayofweek 
co['weekdays']=co['date'].dt.strftime('%A') 
co['weekend'] = (co.date.dt.weekday >=5).astype(int) 
co['date'] = pd.to_datetime(co['date'])
co
quarterly_df = co.groupby(co.date.dt.to_period("Q"))['patient'].agg('sum').rename_axis(['date']).reset_index()
quarterly_df = quarterly_df.rename(columns={'date':'Quarter','patient':'total_patients'})

print(quarterly_df.head())
quarter_year = []
for i in quarterly_df['Quarter']:
    quarter_year.append(str(i))
    
quarterly_df['year'] = quarterly_df['Quarter'].dt.strftime('%Y') 
def plot_quarter(year,color):
    temp_quarter=[]
    for i in quarterly_df.loc[quarterly_df['year']==year]['Quarter']:
        temp_quarter.append(str(i))        
    trace=go.Bar(x=temp_quarter, y = quarterly_df.loc[quarterly_df['year']==year]['total_patients'],
           name=year,marker_color=color)
    return trace

fig = make_subplots(rows=1, cols=1,subplot_titles=("2020"))
fig.add_trace(plot_quarter('2020','cyan'),row=1,col=1)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='Covid patients registered in year 2020 - Quarter wise', title_x=0.5,showlegend=False)
fig.show()
monthly_df = co.groupby(co.date.dt.to_period("M"))['patient'].agg('sum').rename_axis(['date']).reset_index()
monthly_df = monthly_df.rename(columns={'date':'Month','patient':'total_patients'})

mon_year = []
for i in monthly_df['Month']:
    mon_year.append(str(i))
    
monthly_df['month'] = monthly_df['Month'].dt.strftime('%m') 
print(monthly_df.head())
fig = px.bar(monthly_df, x='month', y='total_patients')
fig.show()
co['date'] = pd.to_datetime(co['date'])
d = co.copy()

ds = d.drop(['country_id','dayofweek','Year','Month','Week','Day','weekdays','weekend'],axis=1)
ds['month']=ds['date'].dt.strftime('%m') 

monthly_df = ds.groupby(ds.date.dt.to_period("M"))['patient'].agg('sum').rename_axis(['date']).reset_index()
monthly_df = monthly_df.rename(columns={'date':'Month','patient':'total_patients'})
monthly_df['month']=monthly_df['Month'].dt.strftime('%m') 

fig = px.line(monthly_df, x="month", y="total_patients", title='Month Wise COVID patients across world')
fig.show()
d = co.copy()
ds = d.drop(['country_id','dayofweek','Year','Month','Week','Day','weekdays','weekend'],axis=1)

daily_df = ds.groupby(['date'])['patient'].agg(['sum']).rename_axis(['date']).reset_index() 

print("sdsdsd columns: ",daily_df.columns)
print(daily_df.head(10))

daily_year=[]

for i in daily_df['date']:
    daily_year.append(str(i))

daily_df['month']=daily_df['date'].dt.strftime('%m') 


def plot_month(month,color):
    temp_month=[]
    for i in daily_df.loc[daily_df['month']==month]['date']:
        temp_month.append(str(i))        
    trace=go.Bar(x=temp_month, y=daily_df.loc[daily_df['month']==month]['sum'], name=month, marker_color=color)
    return trace

fig = make_subplots(rows=3, cols=3,subplot_titles=("01","02","03","04","05","06","07","08","09"))
fig.add_trace(plot_month('01','purple'),row=1,col=1)
fig.add_trace(plot_month('02','limegreen'),row=1,col=2)
fig.add_trace(plot_month('03','teal'),row=1,col=3)
fig.add_trace(plot_month('04','red'),row=2,col=1)
fig.add_trace(plot_month('05','pink'),row=2,col=2)
fig.add_trace(plot_month('06','violet'),row=2,col=3)
fig.add_trace(plot_month('07','darkcyan'),row=3,col=1)
fig.add_trace(plot_month('08','blue'),row=3,col=2)
fig.add_trace(plot_month('09','skyblue'),row=3,col=3)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='Covid cases every day - Month Wise', title_x=0.5,showlegend=False)
fig.show()
co['date'] = pd.to_datetime(co['date'])
d = co.copy()

ds = d.drop(['country_id','dayofweek','Year','Month','Day','weekdays','weekend'],axis=1)

weekly_df = ds.groupby("Week")['patient'].agg('sum').rename_axis(['date']).reset_index()
weekly_df = weekly_df.rename(columns={'date':'Week','patient':'total_patients'})
print(weekly_df.dtypes)

print(weekly_df.head(10))

fig = px.line(weekly_df, x="Week", y="total_patients", title='Week Wise COVID patients across world')
fig.show()
d = co.copy()
ds = d.drop(['country_id','dayofweek','Year','Month','Week','Day','weekend'],axis=1)

weekdays_df = ds.groupby(['weekdays'])['patient'].agg(['sum']).rename_axis(['weekdays']).reset_index() 
print(weekdays_df.head(10))

weekdays_df['weekdays'] = pd.Categorical(weekdays_df['weekdays'],categories=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],ordered=True)
weekdays_df = weekdays_df.sort_values('weekdays')

fig = ff.create_table(weekdays_df, height_constant=60)
fig.add_trace(go.Scatter(
    x= weekdays_df['weekdays'], y= weekdays_df['sum'],  
    xaxis='x2', yaxis='y2',
    mode="markers",marker_size=12 ))

fig.update_layout(
    title_text = 'Weekdays COVID cases Report', title_x=0.5,
    margin = {'t':50, 'b':100}, xaxis = {'domain': [0, .5]},
    xaxis2 = {'domain': [0.6, 1.]}, yaxis2 = {'anchor': 'x2', 'title': 'Count'} )

fig.show()
d = co.copy()
ds = d.drop(['country_id','dayofweek','Year','Month','Week','Day','weekdays'],axis=1)

weekend_df = ds.groupby(['weekend'])['patient'].agg(['sum']).rename_axis(['weekend']).reset_index() 
print(weekend_df.head(10))

weekend_df = weekend_df.sort_values('weekend')

fig = ff.create_table(weekend_df, height_constant=60)
fig.add_trace(go.Scatter(
    x= weekend_df['weekend'], y= weekend_df['sum'], 
    xaxis='x2', yaxis='y2', mode="markers",marker_size=12))

fig.update_layout(
    title_text = 'Weekend COVID cases Report',title_x=0.5,
    margin = {'t':50, 'b':100}, xaxis = {'domain': [0, .5]},
    xaxis2 = {'domain': [0.6, 1.]}, yaxis2 = {'anchor': 'x2', 'title': 'sum'}
)
fig.show()
d = co.copy()
ds = d.drop(['dayofweek','Year','Month','Week','Day','weekdays'],axis=1)

weekend_df = ds.groupby(['country_id'])['patient'].agg(['sum']).reset_index() 
weekend_df = weekend_df.rename(columns={'country_id':'id','sum':'total_patients'})
print(weekend_df.head(10))
weekend_df.sort_values('total_patients', ascending=False)

merged = weekend_df.merge(ct, on=['id'])
top10 = merged[:10]
last10 = merged[-10:]
fig = go.Figure(data=[go.Bar(y=top10['country'], x=top10['total_patients'], orientation='h')],
                layout=go.Layout(title=go.layout.Title(text="Top 10 countries wth high number of COVID cases")  ) )
fig.show()

fig1 = go.Figure(data=[go.Bar(y=last10['country'], x=last10['total_patients'], orientation='h')],
                 layout=go.Layout(title=go.layout.Title(text="10 countries wth lowest number of COVID cases")  ))
fig1.show()
co['date'] = pd.to_datetime(co['date'])
d = co.copy()

ds = d.drop(['dayofweek','Year','Month','Week','Day','weekdays','weekend'],axis=1)
ds['month']=ds['date'].dt.strftime('%m') 

monthly_df = ds.groupby(['country_id',ds.date.dt.to_period("M")])['patient'].agg('sum').reset_index()

monthly_df = monthly_df.rename(columns={'date':'Month','patient':'total_patients','country_id':'id'})
monthly_df['month']=monthly_df['Month'].dt.strftime('%m') 
monthly_df = monthly_df.drop(['Month'],axis=1)

merged = monthly_df.merge(ct, on=['id'])
print(merged.head(4))

countries_top10 = merged[:90]
fig = px.line(countries_top10, x="month", y="total_patients", color='country', title="Top 10 countries month wise peak")
fig.show()