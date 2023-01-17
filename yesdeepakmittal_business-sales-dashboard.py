import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')
df=pd.read_excel('/kaggle/input/superstore/US Superstore data.xls')
df.head()
print("This dataset has {} rows and {} columns".format(df.shape[0],df.shape[1]))
df.isnull().sum()
df.info()
df[['Sales','Profit','Discount']].describe()
# from datetime import datetime
# start = datetime(2015,10,11)
# end = datetime(2016,10,11)
# df[(df['Order Date']>start) &(df['Order Date']<end)].shape
# df.drop('Row ID',inplace=True,axis=1)
df[['Order ID','Ship Mode','Customer Name','Order Date','Product Name']].head(10).style\
        .background_gradient(cmap='RdPu',subset=['Order Date'])
fig = px.sunburst(df,path=['Country','Category','Sub-Category'],
                 values='Sales',color='Category',
                 hover_data =['Sales','Quantity','Profit'])
fig.update_layout(height=1000,title_text='Product Categories & Sub-Categories')
fig.show()
temp = df[['State','City','Sales']].groupby(['State','City'])['Sales'].sum().reset_index()
fig = px.treemap(temp,path=['State','City'], values='Sales')
fig.update_layout(height=1000,title='City-wise Sales',)
                 #color_discrete_sequence = px.colors.qualitative.Plotly)
fig.data[0].textinfo = 'label+text+value'
fig.show()
d = []
for i in df["Product Name"].unique():
    d.append([i,round(df[df['Product Name'] == i]['Sales'].sum(),2)])
temp = pd.DataFrame(d,columns=['Product Name','Sales'])
# temp.reset_index(inplace=True)
# del temp['index']
temp.sort_values('Sales',ascending=False).head().fillna(0).style\
        .background_gradient(cmap='Greens',subset=['Sales'])
df['Cost'] = df['Sales'] - df['Profit']
df['Profit%'] = df['Profit']/df['Cost']*100
df[['Product Name','Profit%']].sort_values('Profit%',ascending=False).head().fillna(0).style\
        .background_gradient(cmap='Greens',subset=['Profit%'])
d = []
for i in df['Sub-Category'].unique():
    sales = round(df[df['Sub-Category']==i]['Sales'].sum(),2)
    profit = round(df[df['Sub-Category']==i]['Profit'].sum(),2)
    d.append([i,sales,profit])
temp = pd.DataFrame(d,columns=['Sub-Category','Sales','Profit'])
temp = temp.sort_values('Sales',ascending=True)

fig = go.Figure(data=[go.Bar(name='Sales',x=temp['Sales'],y=temp['Sub-Category'],orientation='h',marker_color = 'green'),
                      go.Bar(name='Profit',x=temp['Profit'],y=temp['Sub-Category'],orientation='h',marker_color = 'navy')])
fig.update_layout(template='simple_white',title='Sales & Profit of each Sub-Category',height=700) #barmode='stack'
fig.show()
# df[(df['Region']=='South') & (df['Sub-Category']=='Binders')]
fig = go.Figure(data=[go.Bar(name=region,x=df['Sub-Category'],y=df[df['Region']==region]['Sales'],marker_color=color) for region,color in zip(df.Region.unique(),['red','navy','green','brown'])])
fig.update_layout(barmode='group',template='simple_white',title='Region-wise Sub-Category products Sales')
fig.show()
temp = df['Segment'].unique()
fig = go.Figure(data=go.Bar(x=temp,y=[df[df['Segment']==i]['Sales'].sum() for i in temp]))
fig.update_traces(marker_color='rgb(171,241,255)', marker_line_color='rgb(12,0,335)',
                  marker_line_width=2, opacity=0.6)
fig.update_layout(template='simple_white',title='Sales of each Segment')
fig.show()
temp = df.sort_values('Profit',ascending=False)[['Sales','Profit','Customer Name']].head(10)
fig = go.Figure(data=[go.Bar(name='Profit',x=temp['Profit'],y=temp['Customer Name'],orientation='h',marker_color='navy'),
                      go.Bar(name='Sales',x=temp['Sales'],y=temp['Customer Name'],orientation='h',marker_color='green')])
fig.update_layout(template='simple_white',title='Sales & Profit of 10 Best Customers',barmode='stack',
                 yaxis_categoryorder = 'total ascending')
fig.show()
df['Customer Name'].value_counts().nlargest(10)
df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days
df['Week'] = df['Order Date'].dt.week
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
week14 = df[df['Year']==2014][['Week','Sales']].groupby(['Week']).sum()
week15 = df[df['Year']==2015][['Week','Sales']].groupby(['Week']).sum()
week16 = df[df['Year']==2016][['Week','Sales']].groupby(['Week']).sum()
week17 = df[df['Year']==2017][['Week','Sales']].groupby(['Week']).sum()
month14 = df[df['Year']==2014][['Month','Sales']].groupby(['Month']).sum()
month15 = df[df['Year']==2015][['Month','Sales']].groupby(['Month']).sum()
month16 = df[df['Year']==2016][['Month','Sales']].groupby(['Month']).sum()
month17 = df[df['Year']==2017][['Month','Sales']].groupby(['Month']).sum()
week14.rename(columns={'Sales':'Sales14'},inplace=True)
week15.rename(columns={'Sales':'Sales15'},inplace=True)
week16.rename(columns={'Sales':'Sales16'},inplace=True)
week17.rename(columns={'Sales':'Sales17'},inplace=True)
month14.rename(columns={'Sales':'Sales14'},inplace=True)
month15.rename(columns={'Sales':'Sales15'},inplace=True)
month16.rename(columns={'Sales':'Sales16'},inplace=True)
month17.rename(columns={'Sales':'Sales17'},inplace=True)
week = week14.join(week15,on='Week').join(week16,on='Week').join(week17,on='Week')
month = month14.join(month15,on='Month').join(month16,on='Month').join(month17,on='Month')
month.head()
fig = make_subplots(rows=4, cols=1,shared_xaxes=True)
fig.add_trace(go.Scatter(x=week.index, y=week['Sales14'],mode='lines+markers',
                         name='2014',marker_color='blue'), row=1, col=1)
fig.add_trace(go.Scatter(x=week.index, y=week['Sales15'],mode='lines+markers',
                         name='2015',marker_color='green'), row=2, col=1)
fig.add_trace(go.Scatter(x=week.index, y=week['Sales16'],mode='lines+markers',
                         name='2016',marker_color='red'), row=3, col=1)
fig.add_trace(go.Scatter(x=week.index, y=week['Sales17'],mode='lines+markers',
                         name='2017',marker_color='navy'), row=4, col=1)
fig.update_layout(template='simple_white',height = 1000,
                   title='Sales per Week')
fig.show()
fig = make_subplots(rows=4, cols=1,shared_xaxes=True)
fig.add_trace(go.Scatter(x=month.index, y=month['Sales14'],mode='lines+markers',
                         name='2014',marker_color='blue'), row=1, col=1)
fig.add_trace(go.Scatter(x=month.index, y=month['Sales15'],mode='lines+markers',
                         name='2015',marker_color='green'), row=2, col=1)
fig.add_trace(go.Scatter(x=month.index, y=month['Sales16'],mode='lines+markers',
                         name='2016',marker_color='red'), row=3, col=1)
fig.add_trace(go.Scatter(x=month.index, y=month['Sales17'],mode='lines+markers',
                         name='2017',marker_color='navy'), row=4, col=1)
fig.update_layout(template='simple_white',height = 1000,
                   title='Sales per month')
fig.show()
df[['City','Profit','Profit%']].sort_values('Profit',ascending=False).head(10).style\
    .background_gradient(cmap='Greens',subset=['Profit'])\
    .background_gradient(cmap='RdPu',subset=['Profit%'])
df[['State','Profit','Profit%']].sort_values('Profit',ascending=False).head(10).style\
    .background_gradient(cmap='Greens',subset=['Profit'])\
    .background_gradient(cmap="RdPu",subset=['Profit%'])
df.groupby(['State'])['Sales'].nunique().sort_values(ascending=False)
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
geolocator = Nominatim(user_agent="my-dashboard")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
 
l = [] 
for address in df.City.unique():
    location = geocode(address)
    l.append([address,location.latitude,location.longitude])
df_city = pd.DataFrame(l,columns=['City','Latitude','Longitude'])
df_city.head()
geography = pd.merge(df,df_city,on='City',how='left')[['Profit','Sales','City','Latitude','Longitude']]
geography.head()
fig = px.scatter_mapbox(
        geography,
        title='map',
        lat="Latitude",
        lon="Longitude",
        color="Sales",
        size="Sales",
        size_max=80,
        hover_name="Profit",
        hover_data=["City", "Profit", "Sales"],
#         color_continuous_scale='Profit',
    )

fig.layout.update(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=700,
        # width=700,
        coloraxis_showscale=True,
        mapbox_style='stamen-toner',
        mapbox=dict(center=dict(lat=39.7837304, lon=-100.4458825), zoom=3),
    )

fig.data[0].update(
        hovertemplate="Sales: ₹%{customdata[2]} <br>Profit: ₹%{customdata[1]}<br>City: %{customdata[0]}"
    )
fig.show()