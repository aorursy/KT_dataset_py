import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
df=pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv')
df['Date']=pd.to_datetime(df['Date'])
df=df.drop(['Source URL', 'Source label', 'Notes','Daily change in cumulative total',
       'Cumulative total per thousand',
       'Daily change in cumulative total per thousand',
       '3-day rolling mean daily change',
       '3-day rolling mean daily change per thousand'], axis=1)
df['Country']=df['Entity']
df=df.drop(['Entity'], axis=1)
df['Country']=df['Country'].str.split('-')
country=[]
for i, j in df.Country:
    country.append(i)
df['country']=country 
df=df.drop(['Country'], axis=1)
df=df.fillna(0)
df_all=pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv').fillna(0)
df_all['Country']=df_all['Country/Region']
df_all['Date']=pd.to_datetime(df_all['Date'])
df_all=df_all.drop(['Country/Region'], axis=1)
df_all=df_all.groupby(['Country', 'Date'])['Confirmed', 'Recovered', 'Deaths'].sum()
df_all=df_all.reset_index()
df_italy_test=df.where(df['country']=='Italy ').dropna()
df_usa_test=df.where(df['country']=='United States ').dropna()
df_japan_test=df.where(df['country']=='Japan ').dropna()
df_germ_test=df.where(df['country']=='Germany ').dropna()
df_spain_test=df.where(df['country']=='Spain ').dropna()
df_french_test=df.where(df['country']=='France ').dropna()
df_uk_test=df.where(df['country']=='United Kingdom ').dropna()
df_korea_test=df.where(df['country']=='South Korea ').dropna(how='all')
df_singapur_test=df.where(df['country']=='Singapore ').dropna()
df_italy=df_all.where(df_all['Country']=='Italy').dropna()
df_usa=df_all.where(df_all['Country']=='US').dropna(how='all')
df_china=df_all.where(df_all['Country']=='China').dropna(how='all')
df_germ=df_all.where(df_all['Country']=='Germany').dropna()
df_spain=df_all.where(df_all['Country']=='Spain').dropna()
df_french=df_all.where(df_all['Country']=='France').dropna()
df_japan=df_all.where(df_all['Country']=='Japan').dropna()
df_uk=df_all.where(df_all['Country']=='United Kingdom').dropna()
df_korea=df_all.where(df_all['Country']=='Korea, South').dropna(how='all')
df_singapur=df_all.where(df_all['Country']=='Singapore').dropna(how='all')
df_mobility=pd.read_csv('https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/apple_reports/apple_mobility_report.csv')
df_mobility_italy=df_mobility.where(df_mobility['country']=='Italy').dropna(how='all')
df_mobility_italy=df_mobility_italy.where(df_mobility_italy['subregion_and_city']=='Total').dropna(how='all')
df_mobility_italy=df_mobility_italy.drop(df_mobility_italy[['country','geo_type', 'subregion_and_city', 'transit']], axis=1)
df_mobility_italy.columns=['Date', 'driving', 'walking']

df_mobility_UK=df_mobility.where(df_mobility['country']=='UK').dropna(how='all')
df_mobility_UK=df_mobility_UK.where(df_mobility_UK['subregion_and_city']=='Total').dropna(how='all')
df_mobility_UK=df_mobility_UK.drop(df_mobility_UK[['country','geo_type', 'subregion_and_city', 'transit']], axis=1)
df_mobility_UK.columns=['Date', 'driving', 'walking']

df_mobility_germ=df_mobility.where(df_mobility['country']=='Germany').dropna(how='all')
df_mobility_germ=df_mobility_germ.where(df_mobility_germ['subregion_and_city']=='Total').dropna(how='all')
df_mobility_germ=df_mobility_germ.drop(df_mobility_germ[['country','geo_type', 'subregion_and_city', 'transit']], axis=1)
df_mobility_germ.columns=['Date', 'driving', 'walking']

df_mobility_spain=df_mobility.where(df_mobility['country']=='Spain').dropna(how='all')
df_mobility_spain=df_mobility_spain.where(df_mobility_spain['subregion_and_city']=='Total').dropna(how='all')
df_mobility_spain=df_mobility_spain.drop(df_mobility_spain[['country','geo_type', 'subregion_and_city', 'transit']], axis=1)
df_mobility_spain.columns=['Date', 'driving', 'walking']

df_mobility_japan=df_mobility.where(df_mobility['country']=='Japan').dropna(how='all')
df_mobility_japan=df_mobility_japan.where(df_mobility_japan['subregion_and_city']=='Total').dropna(how='all')
df_mobility_japan=df_mobility_japan.drop(df_mobility_japan[['country','geo_type', 'subregion_and_city', 'transit']], axis=1)
df_mobility_japan.columns=['Date', 'driving', 'walking']

df_mobility_france=df_mobility.where(df_mobility['country']=='Japan').dropna(how='all')
df_mobility_france=df_mobility_france.where(df_mobility_france['subregion_and_city']=='Total').dropna(how='all')
df_mobility_france=df_mobility_france.drop(df_mobility_france[['country','geo_type', 'subregion_and_city', 'transit']], axis=1)
df_mobility_france.columns=['Date', 'driving', 'walking']

df_mobility_usa=df_mobility.where(df_mobility['country']=='United States').dropna(how='all')
df_mobility_usa=df_mobility_usa.where(df_mobility_usa['subregion_and_city']=='Total').dropna(how='all')
df_mobility_usa=df_mobility_usa.drop(df_mobility_usa[['country','geo_type', 'subregion_and_city', 'transit']], axis=1)
df_mobility_usa.columns=['Date', 'driving', 'walking']

df_mobility_korea=df_mobility.where(df_mobility['country']=='Republic of Korea').dropna(how='all')
df_mobility_korea=df_mobility_korea.where(df_mobility_korea['subregion_and_city']=='Total').dropna(how='all')
df_mobility_korea=df_mobility_korea.drop(df_mobility_korea[['country','geo_type', 'subregion_and_city', 'transit']], axis=1)
df_mobility_korea.columns=['Date', 'driving', 'walking']

df_mobility_sing=df_mobility.where(df_mobility['country']=='Singapore').dropna(how='all')
df_mobility_sing=df_mobility_sing.where(df_mobility_sing['subregion_and_city']=='Total').dropna(how='all')
df_mobility_sing=df_mobility_sing.drop(df_mobility_sing[['country','geo_type', 'subregion_and_city', 'transit']], axis=1)
df_mobility_sing.columns=['Date', 'driving', 'walking']
df_italy=df_italy.set_index('Date')
df_italy_test=df_italy_test.set_index('Date')
df_mobility_italy=df_mobility_italy.set_index('Date')
df_it1 = df_italy.merge(df_italy_test, how='inner', right_index=True, left_index=False, on='Date')
#df_it=pd.concat([df_it1,df_mobility_italy], join='inner', axis=1)
df_it2=df_it1.merge(df_mobility_italy, how='inner', right_index=True, left_index=True)
df_it2
fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df_it2.index, y=df_it2['Confirmed'].pct_change(),
                    mode='lines+markers',
                    name='Confirmed Cases Percentage Change',  yaxis="y2"))
fig.add_trace(go.Scatter(x=df_it2.index, y=df_it2['driving'],
                    mode='lines+markers',
                    name='Driving Mobility'))
fig.add_trace(go.Scatter(x=df_it2.index, y=df_it2['walking'],
                    mode='lines+markers',
                    name='Walking Mobility'))

fig.show()
df_uk=df_uk.set_index('Date')
df_uk_test=df_uk_test.set_index('Date')
df_mobility_UK=df_mobility_UK.set_index('Date')
df_uk1 = df_uk.merge(df_uk_test, how='inner', right_index=True, left_index=False, on='Date')
#df_it=pd.concat([df_it1,df_mobility_italy], join='inner', axis=1)
df_uk2=df_uk1.merge(df_mobility_UK, how='inner', right_index=True, left_index=True)
df_uk2
fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df_uk2.index, y=df_uk2['Confirmed'].pct_change(),
                    mode='lines+markers',
                    name='Confirmed Cases Percentage Change',  yaxis="y2"))
fig.add_trace(go.Scatter(x=df_uk2.index, y=df_uk2['driving'],
                    mode='lines+markers',
                    name='Driving Mobility'))
fig.add_trace(go.Scatter(x=df_uk2.index, y=df_uk2['walking'],
                    mode='lines+markers',
                    name='Walking Mobility'))

fig.show()
df_korea=df_korea.set_index('Date')
df_korea_test=df_korea_test.set_index('Date')
df_mobility_korea=df_mobility_korea.set_index('Date')
df_kr1 = df_korea.merge(df_korea_test, how='inner', right_index=True, left_index=False, on='Date')
#df_it=pd.concat([df_it1,df_mobility_italy], join='inner', axis=1)
df_kr2=df_kr1.merge(df_mobility_korea, how='inner', right_index=True, left_index=True)
df_kr2
fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df_kr2.index, y=df_kr2['Confirmed'].pct_change(),
                    mode='lines+markers',
                    name='Confirmed Cases Percentage Change',  yaxis="y2"))
fig.add_trace(go.Scatter(x=df_kr2.index, y=df_kr2['driving'],
                    mode='lines+markers',
                    name='Driving Mobility'))
fig.add_trace(go.Scatter(x=df_kr2.index, y=df_kr2['walking'],
                    mode='lines+markers',
                    name='Walking Mobility'))

fig.show()
df_usa=df_usa.set_index('Date')
df_usa_test=df_usa_test.set_index('Date')
df_mobility_usa=df_mobility_usa.set_index('Date')
df_usa1 = df_usa.merge(df_usa_test, how='inner', right_index=True, left_index=False, on='Date')
#df_it=pd.concat([df_it1,df_mobility_italy], join='inner', axis=1)
df_usa2=df_usa1.merge(df_mobility_usa, how='inner', right_index=True, left_index=True)
df_usa2
fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df_usa2.index, y=df_usa2['Confirmed'].pct_change(),
                    mode='lines+markers',
                    name='Confirmed Cases Percentage Change',  yaxis="y2"))
fig.add_trace(go.Scatter(x=df_usa2.index, y=df_usa2['driving'],
                    mode='lines+markers',
                    name='Driving Mobility'))
fig.add_trace(go.Scatter(x=df_usa2.index, y=df_usa2['walking'],
                    mode='lines+markers',
                    name='Walking Mobility'))

fig.show()
df_japan=df_japan.set_index('Date')
df_japan_test=df_japan_test.set_index('Date')
df_mobility_japan=df_mobility_japan.set_index('Date')
df_japan1 = df_japan.merge(df_japan_test, how='inner', right_index=True, left_index=False, on='Date')
#df_it=pd.concat([df_it1,df_mobility_italy], join='inner', axis=1)
df_japan2=df_japan1.merge(df_mobility_japan, how='inner', right_index=True, left_index=True)
df_japan2
df_china['change']=df_china['Confirmed'].pct_change()*100
df_china['day']=[x for x, i in enumerate(df_china.index)]
fig = px.line(df_china, x=df_china.day, y='change', title='China')
fig.show()
df_italy['change']=df_italy['Confirmed'].pct_change()*100
df_italy['day']=[x for x, i in enumerate(df_italy.index)]
fig = px.line(df_italy, x=df_italy.day, y='change', title='Italy')
fig.show()
df_spain['change']=df_spain['Confirmed'].pct_change()*100
df_spain['day']=[x for x, i in enumerate(df_spain.index)]
fig = px.line(df_spain, x=df_spain.day, y='change', title='Spain')
fig.show()
df_usa['change']=df_usa['Confirmed'].pct_change()*100
df_usa['day']=[x for x, i in enumerate(df_usa.index)]
fig = px.line(df_usa, x=df_usa.day, y='change', title='usa')
fig.show()
df_uk['change']=df_uk['Confirmed'].pct_change()*100
df_uk['day']=[x for x, i in enumerate(df_uk.index)]
fig = px.line(df_uk, x=df_uk.day, y='change', title='UK')
fig.show()
df_japan['change']=df_japan['Confirmed'].pct_change()*100
df_japan['day']=[x for x, i in enumerate(df_japan.index)]
fig = px.line(df_japan, x=df_japan.day, y='change', title='Japan')
fig.show()
df_germ['change']=df_germ['Confirmed'].pct_change()*100
df_germ['day']=[x for x, i in enumerate(df_germ.index)]
fig = px.line(df_germ, x=df_germ.day, y='change', title='Germany')
fig.show()