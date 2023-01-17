import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import urllib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import date
from scipy.signal import find_peaks
df=pd.read_csv('https://raw.githubusercontent.com/sefatosunn/Covid19-Turkey/master/Covid19-Turkey.csv' , index_col=0).fillna(0)
df=df.reset_index()
df["Date"]=pd.to_datetime(df["Date"])
df["Date"]=df["Date"].dt.strftime("%m/%d/%y")
prevents_turkey={'03/01/2020': 'uçuş0',
          '03/03/2020':'Karantina',
          '03/12/2020':'okullar tatili',     
          '03/14/2020':'ilk vaka',
          '03/16/2020': 'uçuş1',
          '03/17/2020':'ibadet',
          '03/21/2020': 'uçuş3',      
          '03/23/2020': '65+',
          '03/24/2020':'personel',
          '04/03/2020':'20',
          '04/10/2020':'marketi',
          '04/11/2020':'çıkmayasağı',
          '04/12/2020':'çıkmayasağı',
          '04/13/2020':'yasakbitişi',
          '04/18/2020': 'çıkmayasağı2'}

prevents_turkey
pt=pd.DataFrame.from_dict(prevents_turkey, orient='index')
pt=pt.reset_index()
pt.columns=['Date', 'action']
pt['Date']=pd.to_datetime(pt['Date'])
df['Date']=pd.to_datetime(df['Date'])
pt['Date']=pt['Date'].dt.strftime('%m/%d/%Y')
df['Date']=df['Date'].dt.strftime('%m/%d/%Y')
today = date.today()
tasks=[dict(Task= 'uçuş0', Start='2020-03-01', Finish=today),
 dict(Task='Karantina', Start= '2020-03-03', Finish=today),
 dict(Task= 'okullar tatili', Start= '2020-03-12', Finish=today),
 dict(Task= 'uçuş1', Start='2020-03-16', Finish=today),
 dict(Task= 'ibadet', Start= '2020-03-17', Finish='2020-05-29'),
 dict(Task= 'uçuş3', Start='2020-03-21', Finish=today),
 dict(Task= '65+', Start= '2020-03-23', Finish=today),
 dict(Task= 'personel', Start='2020-03-24', Finish=today),
 dict(Task='20', Start= '2020-04-03', Finish=today),
 dict(Task= 'çıkmayasağı', Start='2020-04-11', Finish='2020-04-13'),
 dict(Task='çıkmayasağı', Start= '2020-04-18', Finish='2020-04-20'),
 dict(Task='çıkmayasağı', Start= '2020-04-23', Finish='2020-04-26'),
 dict(Task='çıkmayasağı', Start= '2020-05-01', Finish='2020-05-03'),
 dict(Task='çıkmayasağı', Start= '2020-05-09', Finish='2020-05-10')]
fig = ff.create_gantt(tasks)
fig.show()
df_mobility=pd.read_csv('https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/apple_reports/apple_mobility_report.csv')
df_mobility=df_mobility.where(df_mobility['country']=='Turkey').dropna(how='all')
df_mobility=df_mobility.where(df_mobility['subregion_and_city']=='Total').dropna(how='all')
df_mobility=df_mobility.drop(df_mobility[['country','geo_type', 'subregion_and_city', 'transit', 'sub-region']], axis=1)
df_mobility.columns=['Date', 'driving', 'walking']
df_mobility["Date"]=pd.to_datetime(df_mobility["Date"])
df_mobility["Date"]=df_mobility["Date"].dt.strftime("%m/%d/%y")
df_mobility
df_mobility=df_mobility.set_index('Date')
df=df.set_index('Date')
#from plotly.subplots import make_subplots
fig = go.Figure()
#fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df_mobility.index, y=df_mobility['driving'] ,mode = 'lines+text',
                        name='Araç Trafiği')),
fig.add_trace(go.Scatter(x=df_mobility.index, y=df_mobility['walking'] ,mode = 'lines+text',
                        name='Yaya Trafiği'))
fig.update_layout(
    title_text="Türkiyede mobilite verisi",
    width=1000,
)
fig.update_traces(textposition='bottom right', textfont=dict(
        family="sans serif",
        size=18,
        color="Red"
    ))
fig.show()
df1=pd.merge(df, df_mobility, on='Date', how='outer')
df1=df1.sort_values(by=['Date'])
df1=df1.dropna()
df1=df1.reset_index()
df1['Active Case Rate']=(df1['Active Cases']/df1['Total Cases'])*100
from plotly.subplots import make_subplots
fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df1.Date, y=df1['Case incrase rate %'],   mode = 'lines',
                        name='Vaka Yükseliş Oranı', yaxis="y2"))
fig.add_trace(go.Scatter(x=df1.Date, y=df1['Active Case Rate'], mode = 'lines',
                        name='Aktif  Vaka Oranı', yaxis="y2"))
fig.update_layout(
    title_text="Vaka Değişim oranı ve Aktif vaka oranı karşılaştırması",
    width=1000,
)
fig.update_traces(textposition='top center')
fig.show()
time_series = df1['Daily Cases'] #.pct_change()

indices = find_peaks(time_series, threshold=300)[0]

fig = go.Figure()
fig.add_trace(go.Scatter(
    y=time_series,
    mode='lines',
    name='Tepe Noktası'
))

fig.add_trace(go.Scatter(
    x=indices,
    y=[time_series[j] for j in indices],
    mode='markers',
    marker=dict(
        size=8,
        color='red',
        symbol='cross'
    ),
    name='Günlük Vaka rakamlarında bulunan tepe noktaları'
))

fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=np.log(df['Case incrase rate %']),
                    mode='lines+markers',
                    name='Case incrase rate % logaritmik'))
fig.add_trace(go.Scatter(x=df.index, y=df['Daily(Cases/Test) %'],
                    mode='lines+markers',
                    name='Daily(Cases/Test) %'))
fig.add_trace(go.Scatter(x=df.index, y=df['(Death / Active Cases) %'],
                    mode='lines+markers',
                    name='Death / Active Cases) %'))
fig.add_trace(go.Scatter(x=df.index, y=np.log(df['(Recovered / Active Cases) %']),
                    mode='lines+markers',
                    name='(Recovered / Active Cases) % logaritmik'))

fig.show()
fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df.index, y=df['Total Recovered'].rolling(window=7).mean(),
                    mode='lines+markers',
                    name='Total Recovered'))
fig.add_trace(go.Scatter(x=df.index, y=df['Total Deaths'].rolling(window=7).mean(),
                    mode='lines+markers',
                    name='Total Deaths'))
fig.add_trace(go.Scatter(x=df.index, y=df['Active Cases'].rolling(window=7).mean(),
                    mode='lines+markers',
                    name='Active Cases', yaxis="y2"))
fig.show()
df1['all_case']=((df1['Active Cases']/df1['Total Cases'])*100).pct_change()
fig = px.line(df1, x=df1.Date, y='all_case', title='Aktif vakaların toplam vakalardaki oranın değişim grafiği')
fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=df1.Date, y=df1['Intubated Cases'],
                    mode='lines+markers',
                    name='Intubated Cases'))
fig.add_trace(go.Scatter(x=df1.Date, y=df1['Total Intensive Care'],
                    mode='lines+markers',
                    name='Total Intensive Care'))
fig.add_trace(go.Scatter(x=df1.Date, y=df1['Total Deaths'],
                    mode='lines+markers',
                    name='Total Deaths'))
fig.show()
df1['negativity']=(df1['Daily Cases']/df1['Daily Test Cases'])*100
fig = px.line(df1, x=df1.index, y='negativity', title='Günlük testlerde pozitif çıkan hasta oranı')
fig.show()
df1['day']=[x for x, i in enumerate(df1.index)]
population=8.2e7
infection_period=7
Contact_rate=4
df1['suspectile']=population-df['Daily Test Cases'].cumsum()
df1['Infected premises']=df1['Daily Cases'].shift(1)
df1=df1.replace(np.inf, 0)
df1=df1.fillna(0)
df1.columns
from statsmodels.tsa.vector_ar.vecm import coint_johansen
def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(df1[['Daily Cases', 'Daily Test Cases', 'Total Deaths', 'driving', 'walking']])    