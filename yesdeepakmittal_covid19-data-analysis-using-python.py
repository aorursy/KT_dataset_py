import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
%precision 2
import warnings
warnings.filterwarnings('ignore')
confirmed_color = 'navy'
recovered_color = 'green'
death_color = 'indianred'
active_color = 'purple'
confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'
                        'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
death = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'
                    'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'
                        'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
del confirmed['Lat']
del confirmed['Long']
del confirmed['Province/State']
del death['Lat']
del death['Long']
del death['Province/State']
del recovered['Province/State']
del recovered['Lat']
del recovered['Long']
confirmed_agg = confirmed.groupby('Country/Region').sum()
recovered_agg = recovered.groupby('Country/Region').sum()
death_agg = death.groupby('Country/Region').sum()

tot_confirmed = pd.DataFrame(confirmed_agg[confirmed_agg.columns[-1]])
tot_recovered = pd.DataFrame(recovered_agg[recovered_agg.columns[-1]])
tot_death = pd.DataFrame(death_agg[death_agg.columns[-1]])

tot_confirmed.rename(columns={tot_confirmed.columns[-1]:'Confirmed_tot'},inplace=True)
tot_recovered.rename(columns={tot_recovered.columns[-1]:'Recovered_tot'},inplace=True)
tot_death.rename(columns={tot_death.columns[-1]:'Death_tot'},inplace=True)
country_wise_tot = tot_confirmed.join(tot_recovered,how='inner')
country_wise_tot = country_wise_tot.join(tot_death,how='inner')
country_wise_tot['Active_tot'] = country_wise_tot.Confirmed_tot-\
                                 country_wise_tot['Death_tot']-\
                                 country_wise_tot.Recovered_tot
!pip install pycountry_convert
#Country name into continent
import pycountry_convert as pc
def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name

l = {'Burma':'Asia','Myanmar':'Asia', 'Congo (Brazzaville)':'Africa','Congo (Kinshasa)':'Africa',
    "Cote d'Ivoire":'Africa','Diamond Princess':'Cruise Ship','Holy See':'Europe','Korea, South':'Asia',
    'Kosovo':'Europe','MS Zaandam':'Cruise Ship','Taiwan*':'Asia','Timor-Leste':'Asia',
    'US':'North America','West Bank and Gaza':'Asia','Western Sahara':'Africa','Taiwan':'Asia'}
lis = []
temp = []
for country in country_wise_tot.index.unique():
    try:
        continent = country_to_continent(country)
    except:
        try:
            continent = l[country]
        except:
            continent = float('NaN')
    temp = [country,continent]
    lis.append(temp)
df = pd.DataFrame(lis,columns=['Country/Region','Continent'])
df = df.set_index('Country/Region')
country_wise_tot = country_wise_tot.join(df,how='inner')
country_wise_tot['RecoveryRate%'] = round(country_wise_tot['Recovered_tot']/country_wise_tot['Confirmed_tot']*100,2)
country_wise_tot['MortalityRate%'] = round(country_wise_tot.Death_tot/country_wise_tot.Confirmed_tot*100,2)
country_wise_tot['Active per 100 confirm'] = round(country_wise_tot.Active_tot/country_wise_tot.Confirmed_tot*100,2)
#country_wise_tot.head()
day_wise_confirmed = pd.DataFrame(confirmed_agg.sum(),columns={'Confirmed_dw'})
day_wise_recovered = pd.DataFrame(recovered_agg.sum(),columns={'Recovered_dw'})
day_wise_death = pd.DataFrame(death_agg.sum(),columns={'Death_dw'})
day_wise = day_wise_confirmed.join(day_wise_recovered,how='inner')
day_wise = day_wise.join(day_wise_death,how='inner')
day_wise['Active_dw'] = day_wise.Confirmed_dw-day_wise.Recovered_dw-day_wise.Death_dw
day_wise['7dyMnConfirmed'] = day_wise['Confirmed_dw'].rolling(7).mean().fillna(0).astype(int)
day_wise['7dyMnRecovered'] = day_wise['Recovered_dw'].rolling(7).mean().fillna(0).astype(int)
day_wise['7dyMnActive'] = day_wise['Active_dw'].rolling(7).mean().fillna(0).astype(int)
day_wise['7dyMnDeath'] = day_wise['Death_dw'].rolling(7).mean().fillna(0).astype(int)
l = []
for i in confirmed_agg.columns:
    for j in confirmed_agg.index:
        l.append([i,j,confirmed_agg.loc[j,i],recovered_agg.loc[j,i],death_agg.loc[j,i]])
        time_series = pd.DataFrame(l,columns=['Date','Country/Region','Confirmed','Recovered','Death'])                        
time_series.tail()
time_series['Active'] = time_series.Confirmed-time_series.Recovered-time_series.Death
time_series.set_index('Date',inplace=True)
#time_series.tail()
time_series_dw = time_series.join(day_wise[['Confirmed_dw','Recovered_dw','Death_dw','Active_dw']],how='inner')
time_series_dw['Confirmed%'] = round(time_series_dw.Confirmed/time_series_dw.Confirmed_dw*100,1)
time_series_dw['Recovered%'] = round(time_series_dw.Recovered/time_series_dw.Recovered_dw*100,1)
time_series_dw['Death%'] = round(time_series_dw.Death/time_series_dw.Death_dw*100,1)
time_series_dw['Active%'] = round(time_series_dw.Active/time_series_dw.Active_dw*100,1)
#time_series_dw.head()
# temp = pd.DataFrame(country_wise_tot[['Confirmed_tot','Recovered_tot','Death_tot','Active_tot']].sum()).T
# temp = temp.melt(value_vars=['Confirmed_tot','Recovered_tot','Death_tot','Active_tot'])
# fig = px.treemap(temp, path=["variable"], values="value", height=250, 
#                  color_discrete_sequence=[confirmed_color, recovered_color,active_color,death_color])
# fig.data[0].textinfo = 'label+text+value'
# fig.show()
fig = go.Figure(go.Funnel(
    x = [country_wise_tot['Confirmed_tot'].sum(),country_wise_tot['Recovered_tot'].sum(),
         country_wise_tot['Death_tot'].sum()],
    y = ["Total Confirmed", "Total Recovered",  "Total Death"],
    textposition = "inside",
    textinfo = "value",
    opacity = 0.8, 
    marker = {"color": [confirmed_color,recovered_color,death_color],
              "line": {"width": 2.5, "color": 'Black'}},
    connector = {"line": {"color": "navy", "dash": "dot", "width": 2.5}}))
fig.update_layout(
    template="simple_white",height=700,
    title={'text': "COVID19: Total cases across the globe",'x':0.5,'y':0.9,       
        'xanchor': 'center','yanchor': 'top'})
fig.show()
fig = px.choropleth(country_wise_tot, locations=country_wise_tot.index, locationmode='country names', 
                  color='Confirmed_tot', hover_name=country_wise_tot.index, 
                  title='Confirmed', hover_data=['Confirmed_tot'], color_continuous_scale='Blues')
fig.show()
fig = px.choropleth(country_wise_tot, locations=country_wise_tot.index, locationmode='country names', 
                  color='Recovered_tot', hover_name=country_wise_tot.index, 
                  title='Recovered', hover_data=['Recovered_tot'], color_continuous_scale='Greens')
fig.show()
fig = px.choropleth(country_wise_tot, locations=country_wise_tot.index, locationmode='country names', 
                  color='Active_tot', hover_name=country_wise_tot.index, 
                  title='Active', hover_data=['Active_tot'], color_continuous_scale='Purp')
#fig.update_layout(coloraxis_showscale=False)
fig.show()
fig = px.choropleth(country_wise_tot, locations=country_wise_tot.index, locationmode='country names', 
                  color='Death_tot', hover_name=country_wise_tot.index, 
                  title='Death', hover_data=['Death_tot'], color_continuous_scale='Reds')
fig.show()
fig = px.bar(day_wise, x=day_wise.index, y='Confirmed_dw',
             color_discrete_sequence=[confirmed_color],template='simple_white')
fig.add_scatter(x=day_wise.index,y=day_wise['7dyMnConfirmed'],name='7 day mean Confirmed',
                marker={'color': 'red','opacity': 0.6,'colorscale': 'Viridis'},)
fig.update_layout(title='Confirmed', xaxis_title="Date", 
                  yaxis_title="No. of Confirmed Cases")
fig.show()
fig = px.bar(day_wise, x=day_wise.index, y='Recovered_dw',
             color_discrete_sequence=[recovered_color],template='simple_white')
fig.add_scatter(x=day_wise.index,y=day_wise['7dyMnRecovered'],name='7 day mean Recovered',
                marker={'color': 'red','opacity': 0.6,'colorscale': 'Viridis'},)
fig.update_layout(title='Recovered', xaxis_title="Date",
                  yaxis_title="No. of Recovered Cases")
fig.show()
fig = px.bar(day_wise, x=day_wise.index, y='Active_dw',
             color_discrete_sequence=[active_color],template='simple_white')
fig.add_scatter(x=day_wise.index,y=day_wise['7dyMnActive'],name='7 day mean Active',
                marker={'color': 'red','opacity': 0.6,'colorscale': 'Viridis'},)
fig.update_layout(title='Active', xaxis_title="Date", yaxis_title="No. of Active Cases")
fig.show()
fig = px.bar(day_wise, x=day_wise.index, y='Death_dw',
             color_discrete_sequence=[death_color],template='simple_white')
fig.add_scatter(x=day_wise.index,y=day_wise['7dyMnDeath'],name='7 day mean Death',
                marker={'color': 'black','opacity': 0.6,'colorscale': 'Viridis'},)
fig.update_layout(title='Death', xaxis_title="Date",
                  yaxis_title="No. of Death Cases")
fig.show()
fig = px.line(template='simple_white')
fig.add_scatter(x=day_wise.index,y=day_wise['Confirmed_dw'],name='Confirmed',
               marker={'color': confirmed_color,'opacity': 0.6,'colorscale': 'Viridis'})
fig.add_scatter(x= day_wise.index,y=day_wise['Recovered_dw'],name='Recovered',
                marker={'color': recovered_color,'opacity': 0.6,'colorscale': 'Viridis'},)
fig.add_scatter(x=day_wise.index,y=day_wise.Active_dw,name='Active',
                marker={'color': active_color,'opacity': 0.6,'colorscale': 'Viridis'})
fig.add_scatter(x=day_wise.index,y=day_wise.Death_dw,name='Death',
                marker={'color': death_color,'opacity': 0.6,'colorscale': 'Viridis'})
fig.update_layout(title='Day Wise Analysis', xaxis_title="Date", 
                  yaxis_title="No. of Cases")
fig.show()

fig = px.bar(time_series_dw, x=pd.to_datetime(time_series_dw.index), y='Confirmed%', color='Country/Region', 
             range_y=(0, 100), title='Confirmed', 
             color_discrete_sequence=px.colors.qualitative.Plotly)
fig.show()
fig = px.bar(time_series_dw, x=pd.to_datetime(time_series_dw.index), y='Recovered%', color='Country/Region', 
             range_y=(0, 100), title='Recovered', 
             color_discrete_sequence=px.colors.qualitative.Plotly)
fig.show()
fig = px.bar(time_series_dw, x=pd.to_datetime(time_series_dw.index), y='Active%', color='Country/Region', 
             range_y=(0, 100), title='Active', 
             color_discrete_sequence=px.colors.qualitative.Plotly)
fig.show()
country_wise_tot.sort_values('Confirmed_tot', ascending= False).head(21)#['Death_tot'].max() max death cases
temp = time_series_dw.loc['7/17/20']  #percentage of cases in any country on any single day
temp[temp['Country/Region']=='China']
fig = px.bar(time_series_dw, x=pd.to_datetime(time_series_dw.index), y='Death%', color='Country/Region', 
             range_y=(0, 100), title='Death', 
             color_discrete_sequence=px.colors.qualitative.Plotly)
fig.show()
country_wise_tot.sort_values('Confirmed_tot', ascending= False).head(21).fillna(0).style\
                        .background_gradient(cmap='Blues',subset=["Confirmed_tot"])\
                        .background_gradient(cmap='Greens',subset=["Recovered_tot"])\
                        .background_gradient(cmap='Reds',subset=["Death_tot"])\
                        .background_gradient(cmap='RdPu',subset=["Active_tot"])\
                        .background_gradient(cmap='Greens',subset=["RecoveryRate%"])\
                        .background_gradient(cmap='Reds',subset=["MortalityRate%"])\
                        .background_gradient(cmap= 'RdPu',subset=["Active per 100 confirm"])\

temp = country_wise_tot.sort_values('Confirmed_tot').tail(21)
fig = go.Figure(data=[
    go.Bar(name='Death', y=temp.head(21).index, x=temp['Death_tot'].head(21),orientation='h',marker_color=death_color),
    go.Bar(name='Recovered', y=temp.head(21).index, x=temp['Recovered_tot'].head(21),orientation='h',marker_color=recovered_color),
    go.Bar(name='Confirmed', y=temp.head(21).index, x=temp['Confirmed_tot'].head(21),orientation='h',marker_color=confirmed_color)
])
fig.update_layout(barmode='stack',title='Top21 Confirmed/Recovered/Death Stacked', xaxis_title="Cases", yaxis_title="Country", 
                      yaxis_categoryorder = 'total ascending',
                      uniformtext_minsize=8, uniformtext_mode='hide',template='simple_white')
fig.show()
fig = px.bar(country_wise_tot.sort_values('Active_tot').tail(21), 
                 x='Active_tot', y=country_wise_tot.sort_values('Active_tot').tail(21).index, 
                 color='Continent',color_continuous_scale='Blues',
                 text='Active_tot', orientation='h', 
                 color_discrete_sequence = px.colors.qualitative.T10, 
                 template='simple_white')
fig.update_layout(title='Top21 Active', xaxis_title="", yaxis_title="", 
                      yaxis_categoryorder = 'total ascending',
                      uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(y=temp['MortalityRate%'], x=temp.index,
                    mode='lines+markers',
                    name='Death/100 Confirm',marker_color=death_color))
fig.add_trace(go.Scatter(y=temp['RecoveryRate%'], x=temp.index,
                    mode='lines+markers',
                    name='Recovered/100 Confirm',marker_color=recovered_color))
fig.add_trace(go.Scatter(y=temp['Active per 100 confirm'], x=temp.index,
                    mode='lines+markers', name='Active/100 Confirm',marker_color=active_color))
fig.update_layout(height=700, title_text="Top21 Cases per 100 Confirmed",template='simple_white')
fig.show()
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected = True)
top21=country_wise_tot.sort_values('Confirmed_tot').tail(21).index
fig = make_subplots(rows=7, cols=3, shared_xaxes=False)
n = 0
for row in range(1,8):
    for col in range(1,4):
        fig.add_trace(go.Bar(x = pd.to_datetime(time_series_dw.index.unique()),
        y = time_series_dw.loc[time_series_dw['Country/Region']==top21[n], 'Confirmed'],name=top21[n]),
              row, col)
        n+=1
fig.update_layout(height=2800, title_text="Confirmed cases of top21",template='simple_white')
fig.show()
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected = True)
top21=country_wise_tot.sort_values('Confirmed_tot').tail(21).index
fig = make_subplots(rows=7, cols=3, shared_xaxes=False)
n = 0
for row in range(1,8):
    for col in range(1,4):
        fig.add_trace(go.Bar(x = pd.to_datetime(time_series_dw.index.unique()),
        y = time_series_dw.loc[time_series_dw['Country/Region']==top21[n], 'Recovered'],name=top21[n]),
              row, col)
        n+=1
fig.update_layout(height=2800, title_text="Recovered cases of top21",template='simple_white')
fig.show()
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected = True)
top21=country_wise_tot.sort_values('Confirmed_tot').tail(21).index
fig = make_subplots(rows=7, cols=3, shared_xaxes=False)
n = 0
for row in range(1,8):
    for col in range(1,4):
        fig.add_trace(go.Bar(x = pd.to_datetime(time_series_dw.index.unique()),
        y = time_series_dw.loc[time_series_dw['Country/Region']==top21[n], 'Death'],name=top21[n]),
              row, col)
        n+=1
fig.update_layout(height=2800, title_text="Death cases of top21",template='simple_white')
fig.show()
top21_confirmed_new = pd.DataFrame(confirmed_agg.loc[top21[0]].diff())
top21_recovered_new = pd.DataFrame(recovered_agg.loc[top21[0]].diff())
top21_death_new = pd.DataFrame(death_agg.loc[top21[0]].diff())
for i,country in enumerate(top21):
    top21_confirmed_new[country] = confirmed_agg.loc[country].diff()
    top21_recovered_new[country] = recovered_agg.loc[country].diff()
    top21_death_new[country] = death_agg.loc[country].diff()
#top21_death_new
fig = go.Figure()
for i in top21_confirmed_new:
    fig.add_trace(go.Scatter(y=top21_confirmed_new[i], x=top21_confirmed_new.index,
                    mode='lines+markers',
                    name=i))
fig.update_layout(height=700, title_text="Top21 New Number of Confirmed Cases corresponding to previous day",template='simple_white')
fig.show()
fig = go.Figure()
for i in top21_recovered_new:
    fig.add_trace(go.Scatter(y=top21_recovered_new[i], x=top21_recovered_new.index,
                    mode='lines+markers',
                    name=i))
fig.update_layout(height=700, title_text="Top21 New Number of "
                  "Recovered Cases corresponding to previous day",template='simple_white')
fig.show()
fig = go.Figure()
for i in top21_death_new:
    fig.add_trace(go.Scatter(y=top21_death_new[i], x=top21_death_new.index,
                    mode='lines+markers',
                    name=i))
fig.update_layout(height=700, title_text="Top21 New Number of Death Cases "
                  "corresponding to previous day",template='simple_white')
fig.show()
fig = px.sunburst(country_wise_tot, path=['Continent',country_wise_tot.index],
                    values='Confirmed_tot', color='Continent',
                    hover_data=["Confirmed_tot", "Recovered_tot",'Death_tot' ]
                    )
fig.update_layout(height=1000, title_text="Confirmed")
fig.show()
fig = px.sunburst(country_wise_tot, path=['Continent',country_wise_tot.index],
                    values='Recovered_tot', color='Continent',
                    hover_data=["Confirmed_tot", "Recovered_tot",'Death_tot' ]
                    )
fig.update_layout(height=1000, title_text="Recovered")
fig.show()
fig = px.sunburst(country_wise_tot, path=['Continent',country_wise_tot.index],
                    values='Active_tot', color='Continent',
                    hover_data=["Confirmed_tot", "Recovered_tot",'Death_tot' ]
                    )
fig.update_layout(height=1000, title_text="Active")
fig.show()
fig = px.sunburst(country_wise_tot, path=['Continent',country_wise_tot.index],
                    values='Death_tot', color='Continent',
                    hover_data=["Confirmed_tot", "Recovered_tot",'Death_tot' ]
                    )
fig.update_layout(height=1000, title_text="Death")
fig.show()
!pwd
from IPython.display import FileLink
FileLink(r'time_series.csv')
#Access any distribution of any country
fig = make_subplots(rows=7, cols=3, shared_xaxes=False)
row, col = 1,1
fig.add_trace(go.Bar(x = pd.to_datetime(time_series_dw.index.unique()),
        y = time_series_dw.loc[time_series_dw['Country/Region']=='India', 'Death']),
              row, col)
fig.update_layout(height=2800,template='simple_white')

fig.show()
