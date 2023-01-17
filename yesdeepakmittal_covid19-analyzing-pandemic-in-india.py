import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from urllib.request import urlopen
import json
import requests
import re
import math
import warnings
warnings.filterwarnings('ignore')
confirmed_color = 'navy'
recovered_color = 'green'
death_color = 'indianred'
active_color = 'purple'
df1 = "https://api.covid19india.org/state_district_wise.json"
df2 = "https://api.covid19india.org/data.json"
def getting_data(url):
    response = requests.get(url)
    data = response.content.decode('utf-8')
    return data
df_state = json.loads(getting_data(df1))
df = json.loads(getting_data(df2))
lis = []
state_names = df_state.keys()
for state in state_names:
    district_names = df_state[state]['districtData'].keys() #Districts of Current State
    for district in district_names:
        temp = df_state[state]['districtData'][district]
        var_lis = [state,district,temp.get('confirmed'),temp.get('recovered'),
                   temp.get('active'),temp.get('deceased')]
        lis.append(var_lis)
    district_wise = pd.DataFrame(lis,columns=['State/UT','District','Confirmed',
                                              'Recovered','Active','Death'])
district_wise.head()
temp = [[i['state'],i['confirmed'],i['recovered'],i['active'],i['deaths'],
         i['lastupdatedtime'],i['deltaconfirmed'],i['deltarecovered'],
         i['deltadeaths']] for i in df['statewise']]
statewise_total = pd.DataFrame(temp,columns=['State/UT','Confirmed','Recovered',
                                          'Active','Death','LastUpdateTime',
                              'DeltaConfirmed','DeltaRecovered','DeltaDeath'])
statewise_total['Confirmed']=statewise_total['Confirmed'].astype('int')
statewise_total['Recovered']=statewise_total['Recovered'].astype('int')
statewise_total['Active']=statewise_total['Active'].astype('int')
statewise_total['Death']=statewise_total['Death'].astype('int')
statewise_total['DeltaConfirmed']=statewise_total['DeltaConfirmed'].astype('int')
statewise_total['DeltaRecovered']=statewise_total['DeltaRecovered'].astype('int')
statewise_total['DeltaDeath']=statewise_total['DeltaDeath'].astype('int')
statewise_total['RecoveryRate%'] = round(statewise_total['Recovered']/statewise_total['Confirmed']*100,2)
statewise_total['MortalityRate%'] = round(statewise_total['Death']/statewise_total['Confirmed']*100,2)
statewise_total['Active/100 Confirmed'] = round(statewise_total['Active']/statewise_total['Confirmed']*100,2)
for i,y in enumerate(statewise_total['LastUpdateTime']):
    statewise_total['LastUpdateTime'][i] = pd.to_datetime(y.split(' ')[0])
statewise_total['LastUpdateTime'] = pd.to_datetime(statewise_total['LastUpdateTime'])
statewise_total.head()
timeseries = [list(i.values()) for i in df['cases_time_series']]
timeseries = pd.DataFrame(timeseries,columns=df['cases_time_series'][0].keys())
# timeseries.rename(columns={'date':'Date'},inplace=True)
# timeseries['Date']=timeseries['Date'].replace({' January':'-01-2020',' February':'-02-2020',
#                                                ' March':'-03-2020',' April':'-04-2020',
#                                                ' May':'-05-2020',' June':'-06-2020'},regex=True)
# timeseries['Date']= pd.to_datetime(timeseries['Date'])
# timeseries.set_index('Date',inplace=True)
timeseries['dailyconfirmed'] = timeseries['dailyconfirmed'].astype('int')
timeseries['dailydeceased'] = timeseries['dailydeceased'].astype('int')
timeseries['dailyrecovered'] = timeseries['dailyrecovered'].astype('int')
timeseries['totalconfirmed'] = timeseries['totalconfirmed'].astype('int')
timeseries['totaldeceased'] = timeseries['totaldeceased'].astype('int')
timeseries['totalrecovered'] = timeseries['totalrecovered'].astype('int')
timeseries['7dyMnConfirmed'] = timeseries.totalconfirmed.rolling(7).mean().fillna(0).astype(int)
timeseries['7dyMnRecovered'] = timeseries.totalrecovered.rolling(7).mean().fillna(0).astype(int)
timeseries['7dyMnDeceased'] = timeseries.totaldeceased.rolling(7).mean().fillna(0).astype(int)
timeseries.head()
values = [list(i.values())[-5:] for i in df["tested"]]
tests = pd.DataFrame(values, columns=list(df["tested"][0].keys())[-5:])
for i,value in enumerate(tests['totalsamplestested']):
    if value=='':
      avg = math.ceil((int(tests['totalsamplestested'].iloc[i-1])+int(tests['totalsamplestested'].iloc[i+1]))/2)
      tests['totalsamplestested'].iloc[i] = avg #Taking avg from previous and next value for three missing values
tests['totalsamplestested'] = tests['totalsamplestested'].astype('int')
for i,value in enumerate(tests['testspermillion']):
    if value=='':
      avg = math.ceil((int(tests['testspermillion'].iloc[i-1])+int(tests['testspermillion'].iloc[i+1]))/2)
      tests['testspermillion'].iloc[i] = avg #Taking avg from previous and next value for three missing values
tests['testspermillion'] = tests['testspermillion'].astype('int')
#tests.head()
time_series_state = pd.read_csv('https://api.covid19india.org/csv/latest'
                                        '/state_wise_daily.csv')
del time_series_state['TT']
time_series_state = time_series_state.melt(id_vars=['Status','Date'], 
                      value_vars=time_series_state.columns[2:],
                      value_name='Census',var_name='State')
time_series_state = time_series_state.pivot_table(index=['Date', 'State'], 
                                                columns=['Status'], 
                                                values='Census')
time_series_state = time_series_state.reset_index()
given_data = json.loads(getting_data(df1))
state_names = given_data.keys()
given_lis = []
for state in state_names:
    #district_names = df_state[state]['statecode'].keys() 
    #print(a[state]['statecode'],state)
    given_lis.append([given_data[state]['statecode'],state])
given_dic = {}
for i in given_lis:
    given_dic[i[0]] = i[1]
given_dic['DD'] = 'Daman and Diu'
#code = pd.DataFrame(given_dic.items(),columns={'Code','State'})
time_series_state['State-Name'] = time_series_state['State'].map(given_dic)
time_series_state['Date'] = pd.to_datetime(time_series_state['Date'])
time_series_state.set_index('Date',inplace=True)
#time_series_state.head()
temp = pd.DataFrame(timeseries.iloc[-1].T)
temp = temp.T
temp['totalactive'] = int(temp.totalconfirmed)-int(temp.totalrecovered)-int(temp.totaldeceased)
temp = temp.melt(value_vars=['totalconfirmed','totalrecovered','totalactive','totaldeceased'])
fig = px.treemap(temp, path=["variable"], values="value", height=250, 
                 color_discrete_sequence=[confirmed_color, recovered_color,active_color,death_color])
fig.data[0].textinfo = 'label+text+value'
fig.show()
temp = statewise_total[statewise_total['State/UT']!='Total']
temp = temp[temp['State/UT']!='State Unassigned']
fig = go.Figure(data=[
    go.Bar(name='Death', y=temp['State/UT'], x=temp['Death'],orientation='h',marker_color=death_color),
    go.Bar(name='Recovered', y=temp['State/UT'], x=temp['Recovered'],orientation='h',marker_color=recovered_color),
    go.Bar(name='Confirmed', y=temp['State/UT'], x=temp['Confirmed'],orientation='h',marker_color=confirmed_color)
])
fig.update_layout(barmode='stack',title='Statewise Confirmed/Recovered/Death Stacked', xaxis_title="Cases", yaxis_title="State/UT", 
                      yaxis_categoryorder = 'total ascending', height = 1000,
                      uniformtext_minsize=8, uniformtext_mode='hide',template='simple_white')
fig.show()
temp = statewise_total[statewise_total['State/UT']!='Total']
temp = temp[temp['State/UT']!='State Unassigned']
fig = go.Figure()
fig.add_trace(go.Scatter(y=temp['Death'], x=temp['State/UT'],
                    mode='lines+markers',
                    name='Death',marker_color=death_color))
fig.add_trace(go.Scatter(y=temp['Recovered'], x=temp['State/UT'],
                    mode='lines+markers',
                    name='Recovered',marker_color=recovered_color))
fig.add_trace(go.Scatter(y=temp['Active'], x=temp['State/UT'],
                    mode='lines+markers', name='Active',marker_color=active_color))
fig.add_trace(go.Scatter(y=temp['Confirmed'], x=temp['State/UT'],
                    mode='lines+markers', name='Confirmed',marker_color=confirmed_color))
fig.update_layout(height=900,width= 1200, title_text="Statewise Cases",template='simple_white')
fig.show()
temp = statewise_total[statewise_total['State/UT']!='Total']
temp = temp[temp['State/UT']!='State Unassigned']
fig = go.Figure()
fig.add_trace(go.Scatter(y=temp['MortalityRate%'], x=temp['State/UT'],
                    mode='lines+markers',
                    name='Mortality Rate',marker_color=death_color))
fig.add_trace(go.Scatter(y=temp['RecoveryRate%'], x=temp['State/UT'],
                    mode='lines+markers',
                    name='Recovery Rate',marker_color=recovered_color))
fig.add_trace(go.Scatter(y=temp['Active/100 Confirmed'], x=temp['State/UT'],
                    mode='lines+markers', name='Active/100 Confirmed',marker_color=active_color))
fig.update_layout(height=700,width= 1200, title_text="Statewise Cases per 100 Confirmed",template='simple_white')
fig.show()
fig = px.bar(timeseries, x='date', y='totalconfirmed', color_discrete_sequence=[confirmed_color],template='simple_white')
fig.update_layout(title='Confirmed', xaxis_title="Date", yaxis_title="No. of Confirmed Cases")
fig.add_scatter(x=timeseries['date'],y=timeseries['7dyMnConfirmed'],name='7 day mean Confirmed',
                marker={'color': 'red','opacity': 0.6,'colorscale': 'Viridis'},)
fig.show()
fig = px.bar(timeseries, x='date', y='totalrecovered', 
             color_discrete_sequence=[recovered_color],template='simple_white')
fig.update_layout(title='Recovered', xaxis_title="Date", yaxis_title="No. of Recovered Cases")
fig.add_scatter(x=timeseries['date'],y=timeseries['7dyMnRecovered'],name='7 day mean Recovered',
                marker={'color': 'red','opacity': 0.6,'colorscale': 'Viridis'},)
fig.show()
fig = px.bar(timeseries, x='date', y='totaldeceased', 
             color_discrete_sequence=[death_color],template='simple_white')
fig.update_layout(title='Death', xaxis_title="Date", yaxis_title="No. of Deceased Cases")
fig.add_scatter(x=timeseries['date'],y=timeseries['7dyMnDeceased'],name='7 day mean Deceased',
                marker={'color': 'black','opacity': 0.6,'colorscale': 'Viridis'},)
fig.show()
fig = px.line(color_discrete_sequence=[confirmed_color],template='simple_white')
fig.add_scatter(x=timeseries['date'],y=timeseries['dailyconfirmed'],name='Daily Confirmed',marker={'color': confirmed_color,'opacity': 0.6,'colorscale': 'Viridis'},)
fig.add_scatter(x=timeseries['date'],y=timeseries['dailyrecovered'],name='Daily Recovered',marker={'color': recovered_color,'opacity': 0.6,'colorscale': 'Viridis'},)
fig.add_scatter(x=timeseries['date'],y=timeseries['dailydeceased'],name='Daily Death',marker={'color': death_color,'opacity': 0.6,'colorscale': 'Viridis'})
fig.update_layout(title='Day Wise Analysis', xaxis_title="Date", yaxis_title="No. of Cases")
fig.show()
timeseries['totalactive'] = timeseries.totalconfirmed-timeseries.totalrecovered-timeseries.totaldeceased
fig = px.line(color_discrete_sequence=[confirmed_color],template='simple_white')
fig.add_scatter(x=timeseries['date'],y=timeseries['totalconfirmed'],name='Total Confirmed',marker={'color': confirmed_color,'opacity': 0.6,'colorscale': 'Viridis'},)
fig.add_scatter(x=timeseries['date'],y=timeseries['totalrecovered'],name='Total Recovered',marker={'color': recovered_color,'opacity': 0.6,'colorscale': 'Viridis'},)
fig.add_scatter(x=timeseries['date'],y=timeseries['totaldeceased'],name='Total Death',marker={'color': death_color,'opacity': 0.6,'colorscale': 'Viridis'})
fig.add_scatter(x=timeseries['date'],y=timeseries['totalactive'],name='Total Active',marker={'color': active_color,'opacity': 0.6,'colorscale': 'Viridis'})
fig.update_layout(title='Total Cases', xaxis_title="Date", yaxis_title="No. of Cases")
fig.show()
temp = district_wise.sort_values('Confirmed').tail(21)
fig = go.Figure(data=[
    go.Bar(name='Death', y=temp['District'], x=temp['Death'].head(21),orientation='h',marker_color=death_color),
    go.Bar(name='Recovered', y=temp['District'], x=temp['Recovered'].head(21),orientation='h',marker_color=recovered_color),
    go.Bar(name='Confirmed', y=temp['District'], x=temp['Confirmed'].head(21),orientation='h',marker_color=confirmed_color)
])
fig.update_layout(barmode='stack',title='Top21 Confirmed/Recovered/Death Stacked', xaxis_title="Cases", yaxis_title="District", 
                      yaxis_categoryorder = 'total ascending',
                      uniformtext_minsize=8, uniformtext_mode='hide',template='simple_white')
fig.show() #Here Unknown is Delhi
district_wise = district_wise[district_wise['State/UT']!='State Unassigned']
district_wise.sort_values('Confirmed', ascending= False).head(30).fillna(0).style\
                        .background_gradient(cmap='Blues',subset=["Confirmed"])\
                        .background_gradient(cmap='Greens',subset=["Recovered"])\
                        .background_gradient(cmap='Reds',subset=["Death"])\
                        .background_gradient(cmap='RdPu',subset=["Active"])
temp = district_wise.groupby(['State/UT','District'])['Death'].sum().reset_index()
fig = px.treemap(temp, path=['State/UT','District'], values="Death", 
                 height=1000, title='Number of Deceased Cases', 
                 color_discrete_sequence = px.colors.qualitative.Plotly)
fig.data[0].textinfo = 'label+text+value'
fig.show()
fig = px.sunburst(district_wise, path=['State/UT','District'],
                    values='Confirmed', color='State/UT',
                    hover_data=["Confirmed", "Recovered",'Death' ]
                    )
fig.update_layout(height=1000, title_text="Districtwise Confirmed Cases")
fig.show()
fig = px.sunburst(district_wise, path=['State/UT','District'],
                    values='Recovered', color='State/UT',
                    hover_data=["Confirmed", "Recovered",'Death' ]
                    )
fig.update_layout(height=1000, title_text="Districtwise Recovered Cases")
fig.show()
fig = px.sunburst(district_wise, path=['State/UT','District'],
                    values='Death', color='State/UT',
                    hover_data=["Confirmed", "Recovered",'Death' ]
                    )
fig.update_layout(height=1000, title_text="Districtwise Deceased Cases")
fig.show()
temp = pd.read_csv('https://api.covid19india.org/csv/latest'
                                '/state_wise_daily.csv')
temp = temp[temp['Status']=='Confirmed']
fig = go.Figure()
# fig.add_trace(go.Scatter(y=temp['TT'], x=temp['Date'],
#                     mode='lines+markers',name='Total'))
fig.add_trace(go.Scatter(y=temp['UP'], x=temp['Date'],
                    mode='lines+markers',name='Uttar Pradesh'))
fig.add_trace(go.Scatter(y=temp['MH'], x=temp['Date'],
                    mode='lines+markers',name='Maharashtra'))
fig.add_trace(go.Scatter(y=temp['DL'], x=temp['Date'],
                    mode='lines+markers',name='Delhi'))
fig.add_trace(go.Scatter(y=temp['PB'], x=temp['Date'],
                    mode='lines+markers',name='Punjab'))
fig.add_trace(go.Scatter(y=temp['RJ'], x=temp['Date'],
                    mode='lines+markers',name='Rajasthan'))
fig.update_layout(title_text="Daywise Confirmed Cases of States",
                  template='simple_white',height=700)
fig.show()
temp = pd.read_csv('https://api.covid19india.org/csv/latest'
                                '/state_wise_daily.csv')
temp = temp[temp['Status']=='Confirmed']
temp['TTtotal'] = temp['TT'].cumsum()
temp['UPtotal'] = temp['UP'].cumsum()
temp['DLtotal'] = temp['DL'].cumsum()
temp['GJtotal'] = temp['GJ'].cumsum()
temp['PBtotal'] = temp['PB'].cumsum()
temp['RJtotal'] = temp['RJ'].cumsum()
temp['7dyMnUP'] = temp.UPtotal.rolling(7).mean().fillna(0).astype(int)
temp['7dyMnDL'] = temp.DLtotal.rolling(7).mean().fillna(0).astype(int)
temp['7dyMnGJ'] = temp.GJtotal.rolling(7).mean().fillna(0).astype(int)
temp['7dyMnPB'] = temp.PBtotal.rolling(7).mean().fillna(0).astype(int)
temp['7dyMnRJ'] = temp.RJtotal.rolling(7).mean().fillna(0).astype(int)
temp['7dyMnTT'] = temp.TTtotal.rolling(7).mean().fillna(0).astype(int)
fig1 = make_subplots(rows=3, cols=2, shared_xaxes=False)
fig1.add_trace(go.Bar(x=temp['Date'], y=temp['TTtotal'],name='Total'),1,1)
fig1.add_trace(go.Scatter(x=temp['Date'], y=temp['7dyMnTT'], name="7 day Mean Total"), row=1, col=1)
fig1.add_trace(go.Bar(x=temp['Date'], y=temp['UPtotal'],name='Uttar Pradesh'),1,2)
fig1.add_trace(go.Scatter(x=temp['Date'], y=temp['7dyMnUP'], name="7 day Mean UP"), row=1, col=2)
fig1.add_trace(go.Bar(x=temp['Date'], y=temp['DLtotal'],name='Delhi'),2,1)
fig1.add_trace(go.Scatter(x=temp['Date'], y=temp['7dyMnDL'], name="7 day Mean Delhi"), row=2, col=1)
fig1.add_trace(go.Bar(x=temp['Date'], y=temp['GJtotal'],name='Gujarat'),2,2)
fig1.add_trace(go.Scatter(x=temp['Date'], y=temp['7dyMnGJ'], name="7 day Mean Gujarat"), row=2, col=2)
fig1.add_trace(go.Bar(x=temp['Date'], y=temp['PBtotal'],name='Punjab'),3,1)
fig1.add_trace(go.Scatter(x=temp['Date'], y=temp['7dyMnPB'], name="7 day Mean Punjab"), row=3, col=1)
fig1.add_trace(go.Bar(x=temp['Date'], y=temp['RJtotal'],name='Rajasthan'),3,2)
fig1.add_trace(go.Scatter(x=temp['Date'], y=temp['7dyMnRJ'], name="7 day Mean Rajasthan"), row=3, col=2)
fig1.update_layout(template='simple_white',height = 1500,
                   title='State Confirmed Commulative Distribution')#showlegend=False,
fig1.show()

fig = make_subplots(rows=3, cols=1,shared_xaxes=True)
fig.add_trace(go.Scatter(x=tests['updatetimestamp'], y=tests['totalsamplestested'],mode='lines+markers',
                         name='Total Sample Tested',marker_color='blue'), row=1, col=1)
fig.add_trace(go.Scatter(x=tests['updatetimestamp'], y=tests['totalsamplestested'].diff(),mode='lines+markers',
                         name='Daily Tests',marker_color='green'), row=2, col=1)
fig.add_trace(go.Scatter(x=tests['updatetimestamp'], y=tests['testspermillion'],mode='lines+markers',
                         name='Tests Per Million',marker_color='red'), row=3, col=1)
fig.update_layout(template='simple_white',height = 1000,
                   title='Tests in India')
fig.show()