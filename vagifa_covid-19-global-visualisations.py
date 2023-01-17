import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
global_covid = requests.get('https://api.covid19api.com/summary')
global_df = pd.DataFrame(global_covid.json()['Global'],index=[0])
global_df.drop(['NewConfirmed','NewDeaths','NewRecovered'],axis=1,inplace=True)
fig = go.Figure(data=[go.Table(header=dict(values=list(global_df.columns)),cells=dict(values=[global_df.loc[0][0], global_df.loc[0][1],global_df.loc[0][2],]))])
fig.update_layout(title='COVID-19 Worldwide Statistics',title_x=0.5)
fig.show()
res = requests.get('https://api.thevirustracker.com/free-api?countryTotals=ALL')
df_covid = []
for j in range(1,len(res.json()['countryitems'][0])):
    df_covid.append([res.json()['countryitems'][0]['{}'.format(j)]['title'],
         res.json()['countryitems'][0]['{}'.format(j)]['code'],res.json()['countryitems'][0]['{}'.format(j)]['total_cases'],res.json()['countryitems'][0]['{}'.format(j)]['total_recovered'],res.json()['countryitems'][0]['{}'.format(j)]['total_deaths']])
df_covid = pd.DataFrame(df_covid, columns = ['Country','Code', 'TotalCases','TotalRecovered','TotalDeaths'])
df_covid.sort_values('TotalCases',ascending=False,inplace=True)
fig = go.Figure(data=[go.Table(header=dict(values=list(df_covid.columns)),
                 cells=dict(values=[df_covid['Country'], df_covid['Code'],df_covid['TotalCases'],df_covid['TotalRecovered'],df_covid['TotalDeaths']]))
                     ])
fig.update_layout(title='COVID-19 Total Cases,Total Recovered and Total Deaths per Country',title_x=0.5)
fig.show()
country_codes=df_covid.head(10)['Country']
fig = px.bar(x=country_codes,y=df_covid.head(10)['TotalCases'],labels={'x':'Country','y':'Total Cases'})
fig.update_layout(title='Top 10 Countries with the most Cases',title_x=0.5)
fig.show()
fig = go.Figure(data=[go.Scatter(
    x=df_covid.head(10)['Country'],
    y=df_covid.head(10)['TotalCases'],
    mode='markers',
    marker=dict(
        color=100+np.random.randn(500),
        size=(df_covid.head(10)['TotalCases']/12000),
        showscale=True
        )
)])
fig.update_layout(
    title='Top 10 Countries with the most Cases',
    xaxis_title="Countries",
    yaxis_title="Confirmed Cases",
    template='plotly_white',
    title_x = 0.5

)
fig.show()
df_recoveries = df_covid.sort_values('TotalRecovered',ascending=False)
fig = px.bar(x=df_recoveries.head(10)['Country'],y=df_recoveries.head(10)['TotalRecovered'],labels={'x':'Country','y':'Total Recoveries'})
fig.update_layout(title='Top 10 Countries with the most Recovered Cases',title_x=0.5)
fig.show()
fig = go.Figure(data=[go.Scatter(
    x=df_recoveries.head(10)['Country'],
    y=df_recoveries.head(10)['TotalRecovered'],
    mode='markers',
    marker=dict(
        color=100+np.random.randn(500),
        size=(df_recoveries.head(10)['TotalCases']/12000),
        showscale=True
        )
)])
fig.update_layout(
    title='Top 10 Countries with the most Recovered',
    xaxis_title="Countries",
    yaxis_title="Recovered Cases",
    template='plotly_white',
    title_x = 0.5

)
fig.show()
df_deaths = df_covid.sort_values('TotalDeaths',ascending=False)
fig = px.bar(x=df_deaths.head(10)['Country'],y=df_deaths.head(10)['TotalDeaths'],labels={'x':'Country','y':'Total Deaths'})
fig.update_layout(title='Top 10 Countries with the most Deaths',title_x=0.5)
fig.show()
fig = go.Figure(data=[go.Scatter(
    x=df_deaths.head(10)['Country'],
    y=df_deaths.head(10)['TotalDeaths'],
    mode='markers',
    marker=dict(
        color=100+np.random.randn(500),
        size=(df_deaths.head(10)['TotalCases']/12000),
        showscale=True
        )
)])
fig.update_layout(
    title='Top 10 Countries with the most Deaths',
    xaxis_title="Countries",
    yaxis_title="Deaths",
    template='plotly_white',
    title_x = 0.5

)
fig.show()
fig = px.bar(x=df_covid['TotalCases'],y=df_covid['Country'],labels={'x':'TotalCases','y':'Country'})
fig.update_layout(title='Total Cases per Country',title_x=0.6)
fig.show()
fig = px.bar(x=df_covid['TotalRecovered'],y=df_covid['Country'],labels={'x':'TotalRecovered','y':'Country'})
fig.update_layout(title='Total Recoveries per Country',title_x=0.6)
fig.show()
fig = px.bar(x=df_covid['TotalDeaths'],y=df_covid['Country'],labels={'x':'TotalDeaths','y':'Country'})
fig.update_layout(title='Total Deaths per Country',title_x=0.6)
fig.show()
df_covid.replace('USA', "United States of America", inplace = True)
df_covid.replace('Tanzania', "United Republic of Tanzania", inplace = True)
df_covid.replace('Democratic Republic of Congo', "Democratic Republic of the Congo", inplace = True)
df_covid.replace('Congo', "Republic of the Congo", inplace = True)
df_covid.replace('Lao', "Laos", inplace = True)
df_covid.replace('Syrian Arab Republic', "Syria", inplace = True)
df_covid.replace('Serbia', "Republic of Serbia", inplace = True)
df_covid.replace('Czechia', "Czech Republic", inplace = True)
df_covid.replace('UAE', "United Arab Emirates", inplace = True)
fig = px.choropleth(df_covid, locations=df_covid['Country'],
                    color=df_covid['TotalCases'],locationmode='country names', 
                    hover_name=df_covid['Country'], 
                    color_continuous_scale=px.colors.sequential.Tealgrn )
fig.update_layout(
    title='Chloropleth of Total Cases In Each Country',title_x=0.5
)
fig.show()
fig = px.choropleth(df_covid, locations=df_covid['Country'],
                    color=df_covid['TotalRecovered'],locationmode='country names', 
                    hover_name=df_covid['Country'], 
                    color_continuous_scale=px.colors.sequential.Tealgrn )
fig.update_layout(
    title='Chloropleth of Total Recoveries In Each Country',title_x=0.5
)
fig.show()
fig = px.choropleth(df_covid, locations=df_covid['Country'],
                    color=df_covid['TotalDeaths'],locationmode='country names', 
                    hover_name=df_covid['Country'], 
                    color_continuous_scale=px.colors.sequential.Tealgrn )
fig.update_layout(
    title='Chloropleth of Total Deaths In Each Country',title_x=0.5
)
fig.show()
labels = df_covid['Country']
values = df_covid['TotalCases']
fig = px.pie(df_covid,values=values,labels=labels,names=df_covid['Country'])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title='Total Cases per Country with Percentages',title_x=0.4)
fig.show()
labels = df_covid['Country']
values = df_covid['TotalDeaths']
fig = px.pie(df_covid,values=values,labels=labels,names=df_covid['Country'])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title='Total Deaths per Country with Percentages',title_x=0.4)
fig.show()
labels = df_covid['Country']
values = df_covid['TotalRecovered']
fig = px.pie(df_covid,values=values,labels=labels,names=df_covid['Country'])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title='Total Recoveries per Country with Percentages',title_x=0.4)
fig.show()
pop_all_countries = []

for i in range(len(df_covid)):
    try:
        res = requests.get('https://restcountries.eu/rest/v2/name/{}'.format(str(df_covid['Country'].iloc[i]).replace(' ','%20')))
        pop_all_countries.append(res.json()[0]['population'])
    except:
        pop_all_countries.append(404)
        continue
indices = [i for i, x in enumerate(pop_all_countries) if x == 404]
indices
df_covid_pop = df_covid.copy()

pop_all_countries[3] = 1295210000
pop_all_countries[72] = 50801405
pop_all_countries[122] = 85026000
pop_all_countries[169] = 25281000
pop_all_countries[177] = 4741000

df_covid_pop.insert(5,'Population',pop_all_countries)
df_covid_pop.drop(df_covid_pop[df_covid_pop['Code'] == 'DP'].index,inplace=True)
df_covid_pop.insert(6,'TotalCases rel to pop',list((df_covid_pop['TotalCases'] / df_covid_pop['Population'])* 100))
df_covid_pop.insert(6,'TotalDeaths rel to pop',list((df_covid_pop['TotalDeaths'] / df_covid_pop['Population'])* 100))
df_covid_pop.insert(6,'TotalRecovered rel to pop',list((df_covid_pop['TotalRecovered'] / df_covid_pop['Population'])* 100))
df_covid_pop_cases = df_covid_pop.sort_values('TotalCases rel to pop',ascending=False)
df_covid_pop_deaths = df_covid_pop.sort_values('TotalDeaths rel to pop',ascending=False)
df_covid_pop_recoveries = df_covid_pop.sort_values('TotalRecovered rel to pop',ascending=False)
fig = px.bar(df_covid_pop_cases,y=df_covid_pop_cases['Country'],x=df_covid_pop_cases[df_covid_pop_cases.columns[-3]])
fig.update_layout(title='Countries with the most Cases relative to the population(in %)',title_x=0.5)
fig.update_traces(textposition='inside')
fig.show()
fig = px.pie(df_covid_pop_cases,names=df_covid_pop_cases['Country'],values=df_covid_pop_cases[df_covid_pop_cases.columns[-3]],labels=df_covid_pop_cases[df_covid_pop_cases.columns[-3]],hole=0.5)
fig.update_layout(title='Countries with the most Cases relative to the population(in %)',title_x=0.5)
fig.update_traces(textposition='inside')
fig.show()
fig = px.bar(df_covid_pop_deaths,y=df_covid_pop_deaths['Country'],x=df_covid_pop_deaths[df_covid_pop_deaths.columns[-2]])
fig.update_layout(title='Countries with the most Deaths relative to the population(in %)',title_x=0.5)
fig.update_traces(textposition='inside')
fig.show()
fig = px.pie(df_covid_pop_deaths,names=df_covid_pop_deaths['Country'],values=df_covid_pop_deaths[df_covid_pop_deaths.columns[-2]],hole=0.5)
fig.update_layout(title='Countries with the most Deaths relative to the population(in %)',title_x=0.5)
fig.update_traces(textposition='inside')
fig.show()
fig = px.bar(df_covid_pop_recoveries,y=df_covid_pop_recoveries['Country'],x=df_covid_pop_recoveries[df_covid_pop_recoveries.columns[-1]])
fig.update_layout(title='Countries with the most Recoveries relative to the population(in %)',title_x=0.5)
fig.update_traces(textposition='inside')
fig.show()
fig = px.pie(df_covid_pop_recoveries,names=df_covid_pop_recoveries['Country'],values=df_covid_pop_recoveries[df_covid_pop_recoveries.columns[-1]],hole=0.5)
fig.update_layout(title='Countries with the most Recoveries relative to the population(in %)',title_x=0.5)
fig.update_traces(textposition='inside')
fig.show()
fig = px.bar(df_covid_pop_recoveries,x=df_covid_pop_recoveries['Country'].head(10),y=df_covid_pop_recoveries[df_covid_pop_recoveries.columns[-1]].head(10))
fig.update_layout(title='Top 10 Countries with the most Recoveries relative to the population(in %)',title_x=0.5)
fig.update_traces(textposition='inside')
fig.show()
fig = go.Figure(data=[go.Scatter(
    x=df_covid_pop_recoveries.head(10)['Country'],
    y=df_covid_pop_recoveries.head(10)['TotalRecovered rel to pop'],
    mode='markers',
    marker=dict(
        color=100+np.random.randn(500),
        size=(df_covid_pop_recoveries.head(10)['TotalRecovered rel to pop']*50),
        showscale=True
        )
)])
fig.update_layout(
    title='Top 10 Countries with the most Recoveries realtive to Population',
    xaxis_title="Countries",
    yaxis_title="Confirmed Recoveries",
    template='plotly_white',
    title_x = 0.5

)
fig.show()
fig = px.bar(df_covid_pop_recoveries,x=df_covid_pop_deaths['Country'].head(10),y=df_covid_pop_deaths[df_covid_pop_deaths.columns[-2]].head(10))
fig.update_layout(title='Top 10 Countries with the most Deaths relative to the population(in %)',title_x=0.5)
fig.update_traces(textposition='inside')
fig.show()
fig = go.Figure(data=[go.Scatter(
    x=df_covid_pop_deaths.head(10)['Country'],
    y=df_covid_pop_deaths.head(10)['TotalDeaths rel to pop'],
    mode='markers',
    marker=dict(
        color=100+np.random.randn(500),
        size=(df_covid_pop_deaths.head(10)['TotalDeaths rel to pop']*1000),
        showscale=True
        )
)])
fig.update_layout(
    title='Top 10 Countries with the most Deaths realtive to Population',
    xaxis_title="Countries",
    yaxis_title="Deaths",
    template='plotly_white',
    title_x = 0.5

)
fig.show()
fig = px.bar(df_covid_pop_cases,x=df_covid_pop_cases['Country'].head(10),y=df_covid_pop_cases[df_covid_pop_cases.columns[-3]].head(10))
fig.update_layout(title='Top 10 Countries with the most Cases relative to the population(in %)',title_x=0.5)
fig.update_traces(textposition='inside')
fig.show()
fig = go.Figure(data=[go.Scatter(
    x=df_covid_pop_deaths.head(10)['Country'],
    y=df_covid_pop_deaths.head(10)['TotalCases rel to pop'],
    mode='markers',
    marker=dict(
        color=100+np.random.randn(500),
        size=(df_covid_pop_deaths.head(10)['TotalCases rel to pop']*100),
        showscale=True
        )
)])
fig.update_layout(
    title='Top 10 Countries with the most Cases realtive to Population',
    xaxis_title="Countries",
    yaxis_title="Cases",
    template='plotly_white',
    title_x = 0.5

)
fig.show()
time_series = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/worldwide-aggregated.csv')
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_series.index, y=time_series['Confirmed'],
                    mode='lines',
                    name='Confirmed cases'))


fig.update_layout(
    title='Worldwide Evolution of Confirmed Cases over time',
        template='plotly_white',
      yaxis_title="Confirmed cases",
    xaxis_title="Days",
    title_x = 0.5

)

fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_series.index, y=time_series['Deaths'],
                    mode='lines',
                    name='Deaths'))


fig.update_layout(
    title='Worldwide Evolution of Confirmed Deaths over time',
        template='plotly_white',
      yaxis_title="Confirmed Deaths",
    xaxis_title="Days",
    title_x = 0.5

)

fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_series.index, y=time_series['Recovered'],
                    mode='lines',
                    name='Recoveries'))


fig.update_layout(
    title='Worldwide Evolution of Confirmed Recoveries over time',
        template='plotly_white',
      yaxis_title="Confirmed Recoveries",
    xaxis_title="Days",
    title_x = 0.5

)

fig.show()
fig = go.Figure()
fig.add_trace(go.Bar(x=time_series['Date'], y=time_series['Recovered']))


fig.update_layout(
    title='Confirmed Cases per Day and Month',
        template='plotly_white',
      yaxis_title="Confirmed Cases",
    xaxis_title="Days",
    title_x = 0.5

)

fig.show()
fig = go.Figure()
fig.add_trace(go.Bar(x=time_series['Date'], y=time_series['Deaths']))


fig.update_layout(
    title='Confirmed Deaths per Day and Month',
        template='plotly_white',
      yaxis_title="Confirmed Deaths",
    xaxis_title="Days",
    title_x = 0.5

)

fig.show()
fig = go.Figure()
fig.add_trace(go.Bar(x=time_series['Date'], y=time_series['Recovered']))


fig.update_layout(
    title='Confirmed Recoveries per Day and Month',
        template='plotly_white',
      yaxis_title="Confirmed Recoveries",
    xaxis_title="Days",
    title_x = 0.5

)

fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_series.index, y=time_series['Increase rate'],
                    mode='lines',
                    name='Increase Rate'))


fig.update_layout(
    title='Worldwide Increase Rate over time',
        template='plotly_white',
      yaxis_title="Increase rate",
    xaxis_title="Days",
    title_x = 0.5

)

fig.show()
all_tests = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv')
owid_data = all_tests.drop_duplicates(['ISO code']).groupby('Entity').head(len(all_tests)).sort_values('Cumulative total',ascending=False)
fig = go.Figure()
fig.add_trace(go.Bar(x=owid_data['ISO code'],y=owid_data['Cumulative total']))
fig.update_layout(title='Number of tests per Country',title_x=0.5)
fig.show()
labels = owid_data['ISO code']
values = owid_data['Cumulative total']
names = labels
fig = px.pie(owid_data,values=values,names=names,labels=labels)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title='Number of tests per Country',title_x=0.5)
fig.show()
fig = go.Figure()
fig.add_trace(go.Bar(x=owid_data['ISO code'].head(10),y=owid_data['Cumulative total'].head(10)))
fig.update_layout(title='Top 10 Countries with the most Tests ',title_x=0.5)
fig.show()
df_covid_test = df_covid[df_covid['Code'].isin(['US','IT','ES','SA','PL','KW','IR','AU','DE','BR'])]
df_covid_test.columns = df_covid_test.columns.str.strip()
owid_data = owid_data.head(10).sort_values('ISO code',ascending=False)
df_covid_test.insert(5,'Tests',list(owid_data.head(10)['Cumulative total'].astype(int)))
fig = go.Figure(data=[go.Scatter(
    x=df_covid_test.head(10)['Country'],
    y=df_covid_test.head(10)['Tests'],
    mode='markers',
    marker=dict(
        color=100+np.random.randn(500),
        size=(df_covid_test.head(10)['TotalCases']/12000),
        showscale=True
        )
)])
fig.update_layout(
    title='Top 10 Countries with the most Tests',
    xaxis_title="Countries",
    yaxis_title="Test Cases",
    template='plotly_white',
    title_x = 0.5

)
fig.show()

fig = go.Figure(data=[go.Table(header=dict(values=list(df_covid_test.columns)),
                 cells=dict(values=[df_covid_test['Country'], df_covid_test['Code'],df_covid_test['TotalCases'],df_covid_test['TotalRecovered'],df_covid_test['TotalDeaths'],df_covid_test['Tests']]))
                     ])
fig.update_layout(title='Top 10 Countries with the most Tests, including Cases,Deaths and Recoveries',title_x=0.5)
fig.show()
ccodes= df_covid_test.head(10)['Code']
fig = go.Figure(data=[
    go.Bar(name='Total Tests', x=ccodes, y=df_covid_test.head(10)['Tests']),
    go.Bar(name='Total Cases', x=ccodes, y=df_covid_test.head(10)['TotalCases']),
    go.Bar(name='Recoveries', x=ccodes, y=df_covid_test.head(10)['TotalRecovered']),
    go.Bar(name='Total Deaths', x=ccodes, y=df_covid_test.head(10)['TotalDeaths'])
])
fig.update_layout(barmode='group',title='Top 10 Countries with the Tests, including Cases,Deaths and Recoveries',title_x=0.5)
fig.show()
owid_data = all_tests.drop_duplicates(['ISO code']).groupby('Entity').head(len(all_tests)).sort_values('Cumulative total',ascending=False)

ow = owid_data.groupby('Date')[['Cumulative total']].sum().reset_index().sort_values('Date')
ow

fig = go.Figure(data=[go.Table(header=dict(values=list(ow.columns)),
                 cells=dict(values=[ow['Date'], ow['Cumulative total']]))])
fig.update_layout(title='Sum of test taken worldwide',title_x=0.5)
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(x=ow['Date'],y=ow['Cumulative total'],mode='lines'))
fig.update_layout(title='Evolution of Worldwide Tests',title_x=0.5)
fig.show()