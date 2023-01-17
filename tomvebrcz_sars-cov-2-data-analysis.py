# NumPy is the fundamental package for scientific computing with Python.
import numpy as np

# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd

# Plotly's library makes interactive, publication-quality graphs.
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# COVID-19 Complete Dataset (Updated every 24hrs) Dataset.
world_data = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv", 
                         parse_dates=['Date'])

# Countries and Continents Dataset.
world_countries = pd.read_csv("../input/countriesandcontinents/countries.csv")
# Replacing country names.
world_data['Country/Region'] = world_data['Country/Region'].replace('Mainland China', 'China')
world_data['Country/Region'] = world_data['Country/Region'].replace('US', 'United States')
world_data['Country/Region'] = world_data['Country/Region'].replace('UK', 'United Kingdom')
world_data['Country/Region'] = world_data['Country/Region'].replace('North Macedonia', 'Macedonia')

# Filling missing values with "NA".
world_data[['Province/State']] = world_data[['Province/State']].fillna('NA')

# Calculate "Active cases".
world_data['Active'] = world_data.eval('Confirmed-Deaths-Recovered')
# Separate the list of countries into continents.
asia = world_countries[world_countries['Region']=='ASIA']
europe = world_countries[world_countries['Region']=='EUROPE']
america = world_countries[world_countries['Region']=='NORTHERN_AMERICA']
latinamerica = world_countries[world_countries['Region']=='LATIN']
africa = world_countries[world_countries['Region']=='AFRICA']
oceania = world_countries[world_countries['Region']=='OCEANIA']
neareast = world_countries[world_countries['Region']=='NEAR_EAST']
# Getting last day data
world_data_lastday = world_data[world_data['Date'] == max(world_data['Date'])].reset_index()

# Separate the Diamond Princess Cruise Ship
#ship_data = world_data[world_data['Province/State']=='Diamond Princess cruise ship']
#ship_data_lastday = ship_data[ship_data['Date'] == max(ship_data['Date'])].reset_index()

# Asia data separation and getting last data
asia_data = world_data[world_data['Country/Region'].isin(asia['Country'])]
asia_data_lastday = asia_data[asia_data['Date'] == max(asia_data['Date'])].reset_index()

# Europe data separation and getting last data
europe_data = world_data[world_data['Country/Region'].isin(europe['Country'])]
europe_data_lastday = europe_data[europe_data['Date'] == max(europe_data['Date'])].reset_index()

# America data separation and getting last data
america_data = world_data[world_data['Country/Region'].isin(america['Country'])]
america_data_lastday = america_data[america_data['Date'] == max(america_data['Date'])].reset_index()

# Latin America data separation and getting last data
latinamerica_data = world_data[world_data['Country/Region'].isin(latinamerica['Country'])]
latinamerica_data_lastday = latinamerica_data[latinamerica_data['Date'] == max(latinamerica_data['Date'])].reset_index()

# Africa data separation and getting last data
africa_data = world_data[world_data['Country/Region'].isin(africa['Country'])]
africa_data_lastday = africa_data[africa_data['Date'] == max(africa_data['Date'])].reset_index()

# Oceania data separation and getting last data
oceania_data = world_data[world_data['Country/Region'].isin(oceania['Country'])]
oceania_data_lastday = oceania_data[oceania_data['Date'] == max(oceania_data['Date'])].reset_index()

# Near East data separation and getting last data
neareast_data = world_data[world_data['Country/Region'].isin(neareast['Country'])]
neareast_data_lastday = neareast_data[neareast_data['Date'] == max(neareast_data['Date'])].reset_index()
world_data_lastday.groupby(['Date'])[['Confirmed','Recovered','Deaths','Active']].sum().style.background_gradient(cmap='YlOrRd').hide_index().set_properties(**{'text-align': 'center'})
world_temp = world_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered','Active'].sum().reset_index()

world_graph = go.Figure()

world_graph.add_trace(go.Scatter(x=world_temp['Date'], y=world_temp['Confirmed'], name="Confirmed",
                    line_shape='linear', mode='lines+markers', marker_color='rgba(77, 5, 232, 1)'))
world_graph.add_trace(go.Scatter(x=world_temp['Date'], y=world_temp['Deaths'], name="Deaths",
                    line_shape='linear', mode='lines+markers', marker_color='rgba(242, 38, 19, 1)'))
world_graph.add_trace(go.Scatter(x=world_temp['Date'], y=world_temp['Recovered'], name="Recovered",
                    line_shape='linear', mode='lines+markers', marker_color='rgba(248, 148, 6, 1)'))
world_graph.add_trace(go.Scatter(x=world_temp['Date'], y=world_temp['Active'], name="Active Cases",
                    line_shape='linear', mode='lines+markers', marker_color='rgba(0, 230, 64, 1)'))

world_graph.update_layout(title='WORLD DATA STATISTICS',
                          yaxis_title="Number of cases",
                          font=dict(
                            family="Courier New, monospace",
                            size=12,
                            color="#7f7f7f"
                            ),
                          template ="plotly_white",
                         )

world_graph.update_xaxes(tickformat = '%d.%m.%Y')

world_graph.show()
difference_temp = world_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered','Active'].sum().reset_index()
difference_temp["Difference"] = difference_temp["Active"].diff()
difference_temp["Difference %"] = difference_temp.eval('Difference/(Active/100)/100')

difference = go.Figure()

difference.add_trace(go.Scatter(x=difference_temp['Date'], y=difference_temp['Difference %'], name="Difference",
                    line_shape='linear', mode='lines+markers', marker_color='rgba(0, 0, 0, 1)'))

difference.update_layout(title='DAILY DIFFERENCE OF ACTIVE CASES',
                          yaxis_title="Number of cases",
                          font=dict(
                            family="Courier New, monospace",
                            size=12,
                            color="#7f7f7f"
                            ),
                          template ="plotly_white",
                         )

difference.update_xaxes(tickformat = '%d.%m.%Y')
difference.update_yaxes(tickformat = '.2%')

difference.show()
active_cases = make_subplots(rows=3, cols=3, subplot_titles=("Asia", "Europe", "America", "Latin America","Africa","Oceania","","Near East"))

asia_temp = asia_data.groupby(['Date'])['Active'].sum().reset_index()
active_cases.add_trace(go.Scatter(x=asia_temp['Date'], y=asia_temp['Active'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(0, 230, 64, 1)',), row=1, col=1)

europe_temp = europe_data.groupby(['Date'])['Active'].sum().reset_index()
active_cases.add_trace(go.Scatter(x=europe_temp['Date'], y=europe_temp['Active'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(0, 230, 64, 1)'), row=1, col=2)

america_temp = america_data.groupby(['Date'])['Active'].sum().reset_index()
active_cases.add_trace(go.Scatter(x=america_temp['Date'], y=america_temp['Active'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(0, 230, 64, 1)'), row=1, col=3)

latinamerica_temp = latinamerica_data.groupby(['Date'])['Active'].sum().reset_index()
active_cases.add_trace(go.Scatter(x=latinamerica_temp['Date'], y=latinamerica_temp['Active'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(0, 230, 64, 1)'), row=2, col=1)

africa_temp = africa_data.groupby(['Date'])['Active'].sum().reset_index()
active_cases.add_trace(go.Scatter(x=africa_temp['Date'], y=africa_temp['Active'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(0, 230, 64, 1)'), row=2, col=2)

oceania_temp = oceania_data.groupby(['Date'])['Active'].sum().reset_index()
active_cases.add_trace(go.Scatter(x=oceania_temp['Date'], y=oceania_temp['Active'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(0, 230, 64, 1)'), row=2, col=3)

neareast_temp = neareast_data.groupby(['Date'])['Active'].sum().reset_index()
active_cases.add_trace(go.Scatter(x=neareast_temp['Date'], y=neareast_temp['Active'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(0, 230, 64, 1)'), row=3, col=2)

active_cases.update_layout(title='ACTIVE CASES ACROSS THE WORLD',
                   font=dict(
                        family="Courier New, monospace",
                        size=12,
                        color="#7f7f7f"),
                    template ="plotly_white",
                    showlegend=False)

active_cases.update_xaxes(tickformat = '%d.%m.%y',
                         showticklabels=False)

active_cases.show()
recovered_cases = make_subplots(rows=3, cols=3, subplot_titles=("Asia", "Europe", "America", "Latin America","Africa","Oceania","","Near East"))

asia_temp = asia_data.groupby(['Date'])['Recovered'].sum().reset_index()
recovered_cases.add_trace(go.Scatter(x=asia_temp['Date'], y=asia_temp['Recovered'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(248, 148, 6, 1)',), row=1, col=1)

europe_temp = europe_data.groupby(['Date'])['Recovered'].sum().reset_index()
recovered_cases.add_trace(go.Scatter(x=europe_temp['Date'], y=europe_temp['Recovered'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(248, 148, 6, 1)'), row=1, col=2)

america_temp = america_data.groupby(['Date'])['Recovered'].sum().reset_index()
recovered_cases.add_trace(go.Scatter(x=america_temp['Date'], y=america_temp['Recovered'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(248, 148, 6, 1)'), row=1, col=3)

latinamerica_temp = latinamerica_data.groupby(['Date'])['Recovered'].sum().reset_index()
recovered_cases.add_trace(go.Scatter(x=latinamerica_temp['Date'], y=latinamerica_temp['Recovered'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(248, 148, 6, 1)'), row=2, col=1)

africa_temp = africa_data.groupby(['Date'])['Recovered'].sum().reset_index()
recovered_cases.add_trace(go.Scatter(x=africa_temp['Date'], y=africa_temp['Recovered'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(248, 148, 6, 1)'), row=2, col=2)

oceania_temp = oceania_data.groupby(['Date'])['Recovered'].sum().reset_index()
recovered_cases.add_trace(go.Scatter(x=oceania_temp['Date'], y=oceania_temp['Recovered'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(248, 148, 6, 1)'), row=2, col=3)

neareast_temp = neareast_data.groupby(['Date'])['Recovered'].sum().reset_index()
recovered_cases.add_trace(go.Scatter(x=neareast_temp['Date'], y=neareast_temp['Recovered'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(248, 148, 6, 1)'), row=3, col=2)

recovered_cases.update_layout(title='RECOVERED CASES ACROSS THE WORLD',
                   font=dict(
                        family="Courier New, monospace",
                        size=12,
                        color="#7f7f7f"),
                    template ="plotly_white",
                    showlegend=False)

recovered_cases.update_xaxes(tickformat = '%d.%m.%y',
                            showticklabels=False)

recovered_cases.show()
mortality = world_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered','Active'].sum().reset_index()
mortality['Mortality'] = mortality.eval('Deaths/(Confirmed/100)/100')

mortality_graph = go.Figure()

mortality_graph.add_trace(go.Scatter(x=mortality['Date'], y=mortality['Mortality'], name="Mortality",
                    line_shape='linear', mode='lines+markers', marker_color='rgba(242, 38, 19, 1)'))

mortality_graph.update_layout(title='MORTALITY',
                          font=dict(
                            family="Courier New, monospace",
                            size=12,
                            color="#7f7f7f"
                            ),
                          template ="plotly_white",
                         )

mortality_graph.update_xaxes(tickformat = '%d.%m.%Y')
mortality_graph.update_yaxes(tickformat = '.1%')

mortality_graph.show()
mortality_cases = make_subplots(rows=3, cols=3, subplot_titles=("Asia", "Europe", "America", "Latin America","Africa","Oceania","","Near East"))

asia_mortality = asia_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered','Active'].sum().reset_index()
asia_mortality['Mortality'] = asia_mortality.eval('Deaths/(Confirmed/100)/100')
mortality_cases.add_trace(go.Scatter(x=asia_mortality['Date'], y=asia_mortality['Mortality'], name="Mortality",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)',), row=1, col=1)

europe_mortality = europe_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered','Active'].sum().reset_index()
europe_mortality['Mortality'] = europe_mortality.eval('Deaths/(Confirmed/100)/100')
mortality_cases.add_trace(go.Scatter(x=europe_mortality['Date'], y=europe_mortality['Mortality'], name="Mortality",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)'), row=1, col=2)

america_mortality = america_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered','Active'].sum().reset_index()
america_mortality['Mortality'] = america_mortality.eval('Deaths/(Confirmed/100)/100')
mortality_cases.add_trace(go.Scatter(x=america_mortality['Date'], y=america_mortality['Mortality'], name="Mortality",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)'), row=1, col=3)

latinamerica_mortality = latinamerica_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered','Active'].sum().reset_index()
latinamerica_mortality['Mortality'] = latinamerica_mortality.eval('Deaths/(Confirmed/100)/100')
mortality_cases.add_trace(go.Scatter(x=latinamerica_mortality['Date'], y=latinamerica_mortality['Mortality'], name="Mortality",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)'), row=2, col=1)

africa_mortality = africa_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered','Active'].sum().reset_index()
africa_mortality['Mortality'] = africa_mortality.eval('Deaths/(Confirmed/100)/100')
mortality_cases.add_trace(go.Scatter(x=africa_mortality['Date'], y=africa_mortality['Mortality'], name="Mortality",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)'), row=2, col=2)

oceania_mortality = oceania_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered','Active'].sum().reset_index()
oceania_mortality['Mortality'] = oceania_mortality.eval('Deaths/(Confirmed/100)/100')
mortality_cases.add_trace(go.Scatter(x=oceania_mortality['Date'], y=oceania_mortality['Mortality'], name="Mortality",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)'), row=2, col=3)

neareast_mortality = neareast_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered','Active'].sum().reset_index()
neareast_mortality['Mortality'] = neareast_mortality.eval('Deaths/(Confirmed/100)/100')
mortality_cases.add_trace(go.Scatter(x=neareast_mortality['Date'], y=neareast_mortality['Mortality'], name="Mortality",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)'), row=3, col=2)

mortality_cases.update_layout(title='MORTALITY ACROSS THE WORLD',
                   font=dict(
                        family="Courier New, monospace",
                        size=12,
                        color="#7f7f7f"),
                    template ="plotly_white",
                    showlegend=False)

mortality_cases.update_xaxes(tickformat = '%d.%m.%Y',
                            showticklabels=False)
mortality_cases.update_yaxes(tickformat = '.1%')

mortality_cases.show()
deaths_cases = make_subplots(rows=3, cols=3, subplot_titles=("Asia", "Europe", "America", "Latin America","Africa","Oceania","","Near East"))

asia_temp = asia_data.groupby(['Date'])['Deaths'].sum().reset_index()
deaths_cases.add_trace(go.Scatter(x=asia_temp['Date'], y=asia_temp['Deaths'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)',), row=1, col=1)

europe_temp = europe_data.groupby(['Date'])['Deaths'].sum().reset_index()
deaths_cases.add_trace(go.Scatter(x=europe_temp['Date'], y=europe_temp['Deaths'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)'), row=1, col=2)

america_temp = america_data.groupby(['Date'])['Deaths'].sum().reset_index()
deaths_cases.add_trace(go.Scatter(x=america_temp['Date'], y=america_temp['Deaths'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)'), row=1, col=3)

latinamerica_temp = latinamerica_data.groupby(['Date'])['Deaths'].sum().reset_index()
deaths_cases.add_trace(go.Scatter(x=latinamerica_temp['Date'], y=latinamerica_temp['Deaths'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)'), row=2, col=1)

africa_temp = africa_data.groupby(['Date'])['Deaths'].sum().reset_index()
deaths_cases.add_trace(go.Scatter(x=africa_temp['Date'], y=africa_temp['Deaths'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)'), row=2, col=2)

oceania_temp = oceania_data.groupby(['Date'])['Deaths'].sum().reset_index()
deaths_cases.add_trace(go.Scatter(x=oceania_temp['Date'], y=oceania_temp['Deaths'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)'), row=2, col=3)

neareast_temp = neareast_data.groupby(['Date'])['Deaths'].sum().reset_index()
deaths_cases.add_trace(go.Scatter(x=neareast_temp['Date'], y=neareast_temp['Deaths'], name="Active Cases",
                    line_shape='linear', mode='lines', marker_color='rgba(242, 38, 19, 1)'), row=3, col=2)

deaths_cases.update_layout(title='DEATHS CASES ACROSS THE WORLD',
                   font=dict(
                        family="Courier New, monospace",
                        size=12,
                        color="#7f7f7f"),
                    template ="plotly_white",
                    showlegend=False)

deaths_cases.update_xaxes(tickformat = '%d.%m.%y',showticklabels=False)

deaths_cases.show()