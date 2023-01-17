# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from IPython.display import Image

Image("../input/covid19-pandemic-in-pakistan/zee.jpg")
# importing the required libraries

import pandas as pd



# Visualisation libraries

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight') # Above is a special style template for matplotlib, highly useful for visualizing time series data

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import folium 

from folium import plugins



# Manipulating the default plot size

plt.rcParams['figure.figsize'] = 10, 12



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')
df = pd.read_excel('../input/covid19-pandemic-in-pakistan/Covid cases in Pakistan.xlsx')

df.head()
df.describe()




#Correlation 

df_corr = df[['Name of State / UT','Total Confirmed cases(Pak Nation)','Total Confirmed cases(Forign Nation)','Cured','Deaths','Total cases']].corr()

df_corr
# is Null Values

df.info(null_counts = True)
#Cordinates of Pakistan States 

Pakistan_coord = pd.read_excel('../input/covid19-pandemic-in-pakistan/Pakistan Coordinates.xlsx')

# Coordinates of Pakistan States and Union Territories

Pakistan_coord = pd.read_excel('../input/covid19-pandemic-in-pakistan/Pakistan Coordinates.xlsx')



#Day by day data of Pakistan, Korea, Italy and Wuhan

dbd_Pakistan = pd.read_excel('../input/covid19-pandemic-in-pakistan/per_day_cases.xlsx',parse_dates=True, sheet_name = 'Pakistan')

dbd_Italy = pd.read_excel('../input/covid19-pandemic-in-pakistan/per_day_cases.xlsx',parse_dates=True, sheet_name="Italy")

dbd_Korea = pd.read_excel('../input/covid19-pandemic-in-pakistan/per_day_cases.xlsx',parse_dates=True, sheet_name="Korea")

dbd_Wuhan = pd.read_excel('../input/covid19-pandemic-in-pakistan/per_day_cases.xlsx',parse_dates=True, sheet_name="Wuhan")



dbd_Pakistan.tail()




#Learn how to play around with the dataframe and create a new attribute of 'Total Case'

#Total case is the total number of confirmed cases (Indian National + Foreign National)



df['Total cases'] = df['Total Confirmed cases(Pak Nation)'] + df['Total Confirmed cases(Forign Nation)']

total_cases = df['Total cases'].sum()

print('Total number of confirmed COVID 2019 cases across Pakistan till date (26th March, 2020):', total_cases)

#highlight 

df.style.background_gradient(cmap='Reds') 




#Total Active  is the Total cases - (Number of death + Cured)

df['Total Active'] = df['Total cases'] - (df['Deaths'] + df['Cured'])

total_active = df['Total Active'].sum()

print('Total number of active COVID 2019 cases across Pakistan:', total_active)

Tot_Cases = df.groupby('Name of State / UT')['Total Active'].sum().sort_values(ascending=False).to_frame()

Tot_Cases.style.background_gradient(cmap='Reds')
# Learn how to use folium to create a zoomable map



df_full = pd.merge(Pakistan_coord,df,on='Name of State / UT')

map = folium.Map(location=[30, 70], zoom_start=5,tiles='Stamenterrain')



for lat, lon, value, name in zip(df_full['Latitude'], df_full['Longitude'], df_full['Total cases'], df_full['Name of State / UT']):

    folium.CircleMarker([lat, lon], radius=value*0.12, popup = ('<strong>State</strong>: ' + str(name).capitalize() + '<br>''<strong>Total Cases</strong>: ' + str(value) + '<br>'),color='red',fill_color='red',fill_opacity=0.3 ).add_to(map)

map
#Learn how to use Seaborn for visualization

f, ax = plt.subplots(figsize=(12, 8))

data = df_full[['Name of State / UT','Total cases','Cured','Deaths']]

data.sort_values('Total cases',ascending=False,inplace=True)

sns.set_color_codes("pastel")

sns.barplot(x="Total cases", y="Name of State / UT", data=data,label="Total", color="gray")



sns.set_color_codes("muted")

sns.barplot(x="Cured", y="Name of State / UT", data=data, label="Cured", color="red")





# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 35), ylabel="",xlabel="Cases")

sns.despine(left=True, bottom=True)
#Learn how to create interactive graphs using plotly

# import plotly.graph_objects as go

# Rise of COVID-19 cases in Pakistan



import plotly

plotly.io.renderers.default = 'notebook'



fig = go.Figure()

fig.add_trace(go.Scatter(x=dbd_Pakistan['Date'], y = dbd_Pakistan['Total Cases'], mode='lines+markers',name='Total Cases'))

fig.update_layout(title_text='Trend of Coronavirus Cases in Pakistan (Cumulative cases)',plot_bgcolor='rgb(230, 230,230)')

fig.show()



# New COVID-19 cases reported daily in Pakistan



import plotly.express as px

fig = px.bar(dbd_Pakistan, x="Date", y="New Cases", barmode='group', height=400)

fig.update_layout(title_text='Coronavirus Cases in Pakistan on daily basis',plot_bgcolor='rgb(230, 230, 230)')



fig.show()
# import plotly.express as px

fig = px.bar(dbd_Pakistan, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=600,

             title='Confirmed Cases in Pakistan', color_discrete_sequence = px.colors.cyclical.IceFire)



'''Colour Scale for plotly

https://plot.ly/python/builtin-colorscales/

'''



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()



fig = px.bar(dbd_Italy, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=600,

             title='Confirmed Cases in Italy', color_discrete_sequence = px.colors.cyclical.IceFire)



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()





fig = px.bar(dbd_Korea, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=600,

             title='Confirmed Cases in South Korea', color_discrete_sequence = px.colors.cyclical.IceFire)



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()



fig = px.bar(dbd_Wuhan, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=600,

             title='Confirmed Cases in Wuhan', color_discrete_sequence = px.colors.cyclical.IceFire)



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()

#Learn how to create subplots using plotly

# import plotly.graph_objects as go

from plotly.subplots import make_subplots



fig = make_subplots(

    rows=2, cols=2,

    specs=[[{}, {}],

           [{"colspan": 2}, None]],

    subplot_titles=("S.Korea","Italy", "Pakistan","Wuhan"))



fig.add_trace(go.Bar(x=dbd_Korea['Date'], y=dbd_Korea['Total Cases'],

                    marker=dict(color=dbd_Korea['Total Cases'], coloraxis="coloraxis")),1, 1)



fig.add_trace(go.Bar(x=dbd_Italy['Date'], y=dbd_Italy['Total Cases'],

                    marker=dict(color=dbd_Italy['Total Cases'], coloraxis="coloraxis")),1, 2)



fig.add_trace(go.Bar(x=dbd_Pakistan['Date'], y=dbd_Pakistan['Total Cases'],

                    marker=dict(color=dbd_Pakistan['Total Cases'], coloraxis="coloraxis")),2, 1)



# fig.add_trace(go.Bar(x=dbd_Wuhan['Date'], y=dbd_Wuhan['Total Cases'],

#                     marker=dict(color=dbd_Wuhan['Total Cases'], coloraxis="coloraxis")),2, 2)



fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False,title_text="Total Confirmed cases(Cumulative)")



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()
from plotly.subplots import make_subplots



fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2}, None]], subplot_titles=("S.Korea","Italy", "Pakistan"))



fig.add_trace(go.Scatter(x=dbd_Korea['Date'], y=dbd_Korea['Total Cases'], marker=dict(color=dbd_Korea['Total Cases'], coloraxis="coloraxis")), 1, 1)



fig.add_trace(go.Scatter(x=dbd_Italy['Date'], y=dbd_Italy['Total Cases'], marker=dict(color=dbd_Italy['Total Cases'], coloraxis="coloraxis")), 1, 2)



fig.add_trace(go.Scatter(x=dbd_Pakistan['Date'], y=dbd_Pakistan['Total Cases'], marker=dict(color=dbd_Pakistan['Total Cases'], coloraxis="coloraxis")), 2, 1)



fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False,title_text="Trend of Coronavirus cases")



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
# import plotly.graph_objects as go



title = 'Main Source for News'

labels = ['S.Korea', 'Italy', 'Pakistan']

colors = ['rgb(122,128,0)', 'rgb(255,0,0)', 'rgb(49,130,189)']



mode_size = [10, 10, 12]

line_size = [1, 1, 8]



fig = go.Figure()





fig.add_trace(go.Scatter(x=dbd_Korea['Days after surpassing 100 cases'], 

                 y=dbd_Korea['Total Cases'],mode='lines',

                 name=labels[0],

                 line=dict(color=colors[0], width=line_size[0]),            

                 connectgaps=True,

    ))

fig.add_trace(go.Scatter(x=dbd_Italy['Days after surpassing 100 cases'], 

                 y=dbd_Italy['Total Cases'],mode='lines',

                 name=labels[1],

                 line=dict(color=colors[1], width=line_size[1]),            

                 connectgaps=True,

    ))



fig.add_trace(go.Scatter(x=dbd_Pakistan['Days after surpassing 100 cases'], 

                 y=dbd_Pakistan['Total Cases'],mode='lines',

                 name=labels[2],

                 line=dict(color=colors[2], width=line_size[2]),            

                 connectgaps=True,

    ))

    

    

    

annotations = []



annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,

                              xanchor='center', yanchor='top',

                              text='Days after crossing 100 cases ',

                              font=dict(family='Arial',

                                        size=12,

                                        color='rgb(150,150,150)'),

                              showarrow=False))



fig.update_layout(annotations=annotations,plot_bgcolor='white',yaxis_title='Cumulative cases')



fig.show()
df = pd.read_csv('../input/covid19-pandemic-in-pakistan/covid_19_data.csv',parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)



df_confirmed = pd.read_csv("../input/covid19-pandemic-in-pakistan/time_series_covid_19_confirmed.csv")

df_recovered = pd.read_csv("../input/covid19-pandemic-in-pakistan/time_series_covid_19_recovered.csv")

df_deaths = pd.read_csv("../input/covid19-pandemic-in-pakistan/time_series_covid_19_deaths.csv")



df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)

df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)

df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)
df_deaths.head()
df.head()
df2 = df.groupby(["Date", "Country", "Province/State"])[['Date', 'Province/State', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

df2.head()
# Check for Pakistan's data

df.query('Country=="Pakistan"').groupby("Last Update")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df.groupby('Date').sum().head()
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index()
fig = go.Figure()

#Plotting datewise confirmed cases

fig.add_trace(go.Scatter(x=confirmed['Date'], y=confirmed['Confirmed'], mode='lines+markers', name='Confirmed',line=dict(color='blue', width=2)))

fig.add_trace(go.Scatter(x=deaths['Date'], y=deaths['Deaths'], mode='lines+markers', name='Deaths', line=dict(color='Red', width=2)))

fig.add_trace(go.Scatter(x=recovered['Date'], y=recovered['Recovered'], mode='lines+markers', name='Recovered', line=dict(color='Green', width=2)))

fig.update_layout(title='Worldwide NCOVID-19 Cases', xaxis_tickfont_size=14,yaxis=dict(title='Number of Cases'))



fig.show()
df_confirmed = df_confirmed[["Province/State","Lat","Long","Country"]]

df_temp = df.copy()

df_temp['Country'].replace({'Mainland China': 'China'}, inplace=True)

df_latlong = pd.merge(df_temp, df_confirmed, on=["Country", "Province/State"])
fig = px.density_mapbox(df_latlong, lat="Lat", lon="Long", hover_name="Province/State", hover_data=["Confirmed","Deaths","Recovered"], animation_frame="Date", color_continuous_scale="Portland", radius=7, zoom=0,height=700)

fig.update_layout(title='Worldwide Corona Virus Cases Time Lapse - Confirmed, Deaths, Recovered')



fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})





fig.show()
from fbprophet import Prophet
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index()
confirmed.columns = ['ds','y']

#confirmed['ds'] = confirmed['ds'].dt.date

confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.tail()
m = Prophet(interval_width=0.95)

m.fit(confirmed)

future = m.make_future_dataframe(periods=12)

future.tail()
#predicting the future with date, and upper and lower limit of y value

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast)
confirmed_forecast_plot =m.plot_components(forecast)
Image("../input/covid19-pandemic-in-pakistan/piccovid.jpg")
deaths.columns = ['ds','y']

deaths['ds'] = pd.to_datetime(deaths['ds'])
m = Prophet(interval_width=0.95)

m.fit(deaths)

future = m.make_future_dataframe(periods=12)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
deaths_forecast_plot = m.plot(forecast)
deaths_forecast_plot = m.plot_components(forecast)
recovered.columns = ['ds','y']

recovered['ds'] = pd.to_datetime(recovered['ds'])
m = Prophet(interval_width=0.95)

m.fit(recovered)

future = m.make_future_dataframe(periods=12)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
recovered_forecast_plot = m.plot(forecast)
recovered_forecast_plot = m.plot_components(forecast)

Image("../input/covid19-pandemic-in-pakistan/covid.png")
