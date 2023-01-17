#Importing Libraries for data manipulation and loading files.
import pandas as pd                              
import numpy as np       
import json
import datetime

#Importing libraries for graphical analyses.
import matplotlib.pyplot as plt                  
import plotly.express as px                      
import plotly.offline as py                     
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns                            

#Other essential libraries to import.
import glob                             
import os     
from urllib.request import urlopen
import warnings
warnings.filterwarnings('ignore')

#Required Libraries for analyses
!pip install pivottablejs
from pivottablejs import pivot_ui
#Reading the cumulative cases dataset
covid_cases = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

#Viewing the dataset
covid_cases.head()

#Grouping the coutries together for further analyses
country_list = covid_cases['Country/Region'].unique()

country_grouped_covid = covid_cases[0:1]

for country in country_list:
    test_data = covid_cases['Country/Region'] == country   
    test_data = covid_cases[test_data]
    country_grouped_covid = pd.concat([country_grouped_covid, test_data], axis=0)
    
country_grouped_covid.reset_index(drop=True)
country_grouped_covid.head()
#Plotting a bar graph for confirmed cases vs deaths due to COVID-19 in World.

unique_dates = country_grouped_covid['ObservationDate'].unique()
confirmed_cases = []
recovered = []
deaths = []

for date in unique_dates:
    date_wise = country_grouped_covid['ObservationDate'] == date  
    test_data = country_grouped_covid[date_wise]
    
    confirmed_cases.append(test_data['Confirmed'].sum())
    deaths.append(test_data['Deaths'].sum())
    recovered.append(test_data['Recovered'].sum())
    
#Converting the lists to a pandas dataframe.

country_dataset = {'Date' : unique_dates, 'Confirmed' : confirmed_cases, 'Recovered' : recovered, 'Deaths' : deaths}
country_dataset = pd.DataFrame(country_dataset)

#Plotting the Graph of Cases vs Deaths Globally.

fig = go.Figure()
fig.add_trace(go.Bar(x=country_dataset['Date'], y=country_dataset['Confirmed'], name='Confirmed Cases of COVID-19', marker_color='rgb(55, 83, 109)'))
fig.add_trace(go.Bar(x=country_dataset['Date'],y=country_dataset['Deaths'],name='Total Deaths because of COVID-19',marker_color='rgb(26, 118, 255)'))

fig.update_layout(title='Confirmed Cases and Deaths from COVID-19',xaxis_tickfont_size=14,
                  yaxis=dict(title='Reported Numbers',titlefont_size=16,tickfont_size=14,),
    legend=dict(x=0,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),
    barmode='group',bargap=0.15, bargroupgap=0.1)
fig.show()


fig = go.Figure()
fig.add_trace(go.Bar(x=country_dataset['Date'], y=country_dataset['Confirmed'], name='Confirmed Cases of COVID-19', marker_color='rgb(55, 83, 109)'))
fig.add_trace(go.Bar(x=country_dataset['Date'],y=country_dataset['Recovered'],name='Total Recoveries because of COVID-19',marker_color='rgb(26, 118, 255)'))

fig.update_layout(title='Confirmed Cases and Recoveries from COVID-19',xaxis_tickfont_size=14,
                  yaxis=dict(title='Reported Numbers',titlefont_size=16,tickfont_size=14,),
    legend=dict(x=0,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),
    barmode='group',bargap=0.15, bargroupgap=0.1)
fig.show()
#Reading the dataset
search_data_china = country_grouped_covid['Country/Region'] == 'Mainland China'       
china_data = country_grouped_covid[search_data_china]

#Viewing the dataset
china_data.head()
with open('../input/china-geo-json/china_geojson.json') as json_file:
    china = json.load(json_file)
#Creating the interactive map
py.init_notebook_mode(connected=True)

#GroupingBy the dataset for the map

formated_gdf = china_data.groupby(['ObservationDate', 'Province/State'])['Confirmed', 'Deaths', 'Recovered'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = pd.to_datetime(formated_gdf['ObservationDate'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['log_ConfirmedCases'] = np.log(formated_gdf.Confirmed + 1)

#Plotting the figure

fig = px.choropleth(formated_gdf,geojson = china,locations='Province/State',scope="asia",
                     color="log_ConfirmedCases", hover_name='Province/State',projection="mercator",
                     animation_frame="Date",width=1000, height=800,
                     color_continuous_scale=px.colors.sequential.Viridis,
                     title='The Spread of COVID-19 Cases Across China')

#Showing the figure

fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)
def plot_case_graph(a):
    
    search_city = a

    #Draws the plot for the searched city

    search_data = country_grouped_covid['Province/State'] == search_city       
    search_data = country_grouped_covid[search_data]                           

    x = search_data['ObservationDate']
    y = search_data['Confirmed']
    b = search_data['Confirmed'].values
    
    
    a = b.shape   
    a = a[0]
    growth_rate = []
    
    #Loop to calculate the daily growth rate of cases
    
    for i in range(1,a):                                       
        daily_growth_rate = ((b[i]/b[i-1])-1)*100
        growth_rate.append(daily_growth_rate)                                      

    growth_rate.append(daily_growth_rate)
        
    data = {'Growth' : growth_rate}
    b = pd.DataFrame(data)
    
    #Plotting the chart for confirmed cases vs date     
        
    plt.figure(figsize=(15,5))
    plt.bar(x,y,color="#9ACD32")                              
    plt.xticks(rotation=90)
    
    plt.title('Confirmed Cases Over time in {}'.format(search_city))
    plt.xlabel('Time')
    plt.ylabel('Confirmed Cases')

    plt.tight_layout()
    plt.show()
    
    #Plotting the chart daily growth rate in confirmed COVID-19 Cases.
    
    plt.figure(figsize=(15,5))
    
    plt.plot(x,b,color='red', marker='o', linestyle='dashed',linewidth=2, markersize=8,label="Daily Growth Rate of New Confirmed Cases")
    plt.xticks(rotation=90)
    
    plt.title('Percentage Daily Increase of Confirmed Cases Over time in {}'.format(search_city))
    plt.xlabel('Time')
    plt.ylabel('Percentage Daily Increase')

    plt.tight_layout()
    plt.show()
    
plot_case_graph('Hubei')
#Plotting Data for Anhui Province (Most affected Province in China after Hubei)
plot_case_graph('Anhui')

#Use can use the above function to plot the cases and grwoth rate of covid-19 cases across China. Feel free to fork this notebook and use.
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure(figsize=(15,5))
img=mpimg.imread('../input/china-covid19-data/Anhui.png')
imgplot = plt.imshow(img)
plt.show()

plt.figure(figsize=(15,5))
img=mpimg.imread('../input/china-covid19-data/Beijing.png')
imgplot = plt.imshow(img)
plt.show()

for i in range(1,17):
    plt.figure(figsize=(15,5))
    img=mpimg.imread('../input/china-covid19-data/Screenshot ({}).png'.format(303+i))
    imgplot = plt.imshow(img)
    plt.show()
#Importing the dataset
mitigation_policies = pd.read_csv("/kaggle/input/uncover/UNCOVER/HDE_update/acaps-covid-19-government-measures-dataset.csv")

#Generating the pivoting toolkit
pivot_ui(mitigation_policies)
#Analysis for all the important keywords from the dataset

plt.figure(figsize=(15,20))
frame1 = plt.gca()


img=mpimg.imread('../input/china-covid19-data/Word Art.png')
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)
imgplot = plt.imshow(img)
plt.show()
#We load one more dataset avaialable on Mitigation measures to analyze the cases vs mitigation  measures adopted.
mitigation_measures_tot = pd.read_csv('../input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')

#Generating the pivoting toolkit
pivot_ui(mitigation_measures_tot)