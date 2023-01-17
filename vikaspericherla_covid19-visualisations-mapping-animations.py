import pandas as pd #data analytics and manipulation
import matplotlib.pyplot as plt #data visualisation

import plotly.offline as py 
py.init_notebook_mode(connected=True) #displays plot on the notebook while the kernel is running
import plotly.graph_objs as go
import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 
#NOTEBOOK WHILE KERNEL IS RUNNING
import plotly.io as pio
#pio.renderers.default = 'browser' #to initialise plotly
df1 = pd.read_csv("../input/covid19/covid.csv")
df1.head()
df2=pd.read_csv("../input/covid19/covid_grouped.csv")
df2.head()

df2.tail()
df1.columns
df1.drop(['NewCases','NewDeaths','NewRecovered'],axis=1,inplace=True) #axis ensures all rows are deleted/ inplace ensures that changes are made to existing dataset

df1.head()
from plotly.figure_factory import create_table
table=create_table(df1.head(),colorscale="blackbody")
py.offline.iplot(table)
df1.columns
init_notebook_mode(connected=True) #do not miss this line

px.bar(df1.head(10),x='Country/Region',y='TotalCases',color='TotalDeaths',height=500,hover_data=['Country/Region', 'Continent'])
px.bar(df1.head(10),x='Country/Region',y='TotalTests',color='TotalDeaths',height=500,hover_data=['Country/Region', 'Continent'])
px.bar(df1.head(10),x='Continent',y='TotalCases',color='TotalDeaths',height=500,hover_data=['Country/Region', 'Continent'])
df1.columns
px.scatter(df1.head(50),x='Continent',y='TotalCases',hover_data=['Country/Region', 'Continent'],color='TotalCases',size='TotalCases',size_max=80,log_y=True)
px.scatter(df1.head(30),x='Country/Region',y='Tests/1M pop',hover_data=['Country/Region', 'Continent'],color='TotalCases',size='Tests/1M pop',size_max=80,log_y=True,labels='Country/Region')
px.scatter(df1.head(10),x='TotalCases',y='TotalDeaths',hover_data=['Country/Region', 'Continent'],color='TotalCases',size='TotalDeaths',size_max=80,log_y=True,labels='Country/Region')
df2.columns
df2.head()
df2.tail()
df_IND=df2.loc[df2['Country/Region']=='India']
px.bar(df_IND,x='Date',y='Confirmed',color='Confirmed',height=400)
px.line(df_IND,x='Date',y='New cases',height=400)
px.choropleth(df2,locations='iso_alpha',color='Confirmed',hover_name='Country/Region',color_continuous_scale="Blues",animation_frame="Date")
px.choropleth(df2,locations='iso_alpha',color='Deaths',hover_name='Country/Region',color_continuous_scale="Viridis",animation_frame="Date")
px.choropleth(df2,locations='iso_alpha',color='Deaths',hover_name='Country/Region',color_continuous_scale="Viridis",animation_frame="Date",projection="orthographic")
px.choropleth(df2,locations='iso_alpha',color='Deaths',hover_name='Country/Region',color_continuous_scale="RdYlGn",animation_frame="Date",projection="natural earth")
df2.columns
px.bar(df2,x='WHO Region',y='Confirmed',color='WHO Region',animation_frame="Date",hover_name="Country/Region")
px.bar(df2,x='WHO Region',y='New cases',color='WHO Region',animation_frame="Date",hover_name="Country/Region")