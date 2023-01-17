# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
s=pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv',parse_dates=['Datum'])
s.head()
#dropping the unnecessary columns
s.drop(columns=['Unnamed: 0','Unnamed: 0.1'],inplace=True)
#correcting column names
s.rename(columns= {'Datum':'Date','Status Rocket':'Rocket_status',' Rocket':'Mission_Cost',
                   'Status Mission':'Mission_Status'},inplace=True)
s.head()
s.info()
import missingno as msno
msno.matrix(s)
s.isnull().sum()
s['Company Name'].value_counts(normalize=True).head()*100
s['Location'].value_counts(normalize=True).head()*100
#converting date column into datetime format
s['Date'] = pd.to_datetime(s['Date'],format ="%Y-%m-%d",utc=True)
s['Date'].head()
s['Detail'].value_counts(normalize=True).head()*100
s['Rocket_status'].value_counts()
#changing the values of Rocket_status column to 'Retired' and 'Active'
s['Rocket_status'].replace({'StatusRetired':'Retired','StatusActive':'Active'},inplace=True)
s['Rocket_status'].value_counts(normalize=True)*100
s.head()
#filling up the missing values
s['Mission_Cost']=s['Mission_Cost'].fillna(0.0)
s.isnull().sum()
s['Mission_Status'].value_counts(normalize=True)*100
#adding separate year,month and day columns
s['Year']=s['Date'].dt.year
s['Month']=s['Date'].dt.month
s['Day']=s['Date'].dt.day
#converting mission_cost column from object to float type
s['Mission_Cost'] = s['Mission_Cost'].fillna(0.0).str.replace(',', '')
s['Mission_Cost'] = s['Mission_Cost'].astype(np.float64).fillna(0.0)
s.head()
import matplotlib.pyplot as plt
import plotly.express as px
import plotly
plotly.offline.init_notebook_mode(connected = True)
import seaborn as sns
import plotly.graph_objs as go
ds = s['Company Name'].value_counts().reset_index()
ds.columns = ['Company', 'Number of launches']
px.bar(
    ds, 
    x='Number of launches', 
    y="Company", 
    orientation='h', 
    title='Number of Space Missions Launched By Every Company', 
    height=1000,width=1000,color='Company')
px.pie(s,'Rocket_status')
px.pie(s,'Mission_Status')
y=s['Year'].value_counts().reset_index()
y.columns=['Year','Number of launches']
px.bar(y,x='Year',y='Number of launches',color='Year')
ds = s.groupby(['Year', 'Company Name'])['Mission_Status'].count().reset_index().sort_values(['Year', 'Mission_Status']
                                                                                             ,ascending=False)
ds.columns = ['Year', 'Company', 'Number of Launches']
px.scatter(
    ds, 
    x="Year", 
    y="Number of Launches", 
    color='Company',
    size='Number of Launches',
    title='Distribution of launches over the year by companies',height=1000)
su=s[s['Mission_Status']=='Success']
su=su[su['Mission_Cost']>0.0]
su.rename(columns={'Mission_Cost':'Successful_missions_count'},inplace=True)
t=su.groupby('Company Name')['Successful_missions_count'].count().reset_index()
px.bar(t,y='Company Name',x='Successful_missions_count'
       ,title='Distribution of Success Rate Over Companies',color='Successful_missions_count',height=1000)
e=s.groupby(['Company Name','Year'])['Mission_Cost'].sum().reset_index()
e=e[e['Mission_Cost']>0.0]
px.scatter(e,x='Year',y='Mission_Cost',color='Company Name'
           ,size='Mission_Cost',height=1000,title='Yearly distribution of Cost by Companies')
m=s[s['Mission_Cost']>0.0]
px.bar(m,y='Company Name',x='Mission_Cost',color='Mission_Status',height=1000,width=1000,
       title='Distribution of Mission Status and Cost Over Companies')
me=s.groupby(['Company Name','Month'])['Mission_Cost'].sum().reset_index()
me=me[me['Mission_Cost']>0.0]
px.bar(me,x='Month',y='Mission_Cost',color='Company Name'
           ,height=500,title='Monthly distribution of Cost by Companies')
a=s[s['Rocket_status']=='Active']
r=a.groupby(['Company Name'])['Rocket_status'].count().reset_index()
r.rename(columns={'Rocket_status':'Active_Rockets'},inplace=True)
px.bar(r,y='Company Name',x='Active_Rockets',title='Distribution of Active Rockets Over Companies',color='Active_Rockets',height=1000)
r=s[s['Rocket_status']=='Retired']
f=r.groupby(['Company Name'])['Rocket_status'].count().reset_index()
f.rename(columns={'Rocket_status':'Retired_Rockets'},inplace=True)
px.bar(f,y='Company Name',x='Retired_Rockets'
       ,title='Distribution of Retired Rockets Over Companies',color='Retired_Rockets',height=1000)
#extracting country from the location column
s['Country'] = s['Location'].str.split(', ').str[-1]
s.head()
s['Country'].value_counts()
#mapping of space stations and areas to their respective countries
countries_dict = {
    'Russia' : 'Russian Federation',
    'New Mexico' : 'USA',
    "Yellow Sea": 'China',
    "Shahrud Missile Test Site": "Iran",
    "Pacific Missile Range Facility": 'USA',
    "Barents Sea": 'Russian Federation',
    "Gran Canaria": 'USA'
}
s['Country'] = s['Country'].replace(countries_dict)
s['Country'].value_counts()
ds = s['Country'].value_counts().reset_index()
ds.columns = ['Country', 'Number of launches']
px.bar(
    ds, 
    x='Number of launches', 
    y="Country", 
    orientation='h', 
    title='Number of Space Missions Launched by the countries', 
    height=1000,width=1000,color='Country')
b=s[s['Mission_Cost']>0.0]
av=b.groupby(['Country'])['Mission_Cost'].mean().reset_index()
av.rename(columns={'Mission_Cost':'Average_Budget'},inplace=True)
px.bar(av,x='Average_Budget',y='Country',title='Average Budget of Each Country')
mo = s[s['Mission_Cost']>0]
mo = mo.groupby(['Year','Country'])['Mission_Cost'].mean().reset_index()
px.line(
    mo, 
    x="Year", 
    y="Mission_Cost",
    facet_row='Country',height=1500,width=1000,
    title='Average Money Spent By Countries per Year',color='Country')
mo = s[s['Mission_Cost']>0]
mo = mo.groupby(['Month','Country'])['Mission_Cost'].mean().reset_index()
px.line(
    mo, 
    x="Month", 
    y="Mission_Cost",
    facet_row='Country',height=1500,width=1000,
    title='Average Money Spent By Countries per Month',color='Country')
ds = s[s['Mission_Status']=='Success']
ds = s.groupby(['Year','Country'])['Mission_Status'].count().reset_index().sort_values(['Year', 'Mission_Status'], ascending=False)
ds.columns = ['Year','Country', 'Number of launches']
px.scatter(
    ds, 
    x='Year', 
    y="Number of launches", 
    title='Number of Successful Space Missions Launched by the countries', 
    color='Country',size="Number of launches")
















