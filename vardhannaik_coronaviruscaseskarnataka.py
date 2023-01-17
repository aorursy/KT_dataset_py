# import the necessary libraries
import numpy as np 
import pandas as pd 
import os
# Visualisation libraries
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
import pycountry
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
#from pywaffle import Waffle
py.init_notebook_mode(connected=True)
import folium 
from folium import plugins
plt.style.use("fivethirtyeight")# for pretty graphs
# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5
#plt.rcParams['image.cmap'] = 'viridis'
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Disable warnings 
import warnings
warnings.filterwarnings('ignore')
# Reading the datasets

df= pd.read_csv('/kaggle/input/coronavirus-cases-karnataka/Covid cases in Karnataka.csv')
#df_india = df.copy()
# Coordinates of India States and Uts
#India_coord = pd.read_csv('coronavirus-cases-in-india/Indian Coordinates.csv')
df["Patient#"]= df["Patient#"].astype(str) 
df["Age"]= df["Age"].astype(str) 
df["Linked Patient#"]= df["Linked Patient#"].astype(str) 
#df.drop(['S. No.'],axis=1,inplace=True)
#df['Total cases'] = df['Total Confirmed cases (Indian National)'] + df['Total Confirmed cases ( Foreign National )'] 
df['Active cases'] = df['Total cases'] - (df['Cured/Discharged/Migrated'] + df['Deaths'])
df['Active original cases'] = df['Total cases'] - (df['Cured/Discharged/Migrated'] + df['Deaths']) - df['No of contacts']
print(f'Total number of Confirmed COVID 2019 cases across Karnataka:', df['Total cases'].sum())
print(f'Total number of Active COVID 2019 cases across Karnataka:', df['Active cases'].sum())
print(f'Total number of Active original COVID 2019 cases across Karnataka:', df['Active original cases'].sum())
print(f'Total number of Contacts COVID 2019 cases across Karnataka:', df['No of contacts'].sum())
print(f'Total number of Cured/Discharged/Migrated COVID 2019 cases across Karnataka:', df['Cured/Discharged/Migrated'].sum())
print(f'Total number of Deaths due to COVID 2019  across Karnataka:', df['Deaths'].sum())
print(f'Total number of Places affected:', len(df.groupby(['Place']).sum()))
#df.groupby(['Place']).sum()
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: pink' if v else '' for v in is_max]

#df.style.apply(highlight_max,subset=['Total Confirmed cases (Indian National)', 'Total Confirmed cases ( Foreign National )'])
df.groupby(['Place']).sum().style.apply(highlight_max,subset=['Cured/Discharged/Migrated', 'Deaths','Total cases','Active cases','No of contacts','Active original cases'])
df_t = df.T
#print(df)
df_axes = df.copy()
df_axes1 = df_axes.groupby(['Date'],as_index=False).sum()
df_axes2 = df_axes.groupby(['Discharge Date'],as_index=False).sum()


df_axes2['Date']=df_axes2['Discharge Date']
df_axes2.drop(['Total cases','Deaths','No of contacts','Active cases', 'Active original cases','Discharge Date'], axis= 1, inplace = True)
df_axes2.set_index(['Date'], inplace=True)


df_axes1.drop(['Cured/Discharged/Migrated'], axis= 1, inplace = True)
df_axes1.set_index(['Date'],inplace=True)

df_concat = pd.concat([df_axes1, df_axes2], axis=1, sort=True)
df_concat.replace(0, np.nan, inplace=True)
df_concat.plot(lw=2, colormap='jet', marker='.', markersize=10, title='Covid 2019 cases in Karnataka')

#df_t = df.T
#print(df)
df_axes = df.copy()
df_axes1 = df_axes.groupby(['Place','Date'],as_index=False).sum()
df_axes2 = df_axes.groupby(['Discharge Date','Place'],as_index=False).sum()


df_axes2['Date']=df_axes2['Discharge Date']
df_axes2.drop(['Total cases','Deaths','No of contacts','Active cases', 'Active original cases','Discharge Date'], axis= 1, inplace = True)
df_axes2.set_index(['Place','Date'], inplace=True)


df_axes1.drop(['Cured/Discharged/Migrated'], axis= 1, inplace = True)
df_axes1.set_index(['Place','Date'],inplace=True)

df_concat = pd.concat([df_axes1, df_axes2], axis=1, sort=True)
df_concat.replace(0, np.nan, inplace=True)
#df.reset_index().plot(x="b",y="c")
#df_concat.reset_index().plot(y="Place",lw=2, colormap='jet', marker='.', markersize=10, title='Covid 2019 cases in Karnataka')

#fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
#df_concat
df_concat['Active cases'].groupby(['Date','Place']).sum().unstack().plot(kind='bar',lw=2,stacked=True, colormap='jet', title='Covid 2019 Active cases in Karnataka')
#print(df_concat.groupby(['Date','Place']).sum())
df_concat['Total cases'].groupby(['Date','Place']).sum().unstack().plot(kind='bar',lw=2,stacked=True, colormap='jet', title='Covid 2019 Total cases in Karnataka')
df_concat['Active cases'].groupby(['Date','Place']).sum().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).unstack().plot(kind='bar',stacked=True,colormap='jet', title='Covid 2019 Active cases % in Karnataka')
import matplotlib.ticker as mtick
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()
df_axes = df.copy()
df_axes["Age"]= df_axes["Age"].astype(int) 
bins= [0,20,40,60,80,110]
labels = ['0-20','21-40','41-60','61-80','81-110']
df_axes['AgeGroup'] = pd.cut(df_axes['Age'], bins=bins, labels=labels, right=False)
#df_axes["AgeGroup"]= df_axes["AgeGroup"].astype(str) 
#df_axes["Age"]= df_axes["Age"].astype(str) 
#df_axes['Date'] = df_axes['Date'].astype('datetime64[ns]')
#df_axes['Origin Dates'] = df_axes['Origin Dates'].astype('datetime64[ns]')
#df_axes['AgeGroup'][ df_axes.Age < 21 ] = '1-20'
#df_axes['AgeGroup'][ df_axes.Age > 20 & df_axes.Age < 41 ] = '21-40'
#df_axes['AgeGroup'][ df_axes.Age > 40 & df_axes.Age < 61 ] = '41-60'
#df_axes['AgeGroup'][ df_axes.Age > 60 & df_axes.Age < 81 ] = '61-80'
#df_axes['AgeGroup'][ df_axes.Age > 80  ] = '81-100+'
df_axes1 = df_axes.groupby(['Place','AgeGroup','Gender','Foreign Origin','Date'],as_index=False).sum()
df_axes2 = df_axes.groupby(['Discharge Date','AgeGroup','Gender','Foreign Origin','Place'],as_index=False).sum()


df_axes2['Date']=df_axes2['Discharge Date']
df_axes2.drop(['Total cases','Deaths','No of contacts','Active cases', 'Active original cases','Discharge Date','Age'], axis= 1, inplace = True)
df_axes2.set_index(['Place','AgeGroup','Gender','Foreign Origin','Date'], inplace=True)


df_axes1.drop(['Cured/Discharged/Migrated','Age'], axis= 1, inplace = True)
df_axes1.set_index(['Place','AgeGroup','Gender','Foreign Origin','Date'],inplace=True)

df_concat = pd.concat([df_axes1, df_axes2], axis=1, sort=True)
df_concat.replace(0, np.nan, inplace=True)
df_concat['Total cases'].groupby(['Date','Foreign Origin']).sum().unstack().plot(kind='bar',lw=2,stacked=True, colormap='jet', title='Covid 2019 Total cases in Karnataka')

#df.reset_index().plot(x="b",y="c")
#df_concat.reset_index().plot(y="Place",lw=2, colormap='jet', marker='.', markersize=10, title='Covid 2019 cases in Karnataka')

#fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
#df_concat
#df_concat['Active cases'].groupby(['Date','Gender']).sum().unstack().plot(kind='bar',lw=2,stacked=True, colormap='jet', title='Covid 2019 Active cases in Karnataka')
#print(df_concat.groupby(['Date','Age','Place']).sum())
#df_concat1 = df_concat.groupby(['Date']).sum()
#print(df_concat1)
#fig, ax = plt.subplots()
#ax.plot(df_concat1['Total cases'], marker='.', markersize=2, color='0.6', lw=2, linestyle='dashed', label='Dates')

#df_concat['Active cases'].groupby(['Date','AgeGroup']).sum().groupby(level=0).apply(
#    lambda x: 100 * x / x.sum()
#).unstack().plot(kind='bar',stacked=True,colormap='jet', title='Covid 2019 Active cases % in Karnataka')
#import matplotlib.ticker as mtick
#plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
#plt.show()
#df_concat['Total cases'].groupby(['Date','Origin Dates']).sum().plot()
df
df_axes = df.copy()
df_axes["Age"]= df_axes["Age"].astype(int) 
bins= [0,20,40,60,80,110]
labels = ['0-20','21-40','41-60','61-80','81-110']
df_axes['AgeGroup'] = pd.cut(df_axes['Age'], bins=bins, labels=labels, right=False)
#df_axes["AgeGroup"]= df_axes["AgeGroup"].astype(str) 
#df_axes["Age"]= df_axes["Age"].astype(str) 
#df_axes['Date'] = df_axes['Date'].astype('datetime64[ns]')
#df_axes['Origin Dates'] = df_axes['Origin Dates'].astype('datetime64[ns]')
#df_axes['AgeGroup'][ df_axes.Age < 21 ] = '1-20'
#df_axes['AgeGroup'][ df_axes.Age > 20 & df_axes.Age < 41 ] = '21-40'
#df_axes['AgeGroup'][ df_axes.Age > 40 & df_axes.Age < 61 ] = '41-60'
#df_axes['AgeGroup'][ df_axes.Age > 60 & df_axes.Age < 81 ] = '61-80'
#df_axes['AgeGroup'][ df_axes.Age > 80  ] = '81-100+'
df_axes1 = df_axes.groupby(['Place','AgeGroup','Gender','Date'],as_index=False).sum()
df_axes2 = df_axes.groupby(['Discharge Date','AgeGroup','Gender','Place'],as_index=False).sum()


df_axes2['Date']=df_axes2['Discharge Date']
df_axes2.drop(['Total cases','Deaths','No of contacts','Active cases', 'Active original cases','Discharge Date','Age'], axis= 1, inplace = True)
df_axes2.set_index(['Place','AgeGroup','Gender','Date'], inplace=True)


df_axes1.drop(['Cured/Discharged/Migrated','Age'], axis= 1, inplace = True)
df_axes1.set_index(['Place','AgeGroup','Gender','Date'],inplace=True)

df_concat = pd.concat([df_axes1, df_axes2], axis=1, sort=True)
df_concat.replace(0, np.nan, inplace=True)
df_concat['Total cases'].groupby(['Date','AgeGroup']).sum().unstack().plot(kind='bar',lw=2,stacked=True, colormap='jet', title='Covid 2019 Total cases in Karnataka')
df_concat['Total cases'].groupby(['Date','Gender']).sum().unstack().plot(kind='bar',lw=2,stacked=True, colormap='jet', title='Covid 2019 Total cases in Karnataka')

df_concat['Total cases'].groupby(['Date','Place']).sum().unstack().plot(kind='bar',lw=2,stacked=True, colormap='jet', title='Covid 2019 Total cases in Karnataka')
# Coordinates of India States and Uts
India_coord_district = pd.read_csv('/kaggle/input/coronavirus-cases-karnataka/centroids karnataka.csv')
# create map and display it

df_full_district = pd.merge(India_coord_district ,df.groupby(['Place'])['Active cases'].sum(),on=['Place'])
map2 = folium.Map(location=[20, 80], zoom_start=3.5,tiles='Stamen Toner')

for lat, lon, value, name in zip(df_full_district['Latitude'], df_full_district['Longitude'], df_full_district['Active cases'], df_full_district['Place']):
    folium.CircleMarker([lat, lon],
                        radius=value*0.7,
                        popup = ('<strong>State</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Active Cases</strong>: ' + str(value) + '<br>'),
                        color='red',
                        
                        fill_color='red',
                        fill_opacity=0.3 ).add_to(map2)
map2
x = df.groupby('Place')['Active cases'].sum().sort_values(ascending=False).to_frame()
x.style.background_gradient(cmap='Reds')
df['Origin Dates'] = pd.to_datetime(df['Origin Dates'])
df['Date'] = pd.to_datetime(df['Date'])
df['Death date'] = pd.to_datetime(df['Death date'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df2 = df[df['Origin Dates'].notna()]
df3 = df[df['Origin Dates'].notna() == False ]
df2['Origin Diff'] =  df2['Date'] - df2['Origin Dates']
#df2[['Date', 'Origin Dates', 'Origin Diff']].mean()
#df3['Origin Diff']= df2['Origin Diff'].median()
df3['Origin Diff']= df3['Date']- df3['Date']
#df3['Origin Dates'] = df3['Date']+df3['Origin Diff']
#df3[['Date','Origin Dates','Origin Diff']]
df4=pd.concat([df2, df3])
df4=df4[['Patient#','Origin Dates','Date','Origin Diff','Death date','Discharge Date']]
df4['Origin Diff']=(df4['Origin Diff'] / np.timedelta64(1, 'D')).astype(int)

df2 = df4[df4['Death date'].notna()]
df3 = pd.concat([df4[df4['Death date'].notna() == False ], df2[df2['Origin Dates'].notna() == False ] ])
df2 = df2[df2['Origin Dates'].notna()]
df2['Death days from Origin'] =  df2['Death date'] - df2['Origin Dates']
df2['Death days from +ve']    =  df2['Death date'] - df2['Date']

df3['Death days from Origin']= df3['Date']- df3['Date']
df3['Death days from +ve']= df3['Date']- df3['Date']
df4=pd.concat([df2, df3])

df4=df4[['Patient#','Origin Dates','Date','Origin Diff','Death date','Discharge Date','Death days from Origin','Death days from +ve']]
df4['Death days from Origin']=(df4['Death days from Origin'] / np.timedelta64(1, 'D')).astype(int)
df4['Death days from +ve']=(df4['Death days from +ve'] / np.timedelta64(1, 'D')).astype(int)


df2 = df4[df4['Discharge Date'].notna()]
df3 = df4[df4['Discharge Date'].notna() == False ]
df2['Days discharged in Hospital']       =  df2['Discharge Date'] - df2['Date']
df2['Days recovering in Hospital']    =  df2['Date']- df2['Date']

df3['Days discharged in Hospital']= df3['Date']- df3['Date']
df3['Days recovering in Hospital']= df4['Date'].max()- df3['Date']
df4=pd.concat([df2, df3])
df4['Days discharged in Hospital']=(df4['Days discharged in Hospital'] / np.timedelta64(1, 'D')).astype(int)
df4['Days recovering in Hospital']=(df4['Days recovering in Hospital'] / np.timedelta64(1, 'D')).astype(int)


df4=df4[['Patient#','Origin Dates','Date','Origin Diff','Death date','Discharge Date','Death days from Origin','Death days from +ve','Days discharged in Hospital','Days recovering in Hospital']]

#'Death days from +ve','Discharge Date','
# Days recovering in hospital','Days in Hospital'

#df4['Origin Diff']=df4['Origin Diff'].astype(int) 
#df4.set_index(['Date'],inplace=True)
#df4.sort_index(inplace=True)
df4.sort_values(by=['Patient#'],inplace=True)
df4.replace(0, np.nan, inplace=True)
df4['Date of case']=df4['Date']
ax = df4.plot(lw=2, y='Death days from Origin', x='Death days from +ve', c='Origin Diff', s=45, colormap='jet' , label= 'Death days from Origin/+ve' , kind='scatter', marker='.', title='Covid 2019 cases in Karnataka')
df4.plot(y='Origin Diff' , x='Days recovering in Hospital' ,kind='kde', color='blue' ,label = 'Recovering from Origin', title='Covid 2019 cases in Karnataka')
#ax=df4.plot(y='Origin Dates', x='Date of case', style=".", color='Blue', label='Foreign Origin', title='Covid 2019 cases in Karnataka')
#ax=df4.plot(y='Death date', x='Date of case',style=".", color='Red', label='Death', title='Covid 2019 cases in Karnataka' , ax=ax)
#ax=df4.plot(y='Discharge Date', x='Date of case',style=".", color='Green', label='Discharge', title='Covid 2019 cases in Karnataka' , ax=ax)

#ax1 = df4.plot( y='Origin Dates', x='Date of case', c='Origin Diff' , colormap='jet', kind='scatter', title='Covid 2019 cases in Karnataka')
#ax1 = df4.plot( y='Death date', x='Date of case', c='Death days from Origin', colormap='jet' ,kind='scatter', title='Covid 2019 cases in Karnataka'
#ax1 = df4.plot( y='Death date', x='Date of case', c='Death days from +ve', colormap='jet' ,kind='scatter', title='Covid 2019 cases in Karnataka')
#ax2 = df4.plot( y='Discharge Date', x='Date of case', c='Days discharged in Hospital', colormap='jet' ,kind='scatter', title='Covid 2019 cases in Karnataka')
#ax2 = df4.plot( y='Date', x='Date of case', c='Days recovering in Hospital', colormap='jet' ,kind='scatter', title='Covid 2019 cases in Karnataka')
df4


