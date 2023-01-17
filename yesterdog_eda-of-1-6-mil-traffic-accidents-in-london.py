import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# graphing and visualization
import imageio
import folium
import folium.plugins as plugins
from mpl_toolkits.basemap import Basemap
import networkx as nx
import heapq 
# Importing datasets
df_one = pd.read_csv("../input/accidents_2005_to_2007.csv")
df_two = pd.read_csv("../input/accidents_2009_to_2011.csv")
df_three = pd.read_csv("../input/accidents_2012_to_2014.csv")
df_one.dtypes
# Merging all the datasets was simple because, there of homogenity in the column labels.
accident =pd.concat([df_one, df_two,df_three], axis=0)
# Apparently English use, OSGR for their co-ordinate data, we don't need here, some kind soul has already compiled latitude and 
# longitudinal data.
accident = accident.drop(['Location_Easting_OSGR','Location_Northing_OSGR'],axis=1)
# For the data available, the maximum number casualites of any year is tabulated.
def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

year_wise_casualties = accident.groupby(['Year'])['Number_of_Casualties'].sum()
year_wise_casualties = year_wise_casualties.reset_index()
year_wise_casualties = year_wise_casualties.style.apply(highlight_max, axis=0)
year_wise_casualties
cas_table = accident.groupby(['Day_of_Week']).agg({'Number_of_Casualties':['sum'],'Speed_limit':['min','max']})
cas_table = cas_table.sort_values([('Number_of_Casualties','sum')],ascending=False)
cas_table = cas_table.reset_index()
cas_table.style.apply(highlight_max)
date_wise_casualities = accident.groupby(['Date'])['Date','Number_of_Casualties'].sum()
worst_accident_days = date_wise_casualities.sort_values(by='Number_of_Casualties',ascending=False)
w_20_days = worst_accident_days.head(20)
w_20_days=w_20_days.reset_index()
w_20_days['Date'] = w_20_days.astype({'Date':'str'})
plt.figure(figsize=(16,4))
ax = sns.barplot(w_20_days['Date'], w_20_days['Number_of_Casualties'], alpha=0.8)
plt.title("The most dangerous days to travel!",fontsize=25)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)
plt.ylabel('casualities recorded', fontsize=12)
plt.xlabel('day of accident', fontsize=12)

plt.show()
level3 = accident[accident['Accident_Severity'] == 3]
level2 = accident[accident['Accident_Severity'] == 2]
level1 = accident[accident['Accident_Severity'] == 1]
f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,9))

## level 3 plot
level3.plot(kind='scatter', x='Longitude',y ='Latitude',
                color='yellow', 
                s=.02, alpha=.6, subplots=True, ax=ax1)
ax1.set_title("level_3_severtiy")
ax1.set_facecolor('black')

## level 2 plot
level2.plot(kind='scatter', x='Longitude',y ='Latitude',
                color='yellow', 
                s=.02, alpha=.6, subplots=True, ax=ax2)
ax2.set_title("level_2_severtiy")
ax2.set_facecolor('black')

# level 1 plot

level1.plot(kind='scatter', x='Longitude',y ='Latitude',
                color='yellow', 
                s=.02, alpha=.6, subplots=True, ax=ax3)
ax3.set_title("level_1_severtiy")
ax3.set_facecolor('black')




f.show()

# It looks like, regardless of the severity, a lot of accidents are concentrated around the same area.
# Identifying the worst districts to travel.
area_wise = accident.groupby(['Local_Authority_(District)'])['Number_of_Casualties'].size()
area_wise = area_wise.sort_values(ascending=False)
area_wise = area_wise.reset_index()
area_wise = area_wise.head(10).style
area_wise
highway_to_hell = accident[accident['Local_Authority_(District)'] == 300]['Local_Authority_(Highway)'].unique()
print("London's most dangerous highway is {}".format(highway_to_hell))      
axe = accident[accident['Day_of_Week'] == 6]
axe = axe[axe['Local_Authority_(District)'] == 300]
axe = axe[axe['Year'] == 2005] # df of year 2005 , with loc_dist 300 and saturday.
axe.head()
data=[]

csts=list(axe['Number_of_Casualties'].unique())
for row in csts:
    subset=axe[axe['Number_of_Casualties']==row]
    data.append(subset[['Latitude','Longitude']].values.tolist())

    
lat = 52.476672
lon = -1.856415
zoom_start=11
print("Saturday in the Year 2005")

# lat and lon varaibles , is where the animation begins.
m = folium.Map(location=[lat, lon], tiles="Cartodb dark_matter", zoom_start=zoom_start)
#inprovising the Heatmapwith time plugin to show variations across star ratings 
hm = plugins.HeatMapWithTime(data,max_opacity=0.3,auto_play=True,display_index=True,radius=10,max_speed=0.1)
hm.add_to(m)
m
# Visualizing the network patterns involved in the accidents.
net = axe[['Accident_Index','Local_Authority_(Highway)','Time']]
net.columns=['source','level_1','target']

graph=nx.from_pandas_edgelist(net)
print(nx.info(graph))
print('Density of graph:{}  '.format(nx.density(graph)))
plt.figure(figsize=(8,8))
plt.title("Londons Accidents Data: Chord plot")
nx.draw_circular(graph,node_size=0.001, width=0.1)
plt.figure(figsize=(8,8))
plt.title("London Accidents Data : Random Layout")
nx.draw_random(graph,node_size=0.01, width=0.01)
plt.figure(figsize=(9,9))
plt.title("London Accidents Data : Kamada Kawai layout")
nx.draw_kamada_kawai(graph,node_size=0.01, width=0.1)



