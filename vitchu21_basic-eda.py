# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import libraries

import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')

# Read data 
d=pd.read_csv("../input/911.csv",
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 
     parse_dates=['timeStamp'],date_parser=dateparse)


# Set index
d.index = pd.DatetimeIndex(d.timeStamp)
d.head(3)
# Type of Emergency - count

title_sort=d["title"].value_counts().reset_index()
title_sort=pd.DataFrame(title_sort.sort_values('title', ascending=False)).iloc[0:10,:]
print("Top 10 highest frequency of emergency case is:\n",title_sort)

# Graphical representation of top 10 categories under type of Emergency ('title')

sns.barplot(y="index", x="title",data=title_sort,palette="Blues_d") 
plt.xlabel("Count")
plt.ylabel("Type of Emergency")
plt.title("Frequently appeared ( top 10 ) type of emergency")
plt.show()
# wordcloud for categories under each type of emergency (title)
x=np.array(d.title)

from collections import Counter
word_could_dict=Counter(x)

from wordcloud import WordCloud
wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)

import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
# count of 3 groups in emergency type ('title') for the given period

d['type'] = d["title"].apply(lambda x: x.split(':')[0])
print("The frequency of emergency type is:\n",d["type"].value_counts())

# Graphical representation for the count of 3 groups in emergency type ('title') for the given period

sns.countplot(x=d["type"], data=d, palette="Blues")
plt.xlabel("Type of emergency")
plt.title("Count of 3 groups in type of emergency")
plt.show()
# wordcloud for type of emergency

y=np.array(d.type)

from collections import Counter
word_could_dict=Counter(y)

from wordcloud import WordCloud
wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)

import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
# pivot table with EMS emergency type ('title')

ems=d[d['type'] == 'EMS' ]
ems_pivot=pd.pivot_table(ems, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

# Resampling annually 'A'
ems_annual=ems_pivot.resample('A', how=[np.sum]).reset_index()
ems_annual.columns = ems_annual.columns.get_level_values(0)
ems_annual.head()
# pivot table with fire emergency type ('title')

fire=d[d['type'] == 'Fire' ]
fire_pivot=pd.pivot_table(fire, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

# Resampling annually 'A' 
fire_annual=fire_pivot.resample('A', how=[np.sum]).reset_index()
fire_annual.columns = fire_annual.columns.get_level_values(0)
fire_annual.head()
# pivot table with traffic emergency type ('title')

traffic=d[d['type'] == 'Traffic' ]
traffic_pivot=pd.pivot_table(traffic, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

# Resampling annually 'A'
traffic_annual=traffic_pivot.resample('A', how=[np.sum]).reset_index()
traffic_annual.columns = traffic_annual.columns.get_level_values(0)
traffic_annual.head()
import matplotlib.lines as mlines

fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  


ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 


ax.plot_date(traffic_annual['timeStamp'], traffic_annual['Traffic: VEHICLE ACCIDENT -'],'k')
ax.plot_date(traffic_annual['timeStamp'], traffic_annual['Traffic: VEHICLE ACCIDENT -'],'ro')

ax.plot_date(ems_annual['timeStamp'], ems_annual['EMS: VEHICLE ACCIDENT'],'g')
ax.plot_date(ems_annual['timeStamp'], ems_annual['EMS: VEHICLE ACCIDENT'],'ro')

ax.plot_date(fire_annual['timeStamp'], fire_annual['Fire: VEHICLE ACCIDENT'],'b')
ax.plot_date(fire_annual['timeStamp'], fire_annual['Fire: VEHICLE ACCIDENT'],'ro')

ax.set_title("Traffic: VEHICLE ACCIDENT  vs  EMS: VEHICLE ACCIDENT vs Fire: VEHICLE ACCIDENT")

# Legend

green_line = mlines.Line2D([], [], color='green', marker='o',markerfacecolor='blue',
                          markersize=7, label='EMS: VEHICLE ACCIDENT')
black_line = mlines.Line2D([], [], color='black', marker='o',markerfacecolor='darkred',
                          markersize=7, label='Traffic: VEHICLE ACCIDENT')
blue_line = mlines.Line2D([], [], color='blue', marker='o',markerfacecolor='darkred',
                          markersize=7, label='Fire: VEHICLE ACCIDENT')

ax.legend(handles=[green_line,black_line,blue_line], loc='best')


fig.autofmt_xdate()
plt.show()
import matplotlib.lines as mlines

fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  

ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 

ax.plot_date(traffic_annual['timeStamp'], traffic_annual['Traffic: DEBRIS/FLUIDS ON HIGHWAY -'],'g')
ax.plot_date(traffic_annual['timeStamp'], traffic_annual['Traffic: DEBRIS/FLUIDS ON HIGHWAY -'],'ro')

ax.plot_date(traffic_annual['timeStamp'], traffic_annual['Traffic: DISABLED VEHICLE -'],'k')
ax.plot_date(traffic_annual['timeStamp'], traffic_annual['Traffic: DISABLED VEHICLE -'],'ro')

ax.plot_date(traffic_annual['timeStamp'], traffic_annual['Traffic: HAZARDOUS ROAD CONDITIONS -'],'b')
ax.plot_date(traffic_annual['timeStamp'], traffic_annual['Traffic: HAZARDOUS ROAD CONDITIONS -'],'ro')

ax.plot_date(traffic_annual['timeStamp'], traffic_annual['Traffic: ROAD OBSTRUCTION -'],'c')
ax.plot_date(traffic_annual['timeStamp'], traffic_annual['Traffic: ROAD OBSTRUCTION -'],'ro')

ax.plot_date(traffic_annual['timeStamp'], traffic_annual['Traffic: VEHICLE ACCIDENT -'],'y')
ax.plot_date(traffic_annual['timeStamp'], traffic_annual['Traffic: VEHICLE ACCIDENT -'],'ro')

ax.set_title("Traffic complaints")

# Legend 
green_line = mlines.Line2D([], [], color='green', marker='o',markerfacecolor='blue',
                          markersize=7, label='Traffic: DEBRIS/FLUIDS ON HIGHWAY ')
black_line = mlines.Line2D([], [], color='black', marker='o',markerfacecolor='darkred',
                          markersize=7, label='Traffic: DISABLED VEHICLE ')
blue_line = mlines.Line2D([], [], color='blue', marker='o',markerfacecolor='darkred',
                          markersize=7, label='Traffic: HAZARDOUS ROAD CONDITIONS')
cyan_line = mlines.Line2D([], [], color='cyan', marker='o',markerfacecolor='darkred',
                          markersize=7, label='Traffic: ROAD OBSTRUCTION ')
yellow_line = mlines.Line2D([], [], color='yellow', marker='o',markerfacecolor='darkred',
                          markersize=7, label='Traffic: VEHICLE ACCIDENT')

ax.legend(handles=[green_line,black_line,blue_line,cyan_line,yellow_line], loc='best')

fig.autofmt_xdate()
plt.show()

# Town - counts
d["twp"].value_counts()[0:9,]
print("The top 10 towns that record the highest number of emergency cases is:\n",d["twp"].value_counts()[0:9,])
# Cross tabulation of each town ('twp') againt the emergency type ('title')

tab=pd.crosstab(d['twp'],d['type']) 
tab.head(10)
# Number of complaints (EMS) from each town 

ems_sort= pd.DataFrame(tab.sort_values('EMS', ascending=False).iloc[0:10,0]).reset_index()
ems_sort
# Graphical representation

sns.barplot(y="twp", x="EMS",data=ems_sort,palette="Blues_d") 
plt.xlabel("Count")
plt.ylabel("Towns")
plt.title("Number of complaints (EMS) from each town")
plt.show()
# Number of complaints (fire) from each town 

fire_sort= pd.DataFrame(tab.sort_values('Fire', ascending=False).iloc[0:10,1]).reset_index()
fire_sort
# Graphical representation

sns.barplot(y="twp", x="Fire",data=fire_sort,palette="Blues_d") 
plt.xlabel("Count")
plt.ylabel("Towns")
plt.title("Number of complaints (fire) from each town")
plt.show()
# Number of complaints (traffic) from each town 

traffic_sort= pd.DataFrame(tab.sort_values('Traffic', ascending=False ).iloc[0:10,2]).reset_index()
traffic_sort
# Graphical representation

sns.barplot(y="twp", x="Traffic",data=traffic_sort,palette="Blues_d") 
plt.xlabel("Count")
plt.ylabel("Towns")
plt.title("Number of complaints (traffic) from each town")
plt.show()
EMS_city=d.loc[d['type'] == 'EMS']
EMS_city.head(5)
EMS_city.shape
Fire_city=d.loc[d['type'] == 'Fire']
Fire_city.head(5)
Fire_city.head(5)
Traffic_city=d.loc[d['type'] == 'Traffic']
Traffic_city.head(5)
Traffic_city.dtypes
Traffic_city['lat']=Traffic_city['lat'].astype('float64')
Traffic_city['lng']=Traffic_city['lng'].astype('float64')
location = Traffic_city['lat'].mean(), Traffic_city['lng'].mean()

locationlist = Traffic_city[['lat','lng']].values.tolist()
labels = Traffic_city['twp'].values.tolist()

#Empty map
import folium
m = folium.Map(location=location, zoom_start=14)
#Accesing the latitude
for point in range(1,100): 
    popup = folium.Popup(labels[point], parse_html=True)
    folium.Marker(locationlist[point], popup=popup).add_to(m)

m
