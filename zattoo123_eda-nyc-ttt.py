import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
a = train_df.iloc[:,1:].groupby(train_df.iloc[:,1:].columns.tolist(),as_index=False).size().reset_index().rename(columns={0:'count'})
a.sort_values(['count',],ascending=[False,]).iloc[:10,:]
del a
sns.distplot(train_df.trip_duration[train_df.trip_duration < 10000], color="m")
sns.distplot(train_df.trip_duration[(train_df.trip_duration > 10000) & (train_df.trip_duration < 100000)], color="m")
sns.distplot(train_df.trip_duration[(train_df.trip_duration > 100000) & (train_df.trip_duration < 4000000)], color="m")
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])

train_df['date_'] = train_df['pickup_datetime'].dt.date
fig, axes = plt.subplots(nrows=3, ncols=2)

fig.tight_layout()



train_df.loc[:,['date_','trip_duration']].groupby('date_').min().plot(title = 'min', ax=axes[0,0],figsize=(10,10), rot=90)

train_df.loc[:,['date_','trip_duration']].groupby('date_').median().plot(title = 'median', ax=axes[0,1],figsize=(10,10), rot=90)

train_df.loc[:,['date_','trip_duration']].groupby('date_').max().plot(title = 'max', ax=axes[1,0],figsize=(10,10), rot=90)

train_df.loc[:,['date_','trip_duration']].groupby('date_').mean().plot(title = 'mean', ax=axes[1,1],figsize=(10,10), rot=90)

train_df.loc[:,['date_','trip_duration']].groupby('date_').count().plot(title = 'count', ax=axes[2,0],figsize=(10,10), rot=90)
train_df['hour'] = train_df['pickup_datetime'].dt.hour
fig, axes = plt.subplots(nrows=2, ncols=2)

fig.tight_layout()



sns.boxplot(x="hour", y="trip_duration", data=train_df[(train_df.trip_duration < 1000)], ax=axes[0,0])

sns.boxplot(x="hour", y="trip_duration", data=train_df[(train_df.trip_duration < 1000000)], ax=axes[0,1])

train_df.loc[:,['hour','vendor_id']].groupby('hour').count().plot.bar(title = 'count', ax=axes[1,0],figsize=(10,10), rot=90)
fig, axes = plt.subplots(nrows=2, ncols=2)

fig.tight_layout()



train_df.loc[(train_df.trip_duration < 1000),['date_','trip_duration']].groupby('date_').count().plot(title = 'count', ax=axes[0,0],figsize=(10,10), rot=90)

train_df.loc[(train_df.trip_duration > 1000) & (train_df.trip_duration < 20000),['date_','trip_duration']].groupby('date_').count().plot(title = 'count', ax=axes[0,1],figsize=(10,10), rot=90)

train_df.loc[(train_df.trip_duration > 20000) & (train_df.trip_duration < 80000),['date_','trip_duration']].groupby('date_').count().plot(title = 'count', ax=axes[1,0],figsize=(10,10), rot=90)

train_df.loc[(train_df.trip_duration > 80000) & (train_df.trip_duration < 90000),['date_','trip_duration']].groupby('date_').count().plot(title = 'count', ax=axes[1,1],figsize=(10,10), rot=90)


pickup_df1 = train_df.loc[:,['pickup_latitude','pickup_longitude','id']].sample(n=1000)



graph1 = sns.jointplot(pickup_df1.pickup_longitude, pickup_df1.pickup_latitude,kind="hex", color="#4CB391")



dropoff_df1 = train_df.loc[:,['dropoff_latitude','dropoff_longitude','id']].sample(n=1000)



graph2 = sns.jointplot(dropoff_df1.dropoff_longitude, dropoff_df1.dropoff_latitude, kind="hex", color="#4CB391")

      
del pickup_df1, dropoff_df1
train_df['pickup_coord'] = train_df['pickup_latitude'].round(2).astype(str) + train_df['pickup_longitude'].round(2).astype(str) 



train_df['dropoff_coord'] = train_df['dropoff_latitude'].round(2).astype(str) + train_df['dropoff_longitude'].round(2).astype(str)



a = pd.pivot_table(train_df.ix[:,['pickup_coord','vendor_id']],index=['pickup_coord'],aggfunc=np.sum).sort_values(['vendor_id'],ascending=False).reset_index()[:30]



b = pd.pivot_table(train_df.ix[:,['dropoff_coord','vendor_id']],index=['dropoff_coord'],aggfunc=np.sum).sort_values(['vendor_id'],ascending=False).reset_index()[:30]



a =pd.DataFrame(a) 

a.columns=['coord','count_pu']



b =pd.DataFrame(b) 

b.columns=['coord','count_do']



a = a.merge(b, on = ['coord'])
ax = a.loc[:,['count_pu','count_do']].plot(kind='bar', rot=90)

ax.set_xticklabels(a.coord)
import folium
#To center map, take mean value of coordinates

stamen01 = folium.Map(location=[40.75092, -73.97349], tiles='Stamen Toner',

                    zoom_start=12)
feature_group = folium.FeatureGroup("cluster top coord")



for i in range(28):

    feature_group.add_child(folium.CircleMarker(location=[float(a.coord[i][:5])

                                                ,float(a.coord[i][5:])], 

                                                radius= a.count_pu[i]/10000,           

                                                popup=a.coord[i]))

stamen01.add_child(feature_group)
a = train_df.ix[:,['pickup_coord','trip_duration','pickup_longitude', 'pickup_latitude']].groupby('pickup_coord').median().sort_values(by='trip_duration',ascending =False).reset_index()[:50]
#To center map, take mean value of coordinates

stamen02 = folium.Map(location=[40.75092, -73.97349], tiles='Stamen Toner',

                    zoom_start=8)
feature_group = folium.FeatureGroup("cluster top coord")







for i in range(50):

    feature_group.add_child(folium.CircleMarker(location=[np.round(a.pickup_latitude[i],2)

                                                ,np.round(a.pickup_longitude[i],2)], 

                                                radius= a.trip_duration[i]/1000,           

                                                popup=a.pickup_coord[i]))







stamen02.add_child(feature_group)

del a, b
chord_source = pd.pivot_table(train_df.ix[:,['pickup_coord','dropoff_coord','vendor_id']],index=['pickup_coord','dropoff_coord'],aggfunc=np.sum)
chord_source =  chord_source.reset_index()

chord_source.rename(columns={'vendor_id':'link_count'},inplace =True)

chord_source = chord_source[chord_source.link_count >0]

chord_source =chord_source[chord_source.pickup_coord != chord_source.dropoff_coord]

chord_source =chord_source.sort_values(['link_count',], ascending=[False,])
from bokeh.charts import  Chord

from bokeh.charts import output_file,show
chord_from_df = Chord(chord_source.iloc[:200,:], source="pickup_coord", target="dropoff_coord", value="link_count")

output_file('chord-diagram-bokeh.html', mode="inline")

show(chord_from_df)
from IPython.core.display import HTML
#HTML('chord-diagram-bokeh.html') 

#Not possible to display within kernell limitation. Must be reproduced locally
del chord_from_df, chord_source