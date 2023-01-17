#Libraries for data cleaning and analysis

import pandas as pd

from pandas import DataFrame, Series

import numpy as np



#Libraries for plotting and data visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



#Libraries for plotting geographic data

import geopandas as gpd

from shapely.geometry import Point, Polygon

import folium
#Reading in the data

df = pd.read_csv('../input/montcoalert/911.csv')

df.head()
#This data contains very few null values

#423,909 entries

df.info()
#This dataset includes 141 unique incident titles

df['title'].nunique()
#This dataset includes 170 unique zipcodes

df['zip'].nunique()
#This dataset includes 68 unique townships for Montgomery, PA

df['twp'].nunique()
#Dropping filler column

df.drop(columns='e', inplace=True)



#Creating 'Reason' column based on the category of each incident (EMS, Fire, Traffic)

df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])



#Stripping the 'title' column of the identifier now stored in the 'Reason' column

df['title'] = df['title'].apply(lambda x:x.split(':')[1])
#Formatting the timeStamp column to allow for more flexible analysis by hour, day of week, month, and year



df['timeStamp'] = pd.to_datetime(df['timeStamp'])



#Creating a date column to groupby later

df['date'] = df['timeStamp'].apply(lambda date: date.date())



#'Year'

df['Year'] = df['timeStamp'].apply(lambda time: time.year)



#'Hour'

df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)



#'Month'

df['Month'] = df['timeStamp'].apply(lambda time: time.month)



#'Day of Week'

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)



daysDict={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day of Week'] = df['Day of Week'].map(daysDict)



#Month Name

monthsDict= {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

df['Month Name'] = df['Month'].map(monthsDict)



df.sample(5)
#Reordering columns for readaility

df = df[['timeStamp','twp','Reason', 'title', 'desc', 'Hour', 'Day of Week', 'Month', 'Month Name', 'Year','zip','addr', 'lat', 'lng','date']]



#Resetting the index to the 'timeStamp' column

df.set_index('timeStamp', inplace=True)

#Sorting the dataframe by date (earliest to most recent)

df.index= df.index.sort_values()

df.head()
#Finding the date of the first recorded call in the DataFrame

df.iloc[0].date
#Finding the date of the last recorded call in the DataFrame

df.iloc[-1].date
#Adjusting plot settings

plt.style.use('seaborn-poster')

plt.style.use('ggplot')
#Explanation of peak in late Janaury 2016

df['2016-01-20':'2016-01-30'].groupby('date').count()



#Dates of interest: 1/23/2016

"""Snowfall ranging from 15 to >30 inches across Montgomery County



Source: https://www.weather.gov/phi/01232016wss

...MONTGOMERY COUNTY...

   STOWE                 32.0  1104 AM  1/24  SOCIAL MEDIA

   EAGLEVILLE            31.0   748 AM  1/24  TRAINED SPOTTER

   LIMERICK              31.0   953 AM  1/24  SOCIAL MEDIA

   NORRISTOWN            30.0   820 AM  1/24  TRAINED SPOTTER

   BRYN MAWR             29.0   850 AM  1/24  SOCIAL MEDIA

   GREEN LANE            28.3   857 AM  1/24  TRAINED SPOTTER

   NORTH WALES           28.0   844 AM  1/24  TRAINED SPOTTER

   GRATERFORD            27.2   834 AM  1/24  TRAINED SPOTTER

   GILBERTSVILLE         27.0  1014 PM  1/23  TRAINED SPOTTER

   ROYERSFORD            26.7   517 AM  1/24  TRAINED SPOTTER

   KING OF PRUSSIA       26.1  1000 PM  1/23  TRAINED SPOTTER

   2 NNE POTTSTOWN       26.0   700 AM  1/24  COCORAHS

   COLLEGEVILLE          26.0   600 PM  1/23  SOCIAL MEDIA

   2 NW BLUE BELL        25.7   904 AM  1/24  COCORAHS

   MONTGOMERYVILLE       25.5   230 AM  1/24  TRAINED SPOTTER

   4 WNW HARLEYSVILLE    25.0   700 AM  1/24  COCORAHS

   HARLEYSVILLE          25.0  1000 AM  1/24  SOCIAL MEDIA

   1 W LIMERICK          24.5   900 AM  1/24  COCORAHS

   1 NNE COLLEGEVILLE    24.5   700 AM  1/24  COCORAHS

   WYNDMOOR              24.0   109 AM  1/24  PUBLIC

   SOUDERTON             23.5   700 PM  1/23  COCORAHS

   MAPLE GLEN            23.0   315 PM  1/23  TRAINED SPOTTER

   WYNNEWOOD             21.5   825 AM  1/24  TRAINED SPOTTER

   AMBLER                21.0   809 PM  1/23  TRAINED SPOTTER

   SSW HATFIELD          21.0   600 AM  1/24  COCORAHS

   2 SSW PENNSBURG       21.0   700 AM  1/24  COCORAHS

   2 SW HARLEYSVILLE     18.0   700 AM  1/24  COCORAHS

   GLASGOW               16.0  1104 AM  1/24  TRAINED SPOTTER"""
#Explanation of peak in early March 2018

df['2018-02-26':'2018-03-10'].groupby('date').count()



#Dates of interest: 3/2/2018 and 3/3/2018

"""

Snowfall ranging from 8 to 12 inches across Montgomery County, PA



Source:



Montgomery County

Maple Glen 		11.5

Bryn Mawr	11.0

Haverford	11.0

North Wales	10.0

Willow Grove 	10.0

Horsham	9.1

Hatboro 	9.0

Jenkintown	 8.6

King of Prussia	8.5

Plymouth Meeting	8.0

Lansdale	8.0 

Upper Gwynedd	7.8

Wynnewood 	7.7

Gilbertsville	6.1

Graterford 	5.3

"""
#November 2018 Explanation

df['2018-11-10':'2018-11-20'].groupby('date').count()



#Explanation of peak in mid-November 2018

"""

Snowfall around ~5 inches for Montgomery County



Source: https://patch.com/pennsylvania/doylestown/eastern-pa-snow-totals-town-town-nov-15-storm



Montgomery County:



Gilbertsville, 6.5 inches

Perkiomen Twp, 5.3 inches

Pottstown, 5.2 inches

King Of Prussia, 5.0 inches

North Wales, 4.9 inches

East Norriton Twp, 4.8 inches

Narberth, 4.5 inches

Plymouth Meeting, 4.5 inches

Lower Providence, 4.5 inches



"""
#Count of Type by Type

df['Reason'].value_counts().head(10)
#Creating a bar plot in Seaborn for the count of calls separated by type or the 'Reason' column

splot = sns.countplot(x='Reason', data=df, palette='viridis')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize=15)

plt.xticks(fontsize=15)

plt.ylabel('Count', fontsize=22)

plt.xlabel('Reason', fontsize=22)

plt.title('Count of 911 Calls by Reason', fontsize = 25)
#Creating a dataframe for the number of calls for each incident category. 

countsPerReason = df.groupby(['date','Reason']).count()['lat'].unstack(level=1)

countsPerReason['date'] = countsPerReason.index

countsPerReason.head()
#Plotting the frequency of calls over time, separated by type or the 'Reason' column

fig5, ax5 = plt.subplots()

reasons=['EMS', 'Fire', 'Traffic']

for reason in reasons:

    sns.lineplot(x='date', y=reason, data = countsPerReason, ax=ax5, linewidth=1.0, label=reason, palette='viridis')

plt.xlabel('Date', fontsize=15)

plt.ylabel('Count', fontsize=15)

plt.title('Frequency of Calls over Time by Type', fontsize=20)

plt.xticks(fontsize=12)
top20EMS = pd.DataFrame(df[df['Reason']=='EMS']['title'].value_counts().head(20))

top20EMS.rename(columns={'title':'Count'}, inplace=True)

top20EMS.head(5)

#Creating a bar chart of the top types of EMS 911 calls

fig, ax = plt.subplots(figsize=(25,10))

top = sns.barplot(x=top20EMS.index, y='Count', data=top20EMS, palette='viridis')

top.set_xticklabels(labels=top20EMS.index, rotation=30, ha='right', fontsize=15)

plt.xlabel('Reason', fontsize=25)

plt.ylabel('Count', fontsize=25)

plt.title('EMS Calls by Type', fontsize=30)
#Plotting the counts of different types of EMS calls throughout the week



#Nearly all EMS incident types decrease throughout the weekend.

#The most signficant decreases from weekday to weekend occur in the frequency of vehicle accidents and cardiac emergencies





top10EMSList = top20EMS.index.tolist()[0:10]

top10EMSList

fig, ax= plt.subplots()

for title in top10EMSList:

    tempDf = df[(df['Reason']=='EMS') & (df['title']== title)]

    tempDf.reset_index(inplace=True)

    tempDf = pd.DataFrame(tempDf.groupby('Day of Week').count()['lat'])

    tempDf.reset_index(inplace=True)

    daysDict={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

    revDaysDict = dict((v,k) for k,v in daysDict.items())

    tempDf['Order'] = tempDf['Day of Week'].map(revDaysDict)

    tempDf.sort_values(by='Order', inplace=True)

    tempDf.reset_index(inplace=True, drop=True)

    sns.lineplot(x='Day of Week', y='lat', data=tempDf, ax=ax, sort=False, label=title)

    

plt.xlabel('Day of Week', fontsize=20)

plt.ylabel('Count', fontsize=20)

plt.title('Frequency of EMS Calls by Day of Week', fontsize=25)

plt.legend()

ax.legend(bbox_to_anchor=(1.1, 1.05))
#Creating a bar chart for the top types of 'Fire' 911 Calls

top10Fire = pd.DataFrame(df[df['Reason']=='Fire']['title'].value_counts().head(10))

top10Fire.rename(columns={'title':'Count'}, inplace=True)

top10Fire.plot(kind='barh')

plt.xlabel('Count', fontsize=20)

plt.ylabel('Reason', fontsize=20)

plt.title('Fire Calls by Type', fontsize=25)
#Plotting the top 5 types of Traffic 911 Calls

top10Traffic =  pd.DataFrame(df[df['Reason']=='Traffic']['title'].value_counts().head(5))

top10Traffic.rename(columns = {'title':'Count'}, inplace=True)

top10Traffic.plot(kind='barh')

plt.xlabel('Count', fontsize=25)

plt.ylabel('Reason', fontsize=25)

plt.title('Traffic Calls by Type', fontsize=30)
#Making a dataframe of the count of Vehicle Accidents by Day of the week

vehAcc = df[(df['Reason']=='Traffic') & (df['title']==' VEHICLE ACCIDENT -')]

vehAccGb = pd.DataFrame(vehAcc.groupby('Day of Week').count()['lat'])

vehAccGb.reset_index(inplace=True)

daysDict={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

revDaysDict = dict((v,k) for k,v in daysDict.items())

vehAccGb['Order'] = vehAccGb['Day of Week'].map(revDaysDict)

vehAccGb.sort_values(by='Order', inplace=True)

vehAccGb.set_index('Order', inplace=True)

vehAccGb.reset_index(inplace=True)

vehAccGb.head(5)
#Plotting to see how the frequency of Vehicle Accidents changes throughout the week

fig,ax = plt.subplots()



#Dividing by the number of weeks covered in the data set (152) to give a weekly average of 'Traffic- Vehicle Accident' calls

sns.lineplot(x='Day of Week', y= (vehAccGb['lat']/152), data=vehAccGb, sort=False)

plt.xlabel('Day of Week', fontsize=20)

plt.ylabel('Count', fontsize=20)

plt.title('Frequency of Vehicle Accidents per Week', fontsize=25)
#Grouping the DataFrame by the counts of calls per day of the week and hour of the day

dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

dayHour.reset_index(inplace=True)

daysDict={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

revDaysDict = dict((v,k) for k,v in daysDict.items())

dayHour['Order'] = dayHour['Day of Week'].map(revDaysDict)

dayHour.sort_values(by='Order', inplace=True)

dayHour.reset_index(inplace=True, drop=True)

dayHour.set_index('Day of Week', inplace=True)

dayHour.drop('Order', axis=1, inplace=True)

dayHour.head(5)
#Plotting a heatmap of the frequency of calls by hour and day of week

fig, ax = plt.subplots()

sns.heatmap(data=dayHour, cmap= 'viridis', ax=ax)

ax.invert_yaxis()


titleVC = pd.DataFrame(df['title'].value_counts())

titleVC.head(5)
(98401 + 24081)/423909
#Grouping the data by the countsof calls per day of the week and month

dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()

dayMonth.reset_index(inplace=True)

daysDict={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

revDaysDict = dict((v,k) for k,v in daysDict.items())

dayMonth['Order'] = dayMonth['Day of Week'].map(revDaysDict)

dayMonth.sort_values(by='Order', inplace=True)

dayMonth.reset_index(inplace=True, drop=True)

dayMonth.set_index('Day of Week', inplace=True)

dayMonth.drop('Order', axis=1, inplace=True)

dayMonth.head(5)
#Plotting a heatmap of the frequency of calls by day of the week and month

fig, ax = plt.subplots()

sns.heatmap(dayMonth,cmap='viridis', ax=ax)

ax.invert_yaxis()
#Using the 10 townships with the highest numbers of calls to find which had the highest calls per person

townshipVC = pd.DataFrame(df['twp'].value_counts().head(10))





#Population data taken from 2010 US Census

#Source: https://www.census.gov/prod/cen2010/cph-2-40.pdf

townshipPop = pd.DataFrame.from_dict({'LOWER MERION':57825,

                          'ABINGTON':55310,

                          'NORRISTOWN':34324,

                          'UPPER MERION':28395,

                          'CHELTENHAM':36793,

                          'POTTSTOWN': 22377,

                          'UPPER MORELAND':24015,

                          'LOWER PROVIDENCE': 25436,

                          'PLYMOUTH':16525,

                          'HORSHAM':26147}, orient='index')

townshipPop.rename(columns={0:'Population'}, inplace=True)



#Megrging the two DataFrames and creating a columns for 'Calls per Person'

top10 = townshipVC.join(townshipPop)

top10['Calls per Person'] = top10['twp']/top10['Population']

top10.rename(columns={'twp':'# of Calls'}, inplace=True)





list = top10.index.tolist()

top10['Twp Name'] = top10.index

top10['Twp Name'].apply(str)

top10
#Plotting the calls per person by township

fig, ax = plt.subplots()

ax.set_xlabel('Township')

ax.set_ylabel('Calls per Person')

ax.set_title('Calls per Person by Township')

plt.setp( ax.xaxis.get_majorticklabels(), rotation=45 )

plt.xticks(horizontalalignment='right')

sns.lineplot(x='Twp Name', y='Calls per Person', data = top10,palette='Blues_d', ax=ax)
#Creating a geodataframe with a column for the paired X and Y coordinates for each 911 Call

del list

newDf = df[['lat','lng','Year']]

newDf['Coordinates'] = list(zip(newDf.lng, newDf.lat))

newDf['Coordinates'] = newDf['Coordinates'].apply(Point)

crs = {'init':'epsg:3651'}

gdf = gpd.GeoDataFrame(newDf, geometry='Coordinates', crs=crs)

gdf.head(5)
#Plotting all calls in the DataFrame on a Lat/Long Axis

gdf.plot()
#Removing the outliers from the dataset

from scipy import stats

df = df[(np.abs(stats.zscore(df[['lat','lng']])) < 3).all(axis=1)]

df.info
#Recreating the geodataframe with delimited dataset

newDf = df[['lat','lng','Year']]

newDf['Coordinates'] = list(zip(newDf.lng, newDf.lat))

newDf['Coordinates'] = newDf['Coordinates'].apply(Point)

crs = {'init':'epsg:3651'}

gdf = gpd.GeoDataFrame(newDf, geometry='Coordinates', crs=crs)

gdf.head(5)
#Replotting the GeoDataFrame on a Lat/Long Axis

gdf.plot(markersize=1)
#Creating a heatmap of 911 calls with GeoDataFrame

from scipy import ndimage

def heatmap(d, bins=(100,100), smoothing=1.3, cmap='jet'):

    def getx(pt):

        return pt.coords[0][0]



    def gety(pt):

        return pt.coords[0][1]



    x = list(d.geometry.apply(getx))

    y = list(d.geometry.apply(gety))

    heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins)

    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]



    logheatmap = np.log(heatmap)

    logheatmap[np.isneginf(logheatmap)] = 0

    logheatmap = ndimage.filters.gaussian_filter(logheatmap, smoothing, mode='nearest')

    

    plt.imshow(logheatmap, cmap=cmap, extent=extent)

    plt.colorbar()

    plt.gca().invert_yaxis()

    plt.show()

    

heatmap(gdf, bins=(100,100))
#Uploading a zip file of wtih geospatial data for Pennsylvania from census.gov

#Source: https://www.census.gov/geo/maps-data/data/cbf/cbf_cousub.html

newgdf = gpd.read_file('../input/pennsylvania-shape-file/cb_2017_42_cousub_500k.shp')

newgdf.plot()
#Using the County FIPS code (091) for Montgomery County, PA to plot data on top of county map

montgomeryCounty = newgdf[newgdf['COUNTYFP']== '091' ]

montgomeryCounty.plot()
#GeoDataFrame for montgomeryCounty only

montgomeryCounty.head(5)
#Converting 'NAME' columns of GeoDataFrame to str.upper to match string format of township column in 911 calls dataframe

montgomeryCounty['NAME'] = montgomeryCounty['NAME'].apply(lambda x: x.upper())

montgomeryCounty.head(5)
#Creating Geospatial JSON object to overlay on folium map

montgomeryCounty.to_file('montgomeryPA.geojson',driver='GeoJSON')
#Creating a

keyon = gpd.read_file('montgomeryPA.geojson')

keyon.set_index('NAME',inplace=True)

keyon.head()
#Calculating total number of incidents per district

twpdata2 = pd.DataFrame(df['twp'].value_counts().astype(float))

twpdata2.rename(columns={'twp':'Count'}, inplace=True)

twpdata2.head(10)
#Creating a new GeoDataFrame linking the counts from 'twpdata2' (above)

twpGeoData = keyon.join(twpdata2)

twpGeoData.reset_index(inplace=True)

twpGeoData.head(5)
#Editing the 'twpdata2' dataframe for readability

twpdata2.reset_index(inplace=True)

twpdata2.rename(columns={'index':'Township'},inplace=True)

twpdata2.head(5)
#Creating GeoJSON object of twpGeoData to overlay on folium map

twpGeoData.to_file('montgomeryCountyPA.geojson',driver='GeoJSON')
#Making the chloropleth map in folium

montgomeryChloropleth = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=9.4,tiles='Cartodb Positron')

folium.Choropleth(geo_data = 'montgomeryCountyPA.geojson',  

              data = twpdata2,

              columns = ['Township','Count'],

              key_on = 'feature.properties.index',

              fill_color = 'YlOrRd', 

              fill_opacity = 0.6, 

              line_opacity = 0.2,

              legend_name = 'Number of Calls').add_to(montgomeryChloropleth)

montgomeryChloropleth