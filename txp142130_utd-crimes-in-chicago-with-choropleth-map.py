import numpy as np # linear algebra

import pandas as pd # data processing

import folium

from folium import IFrame, FeatureGroup, LayerControl, Map, Marker, plugins

import seaborn as sns

import matplotlib.pyplot as plt



Chicago_COORDINATES = (41.895140898, -87.624255632)



#crimes1 = pd.read_csv('../input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)

crimes2 = pd.read_csv('../input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)

crimes3 = pd.read_csv('../input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)

crimes4 = pd.read_csv('../input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)

crimedata = pd.concat([crimes2, crimes3, crimes4], ignore_index=False, axis=0)



#Deleting dataframe as they are no longer needed.

del crimes2

del crimes3

del crimes4

#crimes4



crimedata.head()
crimedata.shape
crimedata.ID.duplicated().sum()
crimedata.duplicated().sum()
# Droping duplicate data

#(keep='first') allowed us to keep the first duplicate data in the dataframe and remove any duplicate data found after it.



crimedata.drop_duplicates(subset=None, keep='first', inplace=True)

#crimedata.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)

crimedata.shape
null_data = crimedata[crimedata.isnull().any(axis=1)]

null_data.head(5) 
# Droping column

crimedata = crimedata.drop(columns=['Unnamed: 0', 'Case Number', 'Block', 'IUCR', 'Arrest',

                                    'Domestic', 'Beat', 'Updated On', 'FBI Code', 'X Coordinate', 'Y Coordinate', 

                                    'Latitude', 'Longitude', 'Location'], axis = 1)

#'Location Description'

crimedata.tail()
percent_missing = crimedata.isnull().sum()/ len(crimedata) * 100

percent_missing
crimedata = crimedata.dropna()

crimedata.isnull().sum()
crimedata.tail()
#getting rid of decimal in District, Ward and Community Area and turning them into string type.

crimedata[['District', 'Ward','Community Area']] = crimedata[['District', 'Ward','Community Area']].astype('int')

crimedata[['District', 'Ward','Community Area']] = crimedata[['District', 'Ward','Community Area']].astype('str')

crimedata.head()
# I decided to go with every 2 year for the Chicago community areas

#crimedata2005 = crimedata[crimedata["Year"]==2005]

crimedata2006 = crimedata[crimedata["Year"]==2006]

#crimedata2007 = crimedata[crimedata["Year"]==2007]

crimedata2008 = crimedata[crimedata["Year"]==2008]

#crimedata2009 = crimedata[crimedata["Year"]==2009]

crimedata2010 = crimedata[crimedata["Year"]==2010]

#crimedata2011 = crimedata[crimedata["Year"]==2011]

crimedata2012 = crimedata[crimedata["Year"]==2012]

#crimedata2013 = crimedata[crimedata["Year"]==2013]

crimedata2014 = crimedata[crimedata["Year"]==2014]

crimedata2015 = crimedata[crimedata["Year"]==2015]

crimedata2016 = crimedata[crimedata["Year"]==2016]
crimedata2006.columns = crimedata2006.columns.str.strip().str.lower().str.replace(' ', '_')

crimedata2008.columns = crimedata2008.columns.str.strip().str.lower().str.replace(' ', '_')

crimedata2010.columns = crimedata2010.columns.str.strip().str.lower().str.replace(' ', '_')

crimedata2012.columns = crimedata2012.columns.str.strip().str.lower().str.replace(' ', '_')

crimedata2014.columns = crimedata2014.columns.str.strip().str.lower().str.replace(' ', '_')

crimedata2016.columns = crimedata2016.columns.str.strip().str.lower().str.replace(' ', '_')
crimedata2016.head(5)
#definition of the boundaries in the map

district_geo = r'../input/boundaries-wards/Boundaries_Wards.geojson'



#calculating total number of incidents per district for 2016

WardData2016 = pd.DataFrame(crimedata2016['ward'].value_counts().astype(float))

WardData2016.to_json('Ward_Map.json')

WardData2016 = WardData2016.reset_index()

WardData2016.columns = ['ward', 'Crime_Count']

 

#creating choropleth map for Chicago District 2016

map1 = folium.Map(location=Chicago_COORDINATES, zoom_start=11)

map1.choropleth(geo_data = district_geo, 

                #data_out = 'Ward_Map.json', 

                data = WardData2016,

                columns = ['ward', 'Crime_Count'],

                key_on = 'feature.properties.ward',

                fill_color = 'YlOrRd', 

                fill_opacity = 0.7, 

                line_opacity = 0.2,

                threshold_scale=[0, 4000, 8000, 12000, 16000, 20000],

                legend_name = 'Number of incidents per police ward 2016')



#WardData2016.sort_values('Ward')
map1
#definition of the boundaries in the map

district_geo = r'../input/chicago-police-district/Boundaries_Police_Districts.geojson'



district_data = pd.DataFrame(crimedata2016['district'].value_counts().astype(float))

district_data.to_json('District_Map.json')

district_data = district_data.reset_index()

district_data.columns = ['district', 'Crime_Count']



#creation of the choropleth

map2 = folium.Map(location=Chicago_COORDINATES, zoom_start=11)

map2.choropleth(geo_data = district_geo,  

                data = district_data,

                columns = ['district', 'Crime_Count'],

                key_on = "feature.properties.dist_num",

                fill_color = 'YlOrRd', 

                fill_opacity = 0.7, 

                line_opacity = 0.2,

                threshold_scale=[0, 4000, 8000, 12000, 16000, 20000],

                legend_name = 'Number of incidents per district 2016')
map2
from IPython.display import display_html

def display_side_by_side(*args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline"'),raw=True)   

#display_side_by_side(WardData2016.sort_values('Crime_Count', ascending=True).tail(5),district_data.sort_values('Crime_Count', ascending=True).tail(5))
WardData2016[['ward']] = WardData2016[['ward']].astype('int')

district_data[['district']] = district_data[['district']].astype('int')

display_side_by_side(WardData2016.sort_values('ward', ascending=True),district_data.sort_values('district', ascending=True))
Total = district_data['Crime_Count'].sum()

print (Total)
Total = WardData2016['Crime_Count'].sum()

print (Total)
sns.countplot(x='Year',data=crimedata, color=('BLUE'))

fig = plt.gcf()

plt.ylabel('No of Crimes')

fig.set_size_inches(15,7)



plt.show()
Community_Areas_geo = r'../input/chicago-community-areas/Chicago_Community_Areas.geojson'

# Community_Areas map 2016

Community_Areas_data2016 = pd.DataFrame(crimedata2016['community_area'].value_counts().astype(float))

Community_Areas_data2016.to_json('Community_Area_Map2016.json')

Community_Areas_data2016 = Community_Areas_data2016.reset_index()

Community_Areas_data2016.columns = ['community_area', 'Crime_Count']





map2016 = folium.Map(location=Chicago_COORDINATES, zoom_start=11)

#map2016.add_child(feature_group)

#map8.add_children(folium.map.LayerControl())



map2016.choropleth(geo_data = Community_Areas_geo,

                data = Community_Areas_data2016,

                columns = ['community_area', 'Crime_Count'],

                key_on = "feature.properties.area_numbe",

                fill_color = 'YlOrRd', 

                fill_opacity = 0.7, 

                line_opacity = 0.2,

                threshold_scale=[0, 4000, 8000, 12000, 16000, 20000],

                legend_name = 'Number of incidents per community area 2016')

               



folium.TileLayer('cartodbdark_matter').add_to(map2016)

folium.TileLayer('Stamen Terrain').add_to(map2016)

folium.TileLayer('Stamen Toner').add_to(map2016)

folium.TileLayer('Mapbox Bright').add_to(map2016)





#Marker coordnate for each Comunity area



# Next time use loop, you dummy

# you you Super Dummy

feature_group = FeatureGroup(name='Comunity area number')

feature_group.add_child(Marker([42.01,-87.67],'Comunity area 1, ROGERS PARK'))

feature_group.add_child(Marker([42.0, -87.70],'Comunity area 2, WEST RIDGE'))

feature_group.add_child(Marker([41.965,-87.655],'Comunity area 3, UPTOWN'))

feature_group.add_child(Marker([41.975, -87.685],'Comunity area 4, LINCOLN SQUARE'))

feature_group.add_child(Marker([41.95, -87.685],'Comunity area 5, NORTH CENTER'))

feature_group.add_child(Marker([41.94, -87.655],'Comunity area 6, LAKE VIEW'))

feature_group.add_child(Marker([41.92, -87.655],'Comunity area 7, LINCOLN PARK'))

feature_group.add_child(Marker([41.9, -87.632],'Comunity area 8, NEAR NORTH SIDE'))

feature_group.add_child(Marker([42.006, -87.815],'Comunity area 9, EDISON PARK'))

feature_group.add_child(Marker([41.987, -87.8],'Comunity area 10, NORWOOD PARK'))

feature_group.add_child(Marker([41.98, -87.769],'Comunity area 11, JEFFERSON PARK'))

feature_group.add_child(Marker([41.987, -87.752],'Comunity area 12, FOREST GLEN'))

feature_group.add_child(Marker([41.985, -87.72],'Comunity area 13, NORTH PARK'))

feature_group.add_child(Marker([41.965, -87.72],'Comunity area 14, ALBANY PARK'))

feature_group.add_child(Marker([41.95, -87.764],'Comunity area 15, PORTAGE PARK'))

feature_group.add_child(Marker([41.954, -87.725],'Comunity area 16, IRVING PARK'))

feature_group.add_child(Marker([41.945, -87.808],'Comunity area 17, DUNNING'))

feature_group.add_child(Marker([41.927, -87.8],'Comunity area 18, MONTCLARE'))

feature_group.add_child(Marker([41.925, -87.765],'Comunity area 19, BELMONT CRAGIN'))

feature_group.add_child(Marker([41.925, -87.73501],'Comunity area 20, HERMOSA'))

feature_group.add_child(Marker([41.938, -87.71],'Comunity area 21, AVONDALE'))

feature_group.add_child(Marker([41.923, -87.7],'Comunity area 22, LOGAN SQUARE'))

feature_group.add_child(Marker([41.9, -87.725],'Comunity area 23, HUMBOLDT PARK'))

feature_group.add_child(Marker([41.9, -87.685],'Comunity area 24, WEST TOWN'))

feature_group.add_child(Marker([41.89, -87.761],'Comunity area 25, AUSTIN'))

feature_group.add_child(Marker([41.878, -87.729],'Comunity area 26, WEST GARFIELD PARK'))

feature_group.add_child(Marker([41.878, -87.705],'Comunity area 27, EAST GARFIELD PARK'))

feature_group.add_child(Marker([41.874, -87.665],'Comunity area 28, NEAR WEST SIDE'))

feature_group.add_child(Marker([41.861, -87.714],'Comunity area 29, NORTH LAWNDALE'))

feature_group.add_child(Marker([41.84, -87.714],'Comunity area 30, SOUTH LAWNDALE'))

feature_group.add_child(Marker([41.85, -87.664],'Comunity area 31, LOWER WEST SIDE'))

feature_group.add_child(Marker([41.876, -87.627],'Comunity area 32, LOOP'))

feature_group.add_child(Marker([41.8555, -87.6199],'Comunity area 33, NEAR SOUTH SIDE'))

feature_group.add_child(Marker([41.84, -87.633],'Comunity area 34, ARMOUR SQUARE'))

feature_group.add_child(Marker([41.834, -87.6199],'Comunity area 35, DOUGLA'))

feature_group.add_child(Marker([41.824, -87.602],'Comunity area 36, OAKLAND'))

feature_group.add_child(Marker([41.811, -87.632],'Comunity area 37, FULLER PARK'))

feature_group.add_child(Marker([41.811, -87.617],'Comunity area 38, GRAND BOULEVARD'))

feature_group.add_child(Marker([41.809, -87.595],'Comunity area 39, KENWOOD'))

feature_group.add_child(Marker([41.792, -87.617],'Comunity area 40, WASHINGTON PARK'))

feature_group.add_child(Marker([41.792, -87.595],'Comunity area 41, HYDE PARK'))

feature_group.add_child(Marker([41.78, -87.595],'Comunity area 42, WOODLAWN'))

feature_group.add_child(Marker([41.763, -87.575],'Comunity area 43, SOUTH SHORE'))

feature_group.add_child(Marker([41.738, -87.615],'Comunity area 44, CHATHAM'))

feature_group.add_child(Marker([41.742, -87.589],'Comunity area 45, AVALON PARK'))

feature_group.add_child(Marker([41.739, -87.548],'Comunity area 46, SOUTH CHICAGO'))

feature_group.add_child(Marker([41.728, -87.597],'Comunity area 47, BURNSIDE'))

feature_group.add_child(Marker([41.73, -87.575],'Comunity area 48, CALUMET HEIGHTS'))

feature_group.add_child(Marker([41.709, -87.619],'Comunity area 49, ROSELAND'))

feature_group.add_child(Marker([41.703, -87.598],'Comunity area 50, PULLMAN'))

feature_group.add_child(Marker([41.692, -87.568],'Comunity area 51, SOUTH DEERING'))

feature_group.add_child(Marker([41.71, -87.535],'Comunity area 52, EAST SIDE'))

feature_group.add_child(Marker([41.672, -87.628],'Comunity area 53, WEST PULLMAN'))

feature_group.add_child(Marker([41.658, -87.603],'Comunity area 54, RIVERDALE'))

feature_group.add_child(Marker([41.65, -87.54],'Comunity area 55, HEGEWISCH'))

feature_group.add_child(Marker([41.792, -87.77],'Comunity area 56, GARFIELD RIDGE'))

feature_group.add_child(Marker([41.809, -87.726],'Comunity area 57, ARCHER HEIGHTS'))

feature_group.add_child(Marker([41.815, -87.70],'Comunity area 58, BRIGHTON PARK'))

feature_group.add_child(Marker([41.83, -87.672],'Comunity area 59, MCKINLEY PARK'))

feature_group.add_child(Marker([41.836, -87.648],'Comunity area 60, BRIDGEPORT'))

feature_group.add_child(Marker([41.809, -87.657],'Comunity area 61, NEW CITY'))

feature_group.add_child(Marker([41.792, -87.726],'Comunity area 62, WEST ELSDON'))

feature_group.add_child(Marker([41.795, -87.695],'Comunity area 63, GAGE PARK'))

feature_group.add_child(Marker([41.778, -87.77],'Comunity area 64, CLEARING'))

feature_group.add_child(Marker([41.77, -87.726],'Comunity area 65, WEST LAWN'))

feature_group.add_child(Marker([41.77, -87.695],'Comunity area 66, CHICAGO LAWN'))

feature_group.add_child(Marker([41.775, -87.665],'Comunity area 67, WEST ENGLEWOOD'))

feature_group.add_child(Marker([41.775, -87.644],'Comunity area 68, ENGLEWOOD'))

feature_group.add_child(Marker([41.764, -87.622],'Comunity area 69, GREATER GRAND CROSSING'))

feature_group.add_child(Marker([41.744, -87.708],'Comunity area 70, ASHBURN'))

feature_group.add_child(Marker([41.742, -87.658],'Comunity area 71, AUBURN GRESHAM'))

feature_group.add_child(Marker([41.716, -87.673],'Comunity area 72, BEVERLY'))

feature_group.add_child(Marker([41.716, -87.648],'Comunity area 73, WASHINGTON HEIGHTS'))

feature_group.add_child(Marker([41.694, -87.708],'Comunity area 74, MOUNT GREENWOOD'))

feature_group.add_child(Marker([41.688, -87.67],'Comunity area 75, MORGAN PARK'))

feature_group.add_child(Marker([41.98, -87.91],'Comunity area 76, OHARE'))

feature_group.add_child(Marker([41.985, -87.665],'Comunity area 77, EDGEWATER'))



map2016.add_child(feature_group)

map2016.add_child(folium.map.LayerControl())

#map2016

crimedata['Date'] = pd.to_datetime(crimedata['Date'],format='%m/%d/%Y %I:%M:%S %p')
import calendar

crimedata['Month']=(crimedata['Date'].dt.month).apply(lambda x: calendar.month_abbr[x])
crimedata['Month'] = pd.Categorical(crimedata['Month'] , categories=['Jan','Feb','Mar','Apr','May',

                                                'Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ordered=True)



months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
crimedata.head(5)
crimedata.groupby(['Month','Year'])['ID'].count().unstack().plot(marker='o', figsize=(15,10))

plt.xticks(np.arange(12),months)

plt.ylabel('No of Crimes')



plt.show()
sns.set(rc={'figure.figsize':(15,10)})

sns.countplot(y='Primary Type',data=crimedata,order=crimedata['Primary Type'].value_counts().index, color=('BLUE'))

plt.xticks(rotation='vertical')

plt.xlabel('No of Crimes')

plt.ylabel('Type of Crimes')

plt.show()
df_crime=crimedata[(crimedata['Primary Type']=='THEFT')|(crimedata['Primary Type']=='BATTERY')|

                 (crimedata['Primary Type']=='CRIMINAL DAMAGE')|(crimedata['Primary Type']=='NARCOTICS')|

                 (crimedata['Primary Type']=='BURGLARY')|(crimedata['Primary Type']=='ASSAULT')]
df_crime.groupby([df_crime['Date'].dt.hour,'Primary Type',])['ID'].count().unstack().plot(marker='o')

plt.ylabel('Number of Crimes')

plt.xlabel('Hours of the day')

plt.xticks(np.arange(24))

plt.show()
df_drug = crimedata[crimedata['Primary Type'] == 'NARCOTICS']
plt.figure(figsize = (15, 12))

sns.countplot(y = df_drug['Description'],color=("Blue"))

plt.xlabel('Number of Crimes')

plt.ylabel('Type of Crimes')
df_narcotic=crimedata[(crimedata['Description']=='POSS: CANNABIS 30GMS OR LESS')|(crimedata['Description']=='POSS: HEROIN(WHITE)')|

                    (crimedata['Description']=='POSS: CRACK')|(crimedata['Description']=='POSS: CANNABIS MORE THAN 30GMS')]
df_narcotic.groupby([df_narcotic['Date'].dt.year,'Description',])['ID'].count().unstack().plot(marker='o')

plt.ylabel('Number of Crimes')

plt.xlabel('Year')

#plt.xticks(np.arange(1))

plt.show()


na2016 = df_narcotic[df_narcotic["Year"]==2016]

na2016.columns = na2016.columns.str.strip().str.lower().str.replace(' ', '_')
na2016.shape
# Community_Areas map 2016

Community_Areas_data2016 = pd.DataFrame(na2016['community_area'].value_counts().astype(float))

Community_Areas_data2016.to_json('Community_Area_nMap2016.json')

Community_Areas_data2016 = Community_Areas_data2016.reset_index()

Community_Areas_data2016.columns = ['community_area', 'Crime_Count']





nmap2016 = folium.Map(location=Chicago_COORDINATES, zoom_start=11)

#map2016.add_child(feature_group)

#map8.add_children(folium.map.LayerControl())



nmap2016.choropleth(geo_data = Community_Areas_geo,

                data = Community_Areas_data2016,

                name='choropleth',

                columns = ['community_area', 'Crime_Count'],

                key_on = "feature.properties.area_numbe",

                fill_color = 'YlGn', 

                fill_opacity = 0.7, 

                line_opacity = 0.2,

                threshold_scale=[0, 500, 1000, 1500, 2000, 2500],

                legend_name = 'Number of incidents per community area 2016')

               



folium.TileLayer('cartodbdark_matter').add_to(nmap2016)

folium.TileLayer('Stamen Terrain').add_to(nmap2016)

folium.TileLayer('Stamen Toner').add_to(nmap2016)

folium.TileLayer('Mapbox Bright').add_to(nmap2016)



nmap2016.add_child(feature_group)

nmap2016.add_child(folium.map.LayerControl())

#map2016