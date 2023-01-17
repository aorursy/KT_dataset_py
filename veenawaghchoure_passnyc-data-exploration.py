# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


# Any results you write to the current directory are saved as output.
school = pd.read_csv("../input/2016 School Explorer.csv")
school.head()
# school.columns.values
print(school.dtypes)
#changing variable SIE from object to float
school['School Income Estimate'] = school['School Income Estimate'] .str.replace('$',"")
school['School Income Estimate'] = school['School Income Estimate'] .str.replace(',',"")
school['School Income Estimate'] = school['School Income Estimate'].astype(float)
##Finding percentage of missing values in the data
percent_missing = 100*(pd.isnull(school).sum())/1272
percent_missing
print("before imputation\n",school['School Income Estimate'].describe())

#we can see in the variable school income estimate there are 31% missing values. 
#Missing value imputation needs to be done for that variable since dropping the missing values will 
#lead to loss of information.

school['School Income Estimate'] = school['School Income Estimate'].fillna(method = "bfill" )
pd.isnull(school['School Income Estimate']).sum()
print("\nAfter imputation\n",school['School Income Estimate'].describe())
###Detecting outliers in quantitative variables
school.boxplot(column = 'School Income Estimate')
Q1 = school['School Income Estimate'].quantile(0.25)
Q3 = school['School Income Estimate'].quantile(0.75)
IQR = Q3 - Q1
d = ((school['School Income Estimate'] < (Q1 - 1.5 * IQR)) | (school['School Income Estimate'] > (Q3 + 1.5 * IQR))).sum()
print("There are ",d*100/1272,"% outliers in the variable - School Income Estimate")
#there are 3.5 percent outliers in the column School Income Estimate.

#####For Economic need index
school.boxplot(column = 'Economic Need Index')
Q1 = school['Economic Need Index'].quantile(0.25)
Q3 = school['Economic Need Index'].quantile(0.75)
IQR = Q3 - Q1
e = ((school['Economic Need Index'] < (Q1 - 1.5 * IQR)) | (school['Economic Need Index'] > (Q3 + 1.5 * IQR))).sum()
print("There are ",e*100/1272,"% outliers in the variable - Economic Need Index")
#there are 1.3 percent outliers in the column Economic Need Index.

###Scatterplot for School Income Estimate and Economic Need Index
sns.jointplot(x = "School Income Estimate", y = "Economic Need Index" , data = school)
##From the scatterplot we can see that less is the school Income Estimate, the more is
# the mor is the Economic need index.
## it is negatively correlated
## barplot or different ethnicities
subset = school.loc[:,"Percent Asian":"Percent White"]
subset = subset.drop(columns = "Percent Black / Hispanic" )
# subset.head()
subset.columns.values

## converting percentages to float
subset["Percent Asian"] = subset["Percent Asian"].str.replace( '%',"")
subset['Percent Black'] = subset['Percent Black'].str.replace( '%',"")
subset["Percent Hispanic"] = subset["Percent Hispanic"].str.replace( '%',"")
subset["Percent White"] = subset["Percent White"].str.replace( '%',"")
subset
subset = (subset).astype(float)

subset["City"] = school["City"]
subset.head()

a = subset.groupby(by = subset['City']).sum()
print(type(a))
# city = a.index
# a['city'] = city
# print(a)
# a.columns.values
# a = a.transpose()
# a.head()

a = a.div(a.sum(1), axis=0)
a.plot(kind='bar', stacked=True)
# sns.set()
# a.T.plot(kind='bar', stacked=True)
###Collaborative Teachers Rating and Student Achievement Rating
cross = pd.crosstab(index = school['Collaborative Teachers Rating'], columns = school['Student Achievement Rating'])

cross = cross.div(cross.sum(1), axis=0)
cross.plot(kind = 'bar', stacked = True)


school["Percent ELL"]=school["Percent ELL"].str.replace("%","")
school["Percent ELL"]=school["Percent ELL"].astype(float)
school["Percent Asian"]=school["Percent Asian"].str.replace("%","")
school["Percent Asian"]=school["Percent Asian"].astype(float) 
school["Percent Black"]=school["Percent Black"].str.replace("%","")
school["Percent Black"]=school["Percent Black"].astype(float)
school["Percent White"]=school["Percent White"].str.replace("%","")  
school["Percent White"]=school["Percent White"].astype(float)
school["Student Attendance Rate"]=school["Student Attendance Rate"].str.replace("%","") 
school["Student Attendance Rate"]=school["Student Attendance Rate"].astype(float)
school["Percent of Students Chronically Absent"]=school["Percent of Students Chronically Absent"].str.replace("%","")
school["Percent of Students Chronically Absent"]=school["Percent of Students Chronically Absent"].astype(float)
school["Rigorous Instruction %"]=school["Rigorous Instruction %"].str.replace("%","")
school["Rigorous Instruction %"]=school["Rigorous Instruction %"].astype(float)
school["Collaborative Teachers %"]=school["Collaborative Teachers %"].str.replace("%","") 
school["Collaborative Teachers %"]=school["Collaborative Teachers %"].astype(float)
school["Supportive Environment %"]=school["Supportive Environment %"].str.replace("%","")
school["Supportive Environment %"]=school["Supportive Environment %"].astype(float)
school["Effective School Leadership %"]=school["Effective School Leadership %"].str.replace("%","") 
school["Effective School Leadership %"]=school["Effective School Leadership %"].astype(float) 
school["Strong Family-Community Ties %"]=school["Strong Family-Community Ties %"].str.replace("%","")
school["Strong Family-Community Ties %"]=school["Strong Family-Community Ties %"].astype(float)
school["Trust %"]=school["Trust %"].str.replace("%","").astype(float) 
 
      

#HeatMap to find correlation between the variables
import seaborn as sns
df = school.iloc[:,[16,17,19,20,21,22,23,24,25,26,28,30,32,34,36]]
#Correlation Matrix
corr = df.corr()
corr = (corr)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap="PuOr", center=0)
#From the heatmap, the following relationships are observed
#Economic Need Index and School Income estimate are negatively correlated
#The School with low income has more Economic Need Index
#Percent White and Economic Need Index are negatively correlated. This shows the 
#schools with more percent of white people has less Economic Need Index 
#There is assoication between Collobartive teachers and Effective School Leadership



import folium
import pandas as pd

lat = list(school["Latitude"])
lon = list(school["Longitude"])
elev = list(school["Economic Need Index"])

def color_producer(elevation):
    if elevation < 0.3:
        return 'green'
    elif 0.3 <= elevation < 0.7:
        return 'orange'
    else:
        return 'red'
map = folium.Map(location=[40.721834,-73.978766],tiles="Mapbox Bright",zoom_start=10) #Intialising basemap
fg = folium.FeatureGroup(name="My Map")
for lt, ln, el in zip(lat, lon, elev):
     fg.add_child(folium.CircleMarker(location=[lt, ln], radius = el, popup="Economic Need Index:"+str(el), 
     color=color_producer(el),fill_opacity=0.7))
map.add_child(fg)
map

elev1 = list(school["School Income Estimate"])
def color_producer1(elevation):
    if elevation < 50000:
        return 'green'
    elif 50000 <= elevation < 100000:
        return 'orange'
    else:
        return 'red'

m = folium.Map(location=[40.721834,-73.978766],tiles="Mapbox Bright",zoom_start=10)
fg = folium.FeatureGroup(name="My Map")
for lt, ln, el in zip(lat, lon, elev1):
     fg.add_child(folium.CircleMarker(location=[lt, ln], radius = 2, popup="School Income Estimate:"+str(el),
     color=color_producer1(el),fill_opacity=0.7))
m.add_child(fg)
# turn on layer control
m.add_child(folium.map.LayerControl())
m
