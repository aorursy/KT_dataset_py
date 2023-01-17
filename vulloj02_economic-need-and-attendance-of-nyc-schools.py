#Importing libraries for data cleaning and analysis

import pandas as pd

import numpy as np

import math

import scipy.stats as stats



#Libraries for data visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#Libraries for plotting geographic data

import geopandas as gpd

from shapely.geometry import Point, Polygon

import folium
#Reading in the demographic data of each public school in New York City into a pandas DataFrame

demo = pd.read_csv('../input/2013_-_2018_Demographic_Snapshot_School.csv')

demo.info()
#Dropping the columns specifying the enrollment per grade since enrollment and the percentage of students absent will be aggregated by school

demo.drop(labels=['Grade PK (Half Day & Full Day)','Grade K', 'Grade 1', 'Grade 2',

       'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Grade 7', 'Grade 8',

       'Grade 9', 'Grade 10', 'Grade 11', 'Grade 12'], axis=1, inplace=True)

demo.columns
demo.head(5)
#Parsing the first two digits of the DBN to find the school district of each school

demo['School District'] = demo['DBN'].str.slice(0, 2)
#Getting rid of the leading 0 for each school district number to better match how school districts are labeled in a later GeoJSON file

def leadingZero(row):

    if row[0] == '0':

        return row[1:]

    else:

        return row



demo['School District'] = demo['School District'].apply(leadingZero)

demo['School District'].value_counts().sort_values()
#Deleting the rows from the 2013-2014 school year because they do not contain data for Economic Need Index

demo = demo[demo['Year']!= '2013-14']

demo.shape
#Converting the string values of the Economic Need Index into a float64 object

demo['Economic Need Index'] = demo['Economic Need Index'].str.rstrip('%').astype('float') / 100.0
#Average Economic Need Index of all NYC schools from 2014-2018

demo['Economic Need Index'].describe()
demo['DBN'] = demo['DBN'].astype(str)
#Importing json overlay of geographic boundaries of NYC school districts

import json

geo_json_data = json.load(open('../input/School Districts.geojson'))
#Creating a folium map object to overlay the NYC School District GeoJSON file

sdENI = folium.Map(location=[40.652922, -73.984353], 

                    zoom_start=9.5, tiles='Cartodb Positron')



#Creating a choropleth folium map

folium.Choropleth(geo_data = geo_json_data,  

              data = demo.groupby('School District').mean().reset_index(),

              columns = ['School District', 'Economic Need Index'],

              key_on = 'feature.properties.school_dist',

              fill_color = 'YlOrRd', 

              fill_opacity = 0.6, 

              line_opacity = 0.2,

              legend_name = 'Economic Need Index').add_to(sdENI)

sdENI.save(outfile='sdENI.html')

from IPython.display import IFrame

IFrame('sdENI.html', width=500, height=550)
#Creating a dataframe with the aggregated demographic information for each school

demoGrouped = demo.groupby('DBN').sum().loc(axis=1)['Total Enrollment', '# Female', '# Male', '# Asian', '# Black', '# Hispanic',

       '# Multiple Race Categories Not Represented',

       '# White',

       '# Students with Disabilities',

       '# English Language Learners',

       '# Poverty']



#Finding out what proportion of each school's enrollment

demoList = ['# Female', '# Male', '# Asian', '# Black', '# Hispanic',

       '# Multiple Race Categories Not Represented',

       '# White',

       '# Students with Disabilities',

       '# English Language Learners',

       '# Poverty']



demoProportion  = pd.DataFrame()



#Making a dataframe of the proportion for each statistic rather than the 

for demoColumn in demoList:

    demoProportion['%'+ demoColumn.lstrip('#')] = demoGrouped[demoColumn]/demoGrouped['Total Enrollment']



#Creating an additional column summarizing the percentage of black and hispanic students for each school    

demoProportion['% Black/Hispanic'] = demoProportion['% Black'] + demoProportion['% Hispanic']

demoProportion.head(5)



demoProportion
#The New York Department of Educaiton defines a racially representative school as one where Black and Hispanic Students make up >50% of students



#Creating a function to identify racially representative schools in the dataframe

def racially_diverse(row):

    if row['% Black/Hispanic'] >= 0.50:

        return'Yes'

    else:

       return'No'



demoProportion['Racially Representative?'] = demoProportion.apply(racially_diverse, axis=1)

demoProportion.head(5)
demoProportion['Racially Representative?'].value_counts()
demo.set_index('DBN', inplace=True)
#Aggregating the demographic information by school district

demoDistricts = demoProportion.join(demo.loc(axis=1)['School District'])

demoDistricts.groupby('School District').mean().reset_index()
#Creating a folium map object to overlay the NYC School District GeoJSON file

sdRace = folium.Map(location=[40.652922, -73.984353], 

                    zoom_start=9.5, tiles='Cartodb Positron')



#Creating a choropleth folium map

folium.Choropleth(geo_data = geo_json_data,  

              data = demoDistricts.groupby('School District').mean().reset_index(),

              columns = ['School District', '% Black/Hispanic'],

              key_on = 'feature.properties.school_dist',

              fill_color = 'YlOrRd', 

              fill_opacity = 0.6, 

              line_opacity = 0.2,

              legend_name = '% Black/Hispanic').add_to(sdRace)

sdRace.save(outfile='sdRace.html')

IFrame('sdRace.html', width=500, height=550)
#Creating seperate dataframes by year for hypothesis testing and plotting



#'2014-15'

demo2014 = demo[demo['Year'] == '2014-15']



#'2015-16'

demo2015 = demo[demo['Year'] == '2015-16']



#'2016-17'

demo2016 = demo[demo['Year'] == '2016-17']



#'2017-18'

demo2017 = demo[demo['Year'] == '2017-18']
#Plotting the distribution of Economic Need Index by school year

fig4, ax10 = plt.subplots()



#Making a list of the dataframes

dataframes = [demo2014, demo2015, demo2016, demo2017]



plt.rcParams['patch.edgecolor'] = 'black'

for dataframe in dataframes:

    sns.distplot(dataframe['Economic Need Index'], kde = False, ax = ax10, label = dataframe['Year'])

ax10.legend()

#Graph of ENI over time with confidence intervals

import math

import scipy.stats as stats



#Making a list of the school years

schoolYears = ['2014-15','2015-16','2016-17','2017-18']



#Previously defined list of demographic dataframes

dataframes = [demo2014, demo2015, demo2016, demo2017]



sample_means = []

intervals = []



for dataframe in dataframes:

    sample_size = len(dataframe['Economic Need Index'])

    sample_mean = dataframe['Economic Need Index'].mean()

    sample_means.append(sample_mean)



    z_critical = stats.norm.ppf(q = 0.975)  # Get the z-critical value*         



    pop_stdev = dataframe['Economic Need Index'].std()  # Get the population standard deviation





    margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size))



    confidence_interval = (sample_mean - margin_of_error,

                           sample_mean + margin_of_error)  

    

    intervals.append(confidence_interval)
#Plotting the 95% confidence interval for each school year's ENI

fig3 , ax3 = plt.subplots(figsize=(4,4))

plt.style.use('ggplot')

plt.errorbar(x=schoolYears, 

             y=sample_means, 

             yerr=[(top-bot)/2 for top,bot in intervals],

             capsize=15)



ax3.set_xlabel('School Year', fontsize=13)

ax3.set_ylabel('Mean', fontsize=13)
demo2016.reset_index(inplace=True)

demo2017.reset_index(inplace=True)
demo2017['DBN'] = demo2017['DBN'].astype(str)

demo2016['DBN'] = demo2016['DBN'].astype(str)
#Inner join on 2016 and 2017 dataframes to allow for a pairwise difference of means testing

demoTest = pd.merge(demo2016, demo2017, how='inner', on='DBN', suffixes=('_2016', '_2017'))

demoTest.info()
demoTest['Economic Need Index_2016'].describe()
demoTest['Economic Need Index_2017'].describe()
#Hypothesis Testing of mean ENI recent year vs last year

stats.ttest_rel(a = demoTest['Economic Need Index_2016'],

                b = demoTest['Economic Need Index_2017'])
#Plotting the distribution of the proportion of students in poverty for those schools that are racially representative and those that aren't

fig4, ax7 = plt.subplots()

demoProportion[demoProportion['Racially Representative?']=='Yes']['% Poverty'].hist(label='Diverse', ax=ax7, edgecolor='black')

demoProportion[demoProportion['Racially Representative?']=='No']['% Poverty'].hist(label='Not Diverse', ax=ax7, edgecolor='black')



plt.legend()
#Boxplot of the '% Poverty'  between Racially Representative vs Non Racially Representative Schools

sns.boxplot(x='Racially Representative?', y='% Poverty', data=demoProportion)
#Reading in 2017 Attendance Data

att2017 = pd.read_csv('../input/2017-2018_Monthly_Attendance.csv')

attFinal = att2017.groupby('School').sum()

attFinal['Total']= attFinal['Absent'] + attFinal['Present']

attFinal['Percentage Absent'] = attFinal['Absent']/attFinal['Total']

attFinal
#Distirbution of Absent Rate

attFinal['Percentage Absent'].hist(bins=50, edgecolor='black')
#Joining the dataframe with attendance records with the dataframe of the demographics of each school

demoAbsent = demo.reset_index().rename(columns={'DBN':'School'}).join(attFinal['Percentage Absent'], on='School').groupby('School').mean()

demoAbsent
demoAbsent['Percentage Absent'].describe()
#Economically Stratified is an ENI over .774657

demoStratified = demoAbsent[demoAbsent['Economic Need Index'] >= 0.774657]

demoNorm = demoAbsent[demoAbsent['Economic Need Index'] < 0.774657]

demoNorm.head(5)
demoNorm = demoNorm[demoNorm['Percentage Absent'].notnull()]

demoNorm['Percentage Absent'].describe()
demoStratified = demoStratified[demoStratified['Percentage Absent'].notnull()]

demoStratified['Percentage Absent'].describe()
demoNorm['Percentage Absent'].hist(bins=50, edgecolor='black', label='Norm', alpha=0.6)

demoStratified['Percentage Absent'].hist(bins=50, edgecolor='black', label='Strat',alpha=0.6)

plt.legend()
stats.ttest_ind(a= demoNorm['Percentage Absent'],

                b= demoStratified['Percentage Absent'],

                equal_var=False)    # Assume samples have equal variance?
#Attendance of racially diverse schools (Hypothesis Test)

#Do schools with >50% hispanic and latino students have the same attendance as population



demoProportion = demoProportion.join(demoAbsent['Percentage Absent'])

demoProportion
demoDiverse = demoProportion[demoProportion['% Black/Hispanic'] >= 0.50]

demoNotDiverse = demoProportion[demoProportion['% Black/Hispanic'] < 0.50]
demoDiverse['Percentage Absent'].describe()
demoNotDiverse['Percentage Absent'].describe()
demoNotDiverse['Percentage Absent'].hist(bins=30, edgecolor='black', label='Not Diverse',alpha=0.9)

demoDiverse['Percentage Absent'].hist(bins=30, edgecolor='black', label='Diverse',alpha=0.3)

plt.legend()
#Confidence Intervals

import math



dataframes = [demoNotDiverse, demoDiverse]

categoryList = ['Not Diverse', 'Diverse']

sample_means2 = []

intervals2 = []



for dataframe in dataframes:

    sample_size = len(dataframe['Percentage Absent'])

    sample_mean = dataframe['Percentage Absent'].mean()

    sample_means2.append(sample_mean)



    z_critical = stats.norm.ppf(q = 0.975)  # Get the z-critical value*         



    pop_stdev = dataframe['Percentage Absent'].std()  # Get the population standard deviation





    margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size))



    confidence_interval = (sample_mean - margin_of_error,

                           sample_mean + margin_of_error)  

    

    intervals2.append(confidence_interval)
#Plotting the 95% confidence for each school year's ENI

fig5 , ax11 = plt.subplots(figsize=(4,8))

plt.style.use('ggplot')

plt.errorbar(x=categoryList, 

             y=sample_means2, 

             yerr=[(top-bot)/2 for top, bot in intervals2],

             capsize=30)



ax11.set_xlabel('Diverse?', fontsize=13)

ax11.set_ylabel('Mean', fontsize=13)
stats.ttest_ind(a= demoNotDiverse['Percentage Absent'],

                b= demoDiverse['Percentage Absent'],

                equal_var=False,

               nan_policy = 'omit')    # Assume samples have equal variance?
sdAttDf = demoAbsent.join(demo['School District']).groupby('School District').mean().reset_index()

sdAttDf.head(5)
#Creating a folium map object to overlay the NYC School District GeoJSON file

sdAtt = folium.Map(location=[40.652922, -73.984353], 

                    zoom_start=9.5, tiles='Cartodb Positron')



#Creating a choropleth folium map

folium.Choropleth(geo_data = geo_json_data,  

              data = sdAttDf,

              columns = ['School District', 'Percentage Absent'],

              key_on = 'feature.properties.school_dist',

              fill_color = 'YlOrRd', 

              fill_opacity = 0.6, 

              line_opacity = 0.2,

              legend_name = 'Percentage Absent').add_to(sdAtt)

sdAtt.save(outfile='sdAtt.html')

IFrame('sdAtt.html', width=500, height=550)