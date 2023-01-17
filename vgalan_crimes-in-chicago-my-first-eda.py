# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plots
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
csv_files = ['../input/Chicago_Crimes_2001_to_2004.csv',
            '../input/Chicago_Crimes_2005_to_2007.csv',
            '../input/Chicago_Crimes_2008_to_2011.csv',
            '../input/Chicago_Crimes_2012_to_2017.csv',
            ]

frames = []
for csv in csv_files:
    df = pd.read_csv(csv ,usecols = ['Date','Primary Type','Location Description','District','Community Area','Arrest'])
    frames.append(df)
    
crime = pd.concat(frames)   
crime.head()
crime.shape
# Exploring the missing values:
print("Are There Missing Data? :",crime.isnull().any().any())     
print(crime.isnull().sum())
dfcrime = crime.dropna()
dfcrime.shape
count_data_origin = crime['Date'].count()
count_data_modify = dfcrime['Date'].count()
Value = (count_data_modify/count_data_origin)*100
print('The analysis will be carried on with:',"%.2f" % Value,'% of the total data')
dfcrime.info(null_counts = True)
# It's more comfortable to do the next transformation for the doing analysis.
dfcrime.columns=[each.replace(" ","_") for each in dfcrime.columns]
dfcrime.columns
dfcrime.District.unique()
dfcrime[(dfcrime['District'] == 'Beat')]
dfcrime = dfcrime[dfcrime.District != 'Beat']
# Changing the type 
dfcrime['District'] = dfcrime['District'].astype(float).astype(int)
dfcrime.District.unique()
dfcrime.Community_Area.unique()
# Changing the type
dfcrime['Community_Area'] = dfcrime['Community_Area'].astype(float).astype(int)
dfcrime['Community_Area'].unique()
dfcrime.Primary_Type.unique()
dfcrime['Primary_Type'] = dfcrime['Primary_Type'].replace(['NON - CRIMINAL',
                                                           'NON-CRIMINAL (SUBJECT SPECIFIED)'], 
                                                          'NON-CRIMINAL')
dfcrime.Primary_Type.unique()
dfcrime.Location_Description.unique()
dfcrime['Location_Description'].value_counts().head()
dfcrime.Arrest.unique()
dfcrime.Arrest.value_counts()
dfcrime.Arrest = dfcrime.Arrest.astype('bool')
dfcrime.Arrest.unique()
# Splitting the Date columns will make the analysis easier. 
dfcrime['date'] = dfcrime['Date'].str[:11]
dfcrime['time'] = dfcrime['Date'].str[12:]
dfcrime['date'] = pd.to_datetime(dfcrime['date'])
dfcrime['time'] = pd.to_datetime(dfcrime['time'])
# Establish 'date' as index in the dataframe.
dfcrime.index = dfcrime.date
del dfcrime['Date']
del dfcrime['date']
dfcrime.head(2)
crime_year = dfcrime.resample('Y').count()
crime_year = pd.DataFrame(crime_year.iloc[:,0])
crime_year.columns = ['Total_crime_per_year']
print(crime_year.head())
print(crime_year.tail())
dfcrime = dfcrime['2002' : '2016']
crime_year = crime_year['2002' : '2016']

a = crime_year.index
b = np.arange(2002,2018)

grid = sns.barplot(x = a ,y = 'Total_crime_per_year', data = crime_year, color = 'black')

grid.set_xticklabels(b, rotation = 60)
plt.ylabel('Total Crime')
plt.xlabel('Year')
plt.title('Crime per Year')
plt.axhline(crime_year['Total_crime_per_year'].mean())
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.show()
print('- Crime Activity 2005-2006')
Value_a = ((crime_year['2006'].values- crime_year['2005'].values)/crime_year['2006'].values)*100
print('%.2f' % Value_a, '% has been the crime activity increment from 2005 to 2006', '\n')

print('- Crime Activity 2010-2011')
Value_b = ((crime_year['2010'].values- crime_year['2011'].values)/crime_year['2010'].values)*100
print("%.2f" % Value_b, '% has been the crime activity increment from 2010 to 2011', '\n')
crime_month = dfcrime.resample('M').count()
crime_month = pd.DataFrame(crime_month.iloc[:,0])
crime_month.columns = ['Total_crime_per_month']

crime_month.plot()
plt.xlabel('Time')
plt.ylabel('Total crime')
plt.title('Crime per mes')
plt.axhline(crime_month['Total_crime_per_month'].mean(), color = 'black')
fig=plt.gcf()
fig.set_size_inches(10,5)
print('Top 5 months with more crime activity from 2002 to 2005')
print(crime_month['2002':'2005'].sort_values('Total_crime_per_month', ascending = False).head(), '\n')

print('Top 5 months with more crime activity from 2006 to 2010')
print(crime_month['2006':'2010'].sort_values('Total_crime_per_month', ascending = False).head(),  '\n')

print('Top 5 months with more crime activity from 2011 to 2013')
print(crime_month['2011':'2013'].sort_values('Total_crime_per_month', ascending = False).head(), '\n')

print('Top 5 months with more crime activity from 2014 to 2016')
print(crime_month['2014':'2016'].sort_values('Total_crime_per_month', ascending = False).head(), '\n')
print('Top 10 months with less criminal activity from 2003 to 2005')
print (crime_month['2002':'2005'].sort_values('Total_crime_per_month').head(5), '\n')

print('Top 10  months with less criminal activity from 2006 to 2010')
print(crime_month['2006':'2010'].sort_values('Total_crime_per_month').head(5), '\n')

print('Top 5 months with less criminal activity from 2011 to 2013')
print(crime_month['2011':'2013'].sort_values('Total_crime_per_month').head(), '\n')

print('Top 5 months with less criminal activity from 2015 to 2016')
print(crime_month['2015':'2016'].sort_values('Total_crime_per_month').head(), '\n')
fig, ax = plt.subplots()

ax.plot(crime_month['2002'].values, color = 'Blue', label = '2002')
ax.plot(crime_month['2003'].values, color = 'Green', label = '2003')
ax.plot(crime_month['2004'].values, color = 'Brown', label = '2004')
ax.plot(crime_month['2005'].values, color = 'Magenta', label = '2005')
ax.plot(crime_month['2006'].values, color = 'Yellow', label = '2006')
ax.plot(crime_month['2007'].values, color = 'red', label = '2007')
ax.plot(crime_month['2008'].values, color = 'cyan', label = '2008')
ax.plot(crime_month['2009'].values, color = 'orange', label = '2009')
ax.plot(crime_month['2010'].values, color = 'hotpink', label = '2010')
ax.plot(crime_month['2011'].values, color = 'lime', label = '2011')
ax.plot(crime_month['2012'].values, color = 'm', label = '2012')
ax.plot(crime_month['2013'].values, color = 'silver', label = '2013')
ax.plot(crime_month['2014'].values, color = 'olive', label = '2014')
ax.plot(crime_month['2015'].values, color = 'salmon', label = '2015')
ax.plot(crime_month['2016'].values, color = 'dimgray', label = '2016')

plt.xlabel('Month')
plt.ylabel('Total crimes')
plt.title('Criminal Activity during a year')
plt.axhline(crime_month['Total_crime_per_month'].mean(), color = 'black')
plt.xlabel('Month')

c = ['January','January','March','May','July','September','November']
ax.set_xticklabels(c)

plt.legend(bbox_to_anchor=(1, 0, .3, 1), loc=2,
           ncol=2, mode="expand", borderaxespad=0)

fig=plt.gcf()
fig.set_size_inches(10,5)


Sum_m = [(crime_month['2002'].values + crime_month['2003'].values + crime_month['2004'].values + 
     crime_month['2005'].values + crime_month['2006'] + crime_month['2007'].values + crime_month['2008'].values 
                + crime_month['2009'].values + crime_month['2010'].values + crime_month['2011'].values + 
                crime_month['2012'].values +crime_month['2013'].values + crime_month['2014'].values + 
                crime_month['2015'].values + crime_month['2016'].values)]

Sum_m = Sum_m[0]
Sum_m = Sum_m.reset_index()
Sum_m.index = [['January','Febreary','March','April','May','June',
                'July','August','September','October','November','December']]
del Sum_m['date']
print('Top 5 months with more criminal activity in total')
print(Sum_m.sort_values('Total_crime_per_month', ascending= False).head(), '\n')

print('Top 5 safer months')
print(Sum_m.sort_values('Total_crime_per_month').head(), '\n')
Sum_m.plot(kind = 'bar')
fig=plt.gcf()
fig.set_size_inches(10,5)

plt.xlabel('Month')
plt.ylabel('Total crime')
plt.title('Total crime per month')
crime_day = dfcrime.resample('d').count()
crime_day = pd.DataFrame(crime_day.iloc[:,0])
crime_day.columns = ['Total_crime_per_day']
crime_day = crime_day['2002' : '2016']

median_day = crime_day.Total_crime_per_day.mean()
print('The average of criminal activity in the city fo Chicago is:',"%.2f" % median_day)
crime_day.plot()

plt.ylabel('Total crime')
plt.title('Crime per day')
plt.axhline(crime_day.Total_crime_per_day.mean(), color = 'black')

fig=plt.gcf()
fig.set_size_inches(20,10)
max_day = crime_day['Total_crime_per_day'].max()
print('The day with more crimes in the city of Chicago was:')
crime_day[(crime_day['Total_crime_per_day'] == max_day)]
min_day = crime_day['Total_crime_per_day'].min()
print('The day with less crimes in the city of Chicago was:')
crime_day[(crime_day['Total_crime_per_day'] == min_day)]
dfhours = dfcrime.reset_index()
dfhours.index = dfhours.time
dfhours.head(1)
dfhours = dfhours.resample('h').count()
dfhours = dfhours[['time']]
dfhours.columns = ['Sum_Crimes_per_Hour']
dfhours['Crime_per_hour_median'] = dfhours['Sum_Crimes_per_Hour']/(365*15)
dfhours
median_hour = dfhours.Crime_per_hour_median.median()
print('The mean crime per hour is:',"%.2f" % median_hour,'\n')
x = dfhours.index
y = dfhours.Crime_per_hour_median
a = np.arange(0,24)

grid = sns.pointplot(x = x, y = y, data = dfhours)
grid.set_xticklabels(a)
plt.axhline(dfhours.Crime_per_hour_median.mean(), color = 'black')
fig = plt.gcf()
fig.set_size_inches(20,10)
plt.grid()
print('Top 5 hours with more crime activity')
print(dfhours.sort_values('Crime_per_hour_median', ascending= False).head(), '\n')

print('Top 5 hours with less crime activity')
print(dfhours.sort_values('Crime_per_hour_median').head(7))
dfcrime['Community_Area'].value_counts()
plt.figure(figsize=(20,25))
sns.countplot(y = dfcrime['Community_Area'])
plt.axvline(dfcrime['Community_Area'].value_counts().mean(), color = 'black', alpha = 0.5) 
plt.xlabel('Total crime')
plt.ylabel('Community Area')
plt.title('Crime per Community Area')
c_a = pd.DataFrame(dfcrime['Community_Area'].value_counts())
c_a.columns = ['Number_of_crimes']
c_a.index.name = 'Community_Area'

print('The most dangerous areas in the city of Chicago are:')
print(c_a.sort_values('Number_of_crimes', ascending = False).head(), '\n')

print('The safest areas in the city of chicago are:')
print(c_a.sort_values('Number_of_crimes').head(8))
c_a['Percent_%'] = round((c_a['Number_of_crimes']/c_a['Number_of_crimes'].sum())*100,2)

print(c_a.sort_values('Percent_%', ascending = False).head(), '\n')
c_a_last_five = pd.DataFrame(dfcrime['2012':'2016'])
c_a_last_five = pd.DataFrame(c_a_last_five['Community_Area'].value_counts())
c_a_last_five.columns = ['Number_of_crimes']
c_a_last_five.index.name = 'Community_Area'

print('The most dangerous areas in the city of Chicago in the last 5 years (2012 to 2016):')
c_a_last_five['Percent_%'] = round((c_a_last_five['Number_of_crimes']/c_a_last_five['Number_of_crimes'].sum())*100,2)
print(c_a_last_five.sort_values('Percent_%', ascending = False).head(), '\n')

print('The safest areas in the city of Chicago in the last 5 years (2012 to 2016):')
print(c_a_last_five.sort_values('Number_of_crimes').head(8))


c_a_last_year = dfcrime['2016']
c_a_last_year = pd.DataFrame(c_a_last_year['Community_Area'].value_counts())
c_a_last_year.columns = ['Number_of_crimes']
c_a_last_year.index.name = 'Community_Area'

print('The more dangerous areas in the city of Chicago in 2016:')
c_a_last_year['Percent_%'] = round((c_a_last_year['Number_of_crimes']/c_a_last_year['Number_of_crimes'].sum())*100,2)
print(c_a_last_year.sort_values('Percent_%', ascending = False).head(), '\n')

print('The safest areas in the city of Chicago in 2016:')
print(c_a_last_year.sort_values('Number_of_crimes').head(8))
c_a.index[:15]
top_c_a = dfcrime.groupby('Community_Area').resample('Y').count()
top_c_a = top_c_a[['Community_Area']]
top_c_a.columns = ['Sum_C_A']
top_c_a = top_c_a.reset_index()
top_c_a= top_c_a.set_index(['date','Community_Area'])
top_c_a.head(1)
top_c_a = top_c_a[top_c_a.index.get_level_values('Community_Area').isin([25, 43, 8, 23, 67, 
                                                                         24, 71, 28, 29, 68, 49, 
                                                                         66, 69, 32, 22])]
top_c_a.head(1)
top_c_a.unstack(level=1).plot(kind='line')
fig=plt.gcf()
fig.set_size_inches(20,10)

plt.xlabel('Year')
plt.ylabel('Total crime')
plt.title('top most dangerous community areas during the years')

plt.legend( loc = 'best')
plt.show()
dfcrime.District.value_counts()
plt.figure(figsize=(10,5))
sns.countplot(x = dfcrime['District'])
plt.axhline(dfcrime['District'].value_counts().mean(), color = 'black', alpha = 0.5)
plt.xlabel('District')
plt.ylabel('Total crime')
plt.title('Crime per police district ')
crime['District'].value_counts()[:22].plot(kind='pie',autopct='%1.1f%%')
# Here the decision has been made to use 22 districts since the others did not have almost anything relevant
plt.title('Distribution per district')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()
district = dfcrime.groupby('District').resample('Y').count()
district = district[['District']]
district.columns = ['Sum_D']
district.unstack(level=0).plot(kind='line')
fig=plt.gcf()
fig.set_size_inches(15,8)
plt.legend( bbox_to_anchor=(0.7, 0, .3, 1), loc=2,
           ncol=2, mode="expand", borderaxespad=0)
plt.xlabel('Year')
plt.ylabel('Total crime')
plt.title('District police evolution over the years')
plt.show()
crime['Arrest'].value_counts()[:2].plot(kind='pie',autopct='%1.1f%%')
plt.title('Arrests')
fig=plt.gcf()
fig.set_size_inches(5,5)
plt.show()
arrest = dfcrime.groupby('Arrest').resample('Y').count()
arrest = arrest[['Arrest']]
arrest.columns = ['Sum_Arrest']
ax = arrest.unstack(level=0).plot(kind = 'bar')

plt.legend( bbox_to_anchor=(0.7, 0, .3, 1), loc=2,
           ncol=1, mode="expand", borderaxespad=0)

ax.set_xticklabels(b, rotation = 60)

plt.ylabel('Total')
plt.xlabel('Year')
plt.title('Arrests')
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.show()
dfcrime.Primary_Type.value_counts().head(11)
plt.figure(figsize=(10,8))
sns.countplot(y = dfcrime['Primary_Type'])
plt.axvline(dfcrime['Primary_Type'].value_counts().mean(), color = 'black', alpha = 0.5) 

plt.ylabel('Crime')
plt.xlabel('Total')
plt.title('Total Crime')
top_type = dfcrime.groupby('Primary_Type').resample('Y').count()
top_type  = top_type [['Primary_Type']]
top_type .columns = ['Sum_type']
top_type  = top_type.reset_index()
top_type = top_type.set_index(['date','Primary_Type'])
top_type .head(1)
top_type = top_type[top_type.index.get_level_values('Primary_Type').isin(['THEFT','BATTERY','CRIMINAL DAMAGE',
                                                        'NARCOTICS','OTHER OFFENSE','ASSAULT',
                                                        'BURGLARY','MOTOR VEHICLE THEFT','ROBBERY',
                                                        'DECEPTIVE PRACTICE','CRIMINAL TRESPASS'])]
top_type.head(1)
top_type.unstack(level=1).plot(kind='line')
fig=plt.gcf()
fig.set_size_inches(15,5)
plt.legend( loc = 'best')
plt.show()
e = dfcrime[['District', 'Community_Area' ]]
e.index = e.Community_Area
del e['Community_Area']

h = e[e.index.get_level_values('Community_Area').isin([25, 43, 8, 23, 67, 
                                                        24, 71, 28, 29, 68, 49, 
                                                        66, 69, 32, 22])]
h = h.reset_index() 
h = pd.DataFrame (h.groupby(['District','Community_Area']).size())
h.columns = ['T_Distric']
h.head()
fig = plt.figure()

sns.heatmap(h.unstack(level=0), linewidths=.5, cmap="BuPu",vmin=-100000, vmax=333040)

plt.ylabel('Community Area')
plt.xlabel('District')
plt.title('Community Area vs Police Districts')

fig = plt.gcf()
fig.set_size_inches(15,5)
print('Top 10 police districts')
print(dfcrime.District.value_counts().head(10).index)
print('Top 10 community Areas')
print(dfcrime.Community_Area.value_counts().head(10).index)
Type_vs_C_A = dfcrime[['Primary_Type', 'Community_Area']]

#Filter by community Areas
Type_vs_C_A.index = Type_vs_C_A.Community_Area
del Type_vs_C_A['Community_Area']
Type_vs_C_A = Type_vs_C_A[Type_vs_C_A.index.get_level_values('Community_Area').isin([25, 43, 8, 23, 67, 
                                                        24, 71, 28, 29, 68, 49, 
                                                        66, 69, 32, 22])]
Type_vs_C_A = Type_vs_C_A.reset_index() 

#Filter by Primary_Type
Type_vs_C_A.index = Type_vs_C_A.Primary_Type
del Type_vs_C_A['Primary_Type']
Type_vs_C_A = Type_vs_C_A[Type_vs_C_A.index.get_level_values('Primary_Type').isin(['THEFT','BATTERY','CRIMINAL DAMAGE',
                                                        'NARCOTICS','OTHER OFFENSE','ASSAULT',
                                                        'BURGLARY','MOTOR VEHICLE THEFT','ROBBERY',
                                                        'DECEPTIVE PRACTICE','CRIMINAL TRESPASS'])]
Type_vs_C_A = Type_vs_C_A.reset_index() 


Type_vs_C_A = pd.DataFrame (Type_vs_C_A.groupby(['Primary_Type','Community_Area']).size())
Type_vs_C_A.columns = [' ']

Type_vs_C_A.head()
grip = sns.heatmap(Type_vs_C_A.unstack(level=0), linewidths=.5, cmap="YlGnBu", annot=True)

plt.ylabel('Community_Area')
plt.xlabel('Primary Type')
plt.title('Community Area vs Primary Type')

fig = plt.gcf()
fig.set_size_inches(15,5)