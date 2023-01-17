import pandas as pd

import matplotlib.pyplot as plt

import plotly

import plotly.offline as py

py.init_notebook_mode(connected=False)

import plotly.graph_objects as go 

import numpy as np

import seaborn as sns

from sklearn.linear_model import LinearRegression
df = pd.read_csv('../input/us-accidents/US_Accidents_June20.csv',index_col='ID',parse_dates=['Start_Time','End_Time'])

df['Month'] = df['Start_Time'].dt.month

df['Year'] = df['Start_Time'].dt.year

df['Hour'] = df['Start_Time'].dt.hour

df['Weekday'] = df['Start_Time'].dt.weekday

df['Impact'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds()/60
# clean the data based on the condition that the impact on traffic is between zero-one week,and drop duplicates

oneweek = 60*24*7

df_clean = df[(df['Impact']>0) & (df['Impact']< oneweek)].drop_duplicates(subset=['Start_Time','End_Time','City','Street','Number','Description'])
#summary of the dataset

df_clean.info()
df_clean.head(2)
df_clean.describe().T
#time series analysis

df1 = df_clean[['Country','Start_Time','End_Time','Year','Month','Weekday','Hour','Impact','Severity']]
sns.set_style('whitegrid')

sns.set_context('talk')

sns.set_palette('GnBu_d')

a = sns.catplot(x='Year',data=df_clean[df_clean['Year'] < 2020],kind='count')

a.fig.suptitle('Yearly accidents cases(2016-2019)',y=1.03)

a.set(ylabel='yearly cases',xlabel='year')

plt.show()

# there is a growing trend of year accidents cases
dfA = df1[df1['Year'] < 2020].set_index('Start_Time').resample('A').count()

dfA['YEAR']=[2016,2017,2018,2019]

plt.scatter(dfA.YEAR,dfA.Country)

#use linear regression and scatter plot to test if there exists a linear regression

lrModel = LinearRegression()

x=dfA['YEAR'].values.reshape(-1, 1)

y=dfA.Country

# r^2 = 0.915, which indicate there is a strong linear relationship between year and accident cases

# did a regression fit test on quarterly increase, r^2 is 0.74, therefore yearly increase is a better fit

lrModel.fit(x,y)

print(lrModel.score(x,y))

# use linear regression parameter to predict the accident number in 2020

dfA.loc['2020-12-31 00:00:00','Country'] = lrModel.coef_*2020+ lrModel.intercept_

dfA.loc['2020-12-31 00:00:00','YEAR'] = 2020



print(dfA[['YEAR','Country']])



sns.set_context('talk')

p = sns.catplot(x='YEAR',y='Country',data=dfA,kind='bar')

p.fig.suptitle('Yearly accidents cases(2016-2020)',y=1.03)

p.set(ylabel='yearly cases',xlabel='year')

plt.show()
sns.set_context('talk')

m = sns.catplot(x='Month',data=df1[df1['Year'] < 2020],kind='count')

m.fig.suptitle('monthly accidents cases(2016-2019)',y=1.03)

m.set(ylabel='monthly cases')

plt.show()

# there were more cases druing 8-12 compared to other months,excluding the data from 2020

# guess there are more bad weather conditions in the winter
sns.set_context('talk')

w = sns.catplot(x='Weekday',data=df1,kind='count')

w.fig.suptitle('weekday accidents cases',y=1.03)

w.set(ylabel='weekday cases')

plt.show()

# accidents cases on working day is much larger then those on weekend, as less people go out to work
sns.set_context('paper')

h = sns.catplot(x='Hour',data=df1,kind='count')

h.fig.suptitle('Hourly accidents cases',y=1.03)

h.set(ylabel='hourly cases',xlabel='hour')

plt.annotate('morning peak',xy=(6,330000))

plt.annotate('afternoon peak',xy=(15,270000))

plt.annotate('bottom',xy=(1,25000))

plt.annotate('go to work',xy=(7.5,0),xytext=(1,125000),arrowprops={'arrowstyle':'fancy'})

plt.annotate('get off work',xy=(17.5,0),xytext=(19,150000),arrowprops={'arrowstyle':'fancy'})

plt.show()

# most accidents happend during the day time, and there are two peaks on 7-8 and 16-17 when people are on commute 

# between workplace and home

# during 23 to 3 o'clock，before dawn.cases numbers are relatively at the bottom level as most people are in sleep
df1.groupby('Year')['Severity'].mean().plot(kind='line')

plt.xticks([2016,2017,2018,2019,2020])

plt.ylabel('Severity')

plt.show()



# the accidents severity declined since 2017, we can assume that people have better security awareness 

# the improve of infrastructure and traffic education also contribute to the decrease

# we have strong reason to predict the traffic severity will continue to drop
print(df1.groupby('Hour')['Severity'].mean())
print(df1.groupby('Weekday')['Severity'].mean())
print(df1.groupby('Month')['Severity'].mean())
df1['Severity'].value_counts(normalize=True)
df1['Severity'].value_counts(normalize=True).plot(kind='pie')

# from the proportion of severity we can see that degree 2 and 3 together make up almost 96% of the total cases 

# proportion of degree 2 alone is nearly 70%

# shows the reason of that at any time level,the mean severity is always around 2.5 and fluctuation is not significant 

# but given the huge sample size, we should believe that evev tiny fluctuation in severity also matters
impact_h = df1.groupby('Hour')['Impact'].mean()

severity_h = df1.groupby('Hour')['Severity'].mean()

fig,ax=plt.subplots()

ax.plot(impact_h,color='blue',label='impact time')

ax.set_xlabel('hour')

ax.set_ylabel('average traffic impact(minuts)',color='blue')

ax.legend(loc='upper right')



ax2 = ax.twinx()

ax2.plot(severity_h,color='green',label='severity')

ax2.set_ylabel('average hourly severity ',color='green')

ax2.set_label('severity')

ax.set_title('hourly accidents impact and severity')

ax2.legend(loc='upper center')

plt.style.use('bmh')

plt.xlim((0,23))

plt.show()

#the basic trend of severity and impact time on traffic overlap, night-time severity and impact is severe than daytime

cases_w = df1.groupby('Weekday')['Impact'].count()

severity_w = df1.groupby('Weekday')['Severity'].mean()

fig,ax=plt.subplots()

ax.plot(cases_w,color='blue',label='cases number')

ax.set_xlabel('weekday')

ax.set_ylabel('cases in a week',color='blue')

ax.legend(loc='center left')



ax2 = ax.twinx()

ax2.plot(severity_w,color='green',label='severity')

ax2.set_ylabel('average accidents severity in a week ',color='green')

ax2.set_label('severity')

ax.set_title('weekday accidents cases and severity')

ax2.legend(loc='center right')

plt.style.use('bmh')

plt.show()

#although cases dropped a lot on weekend, the average impact of cases on weekend is much higher compared to working day

# guess the reason is that on weekend, the reaction speed of police and other department is slower
# drop the rows with missing weather condition description

df_weather=df_clean[['Month','Weather_Condition','Impact','Severity']].dropna()
df_weather.isna().sum()
weatherDict = {'Light Rain':'Rain','Rain':'Rain','Clear':'Fair','Fair':'Fair','Mostly Cloudy':'Cloudy','Overcast':'Cloudy',

        'Partly Cloudy':'Cloudy','Cloudy':'Cloudy','Scattered Clouds':'Cloudy','Light Snow':'Ice','Haze':'Fog',

       'Fog':'Fog','Heavy Rain':'Rain','Light Drizzle':'Rain','Fair / Windy':'Fair','Snow':'Ice',

        'Light Thunderstorms and Rain':'Thunder','Thunderstorm':'Thunder','Mostly Cloudy / Windy':'Cloudy','Cloudy / Windy':'Cloudy',

       'T-Storm':'Thunder','Smoke':'Fog','Thunder in the Vicinity':'Thunder','Light Rain with Thunder':'Thunder','Partly Cloudy / Windy':'Cloudy',

      'Patches of Fog':'Fog','Drizzle':'Rain','Heavy Thunderstorms and Rain':'Thunder','Mist':'Fog','Thunder':'Thunder',

       'Thunderstorms and Rain':'Thunder','Light Freezing Rain':'Ice','Light Rain / Windy':'Rain','Heavy T-Storm':'Thunder',

       'Wintry Mix':'Ice','Heavy Snow':'Ice','Shallow Fog':'Fog','Light Snow / Windy ':'Ice','Light Freezing Fog':'Ice',

       'Light Freezing Drizzle':'Ice','Rain / Windy':'Rain','N/A Precipitation':'Fair','Showers in the Vicinity':'Rain',

       'Blowing Snow':'Ice','Heavy Rain / Windy':'Rain','Heavy Drizzle':'Rain','Light Ice Pellets':'Ice','Heavy T-Storm / Windy':'Thunder',

       'T-Storm / Windy':'Thunder','Haze / Windy':'Fog','Light Rain Showers':'Rain','Widespread Dust':'Fog','Light Rain Shower':'Rain',

       'Drizzle and Fog':'Fog','Snow / Windy':'Ice','Rain Showers':'Rain','Blowing Dust / Windy':'Fog','Thunder / Windy':'Thunder',

       'Ice Pellets':'Ice','Fog / Windy':'Fog','Blowing Snow / Windy':'Ice','Heavy Snow / Windy':'Ice','Wintry Mix / Windy':'Ice',

       'Small Hail':'Ice','Sand / Dust Whirlwinds':'Fog','Squalls':'Cloudy','Light Snow Showers':'Ice','Light Thunderstorms and Snow':'Thunder',

       'Volcanic Ash':'Fog','Partial Fog':'Fog','Freezing Rain':'Ice','Rain Shower':'Rain','Light Snow / Windy':'Ice',

       'Blowing Dust':'Fog','Light Drizzle / Windy':'Rain','Light Snow and Sleet':'Ice','Light Sleet':'Ice','Snow and Sleet':'Ice',

       'Funnel Cloud':'Cloudy','Smoke / Windy':'Fog','Light Rain Shower / Windy':'Rain','Squalls / Windy':'Cloudy','Light Haze':'Fog'}
df_weather.loc[:,'Condition'] = df_weather.Weather_Condition.map(weatherDict)
df_weather.head(10)
print(df_weather.isna().sum())

df_weather_sort = df_weather.dropna()

print(df_weather_sort.isna().sum())
df_weather_sort.Condition.value_counts(normalize = True)
sns.set_context('notebook')

w = sns.catplot(x='Condition',y='Impact',data=df_weather_sort,kind='point',ci=None,

                order=['Fog','Thunder','Rain','Fair','Cloudy','Ice'])

w.set(title='The impact time in different weather condition',

      xlabel= 'weather condition',ylabel='traffic impact(minute)')



# the point plot shows that under weather condition with ice, the impact accidents have on traffic is the longest of

# more than 100 minuts

# impact under other weather conditions are nearly the same
sns.set_context('notebook')



s = sns.catplot(x='Condition',y='Severity',data=df_weather_sort,kind='point',ci=None,color='c',

                order=['Fog','Fair','Cloudy','Rain','Thunder','Ice'])

s.set(title='The accident severity in different weather condition',

      xlabel='weather condition',ylabel='severity')

# under extreme weather conditions like ice and thunder, the severity is much higher
#reporting by state

df_clean['State'].describe()
fig, ax = plt.subplots(figsize = (15,8))

sns.set_context("paper")

g = sns.countplot(x = "State", data = df_clean, ax = ax)

g.set_title("Reporting by State")

plt.show()

#california and taxes have most accident rates
#10 states with the highest accident rates

df_st = df_clean.groupby('State').size().to_frame('Counts')

df_st = df_st.reset_index().sort_values('Counts', ascending = False)[:10]



fig, ax = plt.subplots(figsize = (12,8))

b = sns.barplot(y = 'State',x = 'Counts', data = df_st )



b.set_title("10 States With The Highest Accident Rates")



plt.show()

# these states are consistent with the states with largest population in the U.S.
#average accident severity of states

st_sev = df_clean.groupby('State').mean('Severity')[['Severity']]



fig = go.Figure(data=go.Choropleth( 

    locations=list(st_sev.index),

    z = st_sev['Severity'].astype(float),  

    locationmode = 'USA-states', 

    colorscale = 'Reds', 

    colorbar_title = "Average value of severity", 

)) 



fig.update_layout( 

    title_text = 'Accident Severity of Each State', 

    geo_scope='usa', 

    

)



py.iplot(fig,filename = 'Severity_Map.html')



#SD & WY have very few accident records but the average value of severity are much higher than other states; 

#it probably because the population or the number of cars are less in these two states and there are many mountains and most of the roads are highway or mountian road.

#despite of these two states, overall, the eastern US is more serious than the western US in terms of accident severity
#report by cities

#top 10 cities with highest severity

df_city = df_clean.groupby('City').sum('Severity')[['Severity']]

df_city = df_city.reset_index().sort_values('Severity', ascending = False)[:10]



sns.set_style('whitegrid')

fig, ax = plt.subplots(figsize = (12,8))

c = sns.barplot(x = 'Severity', y = 'City', data = df_city)

c.set_title("Top 10 Cities with Highest Severity")



plt.show()



#most of these cities are large cities.



#10 cities with the highest accident rates

df_ci_cnt = df_clean.groupby('City').size().to_frame('Count_city')

df_ci_cnt = df_ci_cnt.reset_index().sort_values('Count_city', ascending = False)[:10]



fig, ax = plt.subplots(figsize = (12,8))

b = sns.barplot(y = 'City',x = 'Count_city', data = df_ci_cnt )

#给barplot加数据标签



b.set_title("10 Cities With The Highest Accident Rates")



plt.show()



#street classification

def str_type(text):

    if '-' in text or 'Fwy'in text or 'Expy' in text or 'Highway'in text or 'Hwy'in text :

        result = 'Highway'

    else:

        result = 'others'

    return result



df_clean['Street_Type'] = df_clean['Street'].apply(str_type)
df_clean[['Street_Type']].head(5)
# accident rates vs. street_type

e = sns.countplot(x = 'Street_Type', data = df_clean)

e.set_title('Accident Rate VS. Street Type')

plt.xticks(rotation = 90)

plt.show()

#given that the milage of highway is much less than other roads, this plot indicates that there is a higher probability of accident occurs in highway
#accident severity vs. street type

df_str_sev = df_clean.groupby('Street_Type').mean('Severity')[['Severity']]

df_str_sev = df_str_sev.reset_index().sort_values('Severity', ascending = False)



sns.set_style('whitegrid')

fig, ax = plt.subplots(figsize = (12,8))

c = sns.barplot(x = 'Severity', y = 'Street_Type', data = df_str_sev)

c.set_title("Accident Severity VS. Street_Type")



plt.show()

#The average accident severity of highway is higher than the other roads. 

#it might be because the speed of motor vehicles is much higher in highways.
#impact time vs. street type

f = sns.catplot(x = 'Street_Type', y = 'Impact', data = df_clean, kind = 'box', sym = '')

f.fig.suptitle("Impact Time VS. Street Type", y = 1.05)

plt.xticks(rotation = 90)



plt.show()



#the impact time of accidents in highway is much longer than that in other roads.

#related to the accident severity
#analyzing the facilities

#impact time vs. traffic signal/junction/crossing

d = sns.catplot(x = 'Traffic_Signal', y = 'Impact', data = df_clean, kind = 'box', col = 'Crossing', row = 'Junction',sym='')

d.fig.suptitle("The Relationship Between Impact Time And Traffic Signal(With/Without Junction/Crossing)", y = 1.05)

plt.show()

# impact time tends to be shorter with the presence of traffic signals 

#impact time tends to be longer with the presence of junctions

# when accidents occured in the location with traffic signals and junctions, if there is a crossing, the impact time tend to be longer
#relationship between average severity and traffic signal

tra_sev = df_clean.groupby('Traffic_Signal').mean('Severity')[['Severity']]

tra_sev.head()

#given the huge sample size and the severity distribution, we should believe that every tiny fluctuation in severity also matters.

#these following statistics show that except junction, the presence of those road features(like traffic signals, crossing, amenity) can help reduce the accident severity

#the reason might be that when there are traffic signals, crossing or other facilities, people tend to be cautious

#while when there are junctions, cars are more likely to collide due to the limited visibility in junctions
#relationship between average severity and amenity

ame_sev = df_clean.groupby('Amenity').mean('Severity')[['Severity']]

ame_sev.head()
#relationship between average severity and bump

bum_sev = df_clean.groupby('Bump').mean('Severity')[['Severity']]

bum_sev.head()
#relationship between average severity and Crossing

cro_sev = df_clean.groupby('Crossing').mean('Severity')[['Severity']]

cro_sev.head()
#relationship between average severity and Give way

giv_sev = df_clean.groupby('Give_Way').mean('Severity')[['Severity']]

giv_sev.head()
#relationship between average severity and Junction

jun_sev = df_clean.groupby('Junction').mean('Severity')[['Severity']]

jun_sev.head()
#relationship between average severity and no-exit

noe_sev = df_clean.groupby('No_Exit').mean('Severity')[['Severity']]

noe_sev.head()
#relationship between average severity and roundabout

rou_sev = df_clean.groupby('Roundabout').mean('Severity')[['Severity']]

rou_sev.head()
#relationship between average severity and station

sta_sev = df_clean.groupby('Station').mean('Severity')[['Severity']]

sta_sev.head()
#relationship between average severity and stop

sto_sev = df_clean.groupby('Stop').mean('Severity')[['Severity']]

sto_sev.head()
#relationship between average severity and traffic calming

trac_sev = df_clean.groupby('Traffic_Calming').mean('Severity')[['Severity']]

trac_sev.head()