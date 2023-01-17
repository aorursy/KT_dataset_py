import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read in the dataset and take a quick pick at it

df = pd.read_csv("/kaggle/input/us-accidents-may19/US_Accidents_May19.csv")

df.sample(10)
print('The DataFrame has {} rows and {} columns'.format(df.shape[0],df.shape[1]))

print('\n')

missing = df.isnull().sum().sort_values(ascending=False)

percent_missing = ((missing/df.isnull().count())*100).sort_values(ascending=False)

missing_df = pd.concat([missing,percent_missing], axis=1, keys=['Total', 'Percent'],sort=False)

missing_df[missing_df['Total']>=1]
lst = ['Humidity(%)','Precipitation(in)','Wind_Chill(F)','Wind_Speed(mph)','Visibility(mi)']

for l in lst:

    df[l] = df[l].fillna(0)
lst = ['Temperature(F)','Pressure(in)']

for l in lst:

    df[l]=df[l].fillna(df[l].mean())
'''

This is a good time to take a look at our missing values again. I have added a third column showing the respective data types

'''

missing = df.isnull().sum().sort_values(ascending=False)

percent_missing = ((missing/df.isnull().count())*100).sort_values(ascending=False)

missing_df = pd.concat([missing,percent_missing,df[missing.index].dtypes], axis=1, keys=['Total', 'Percent','Data Types'],sort=False)

missing_df[missing_df['Total']>=1]
missing_copy = missing_df[missing_df['Total']>=1].copy()
object_columns = missing_copy[missing_copy['Data Types']=='object'].index

df[object_columns].head()
df['City'] = df.groupby('State')['City'].transform(lambda grp: grp.fillna(grp.value_counts().index[0]))
df['Start_Time'] = pd.to_datetime(df['Start_Time']) # convert Start_Time to datetime

df['End_Time'] = pd.to_datetime(df['End_Time']) # convert End_Time to datetime

df['Weather_Timestamp'] = pd.to_datetime(df['Weather_Timestamp']) # convert Weather_Timestamp to datetime
# fill the Nautical_Twilight column with Day/Night by inferring the Start_Time column



def filler(df,columns):

    # get list comprising column missing data

    lst = df[df[columns].isna()].index

    for i in lst:

        if 6<= df.loc[i,'Start_Time'].hour and df.loc[i,'Start_Time'].hour <18:

            df[columns] = df[columns].fillna('Day')

        else:

            df[columns] = df[columns].fillna('Night')



filler(df,'Nautical_Twilight')
# Another easier option is to just impute the Day/Night values wth the mode as ['Sunrise_Sunset','Civil_Twilight','Astronomical_Twilight'] 

# vary depending on time of year and might be difficult to infer based on hour of day.



def median_imputer(x):

    df[x].fillna(df[x].mode()[0],inplace=True)



median_impute = ['Sunrise_Sunset','Civil_Twilight','Astronomical_Twilight','Wind_Direction','Weather_Condition']

for col in median_impute:

    median_imputer(col)
# impute the timezone based on the State column



df['Timezone'] = df.groupby('State')['Timezone'].transform(lambda tz: tz.fillna(tz.value_counts().index[0]))
# impute the Weather_Timestamp with the value at Start_Time. This column records the time the weather was taken (we won't really need it)



df.loc[(pd.isnull(df.Weather_Timestamp)), 'Weather_Timestamp'] = df.Start_Time
'''

This is a good time to take a look at our missing values again.

'''

missing = df.isnull().sum().sort_values(ascending=False)

percent_missing = ((missing/df.isnull().count())*100).sort_values(ascending=False)

missing_df = pd.concat([missing,percent_missing,df[missing.index].dtypes], axis=1, keys=['Total', 'Percent','Data Types'],sort=False)

missing_df[missing_df['Total']>=1]
# we do for Zipcode and Airport_Code what we did for columns like Timezone

df['Zipcode'] = df.groupby('State')['Zipcode'].transform(lambda zc: zc.fillna(zc.value_counts().index[0]))

df['Airport_Code'] = df.groupby('State')['Airport_Code'].transform(lambda ac: ac.fillna(ac.value_counts().index[0]))
# we will fill the one record in Description with 'Accident'



df.Description = df.Description.fillna('Accident')
df.drop(labels=['End_Lat', 'End_Lng'],axis=1,inplace=True)
df['Number'] = df.groupby('State')['Number'].transform(lambda n: n.fillna(n.value_counts().index[0]))

df.TMC = df.TMC.fillna(201.0)
'''

This is a good time to take a look at our missing values again.

'''

missing = df.isnull().sum().sort_values(ascending=False)

percent_missing = ((missing/df.isnull().count())*100).sort_values(ascending=False)

missing_df = pd.concat([missing,percent_missing,df[missing.index].dtypes], axis=1, keys=['Total', 'Percent','Data Types'],sort=False)

missing_df[missing_df['Total']>=1]
df.sample(10)
# write and store the cleaned file to a pickle file

df.to_pickle('US_Accidents_Cleaned.pkl')
# import libraries for Visualization

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')
df = pd.read_pickle('US_Accidents_Cleaned.pkl')
# create new features for timeseries analysis.

df['Hour'] = df['Start_Time'].dt.hour

df['Day'] = df['Start_Time'].dt.day

df['Day_Name'] = df['Start_Time'].dt.day_name()

df['Week'] = df['Start_Time'].dt.week

df['Month'] = df['Start_Time'].dt.month

df['Count'] = 1
df.groupby('Month')['Count'].value_counts()
import calendar

df.groupby('Month')['Count'].value_counts().plot(kind='bar')

df.groupby('Month')['Count'].value_counts().plot(color='k',linestyle='-',marker='.',linewidth=0.4)

plt.xticks(np.arange(12),calendar.month_name[1:13],rotation=45)

plt.xlabel('Month')

plt.title('Monthly Accident Count')
plt.figure(figsize=(10,6))

df.groupby('Week')['Count'].value_counts().plot(linewidth=1,marker='.')

plt.xticks(np.arange(52),np.arange(1,53),rotation = 90)

plt.xlabel('Week of Year')

plt.title('Accident Count by Week of Year')

plt.show()
plt.figure(figsize=(10,6))

df.groupby('State')['Count'].value_counts().plot(kind='bar')

plt.xticks(np.arange(50),sorted(df['State'].unique()),rotation = 90)

plt.xlabel('State')

plt.title('Accident Count by State')

plt.show()
by_severity = df.groupby('Severity')['Count'].sum()
sns.countplot(x='Severity',data=df)
# Bivariate visualization of categorical variables



#create a frequency table of state against severity

cat_var = pd.crosstab(columns=df['Severity'],

    index=df['State'])



#plot a stacked plot

cat_var.plot(kind='bar',stacked=True,figsize=(16,8),color=['purple','orange','blue','red','green'])

plt.title('Stacked plot of Accident Severity in respective State')

plt.ylabel('Frequency')

plt.show()
plt.figure(figsize=(12,6))

sns.boxplot(x='Severity',y='Wind_Speed(mph)',data=df,hue='Severity')

plt.ylim(0,100)
# I used median here because there are so many outliers in the boxplot that i felt using mean would skew the data



df.groupby('Severity')['Wind_Speed(mph)'].median().plot(kind='bar')

plt.ylabel('Wind_Speed(mph)')

plt.title("Median 'Wind_Speed(mph)' by Severity")

plt.show()
plt.figure(figsize=(12,6))

sns.boxplot(x='Severity',y='Wind_Chill(F)',data=df,hue='Severity')

plt.legend(loc='best')

plt.show()
df.groupby('Severity')['Wind_Chill(F)'].mean().plot(kind='bar')

plt.ylabel('Wind_Chill(F)')

plt.title("Average 'Wind_Chill(F)' by Severity")

plt.show()
def catplotter(col):

    x = df.groupby([col, 'Severity'])['Count'].sum().reset_index()

    sns.catplot("Severity", "Count", col=col, data=x, kind="bar")

    plt.show()
catplotter('Roundabout')
catplotter('Bump')
catplotter('Amenity')
catplotter('Crossing')
catplotter('Give_Way')
catplotter('Junction')
catplotter('No_Exit')
catplotter('Railway')
catplotter('Station')
catplotter('Stop')
catplotter('Traffic_Signal')
catplotter('Turning_Loop')
catplotter('Side')
catplotter('Sunrise_Sunset')
catplotter('Civil_Twilight')
catplotter('Nautical_Twilight')
catplotter('Astronomical_Twilight')
df.sample(10)
# Severity Impact by Temperature

plt.figure(figsize = (16, 6))

sns.violinplot(y="Temperature(F)", x="Severity", data=df,width=0.6,linewidth=0.5)

plt.show()
# Severity Impact by Humidity 

plt.figure(figsize = (16, 6))

sns.violinplot(y="Humidity(%)", x="Severity", data=df,width=0.6,linewidth=0.5)

plt.show()
# Severity Impact by Precipitation(in) 

plt.figure(figsize = (16, 6))

sns.violinplot(y='Precipitation(in)', x="Severity", data=df,width=0.6,linewidth=0.5)

plt.show()
# Severity Impact by Pressure(in)

plt.figure(figsize = (16, 6))

sns.violinplot(y='Pressure(in)', x="Severity", data=df,width=0.6,linewidth=0.5)

plt.show()
# Top 10 weather condition

plt.figure(figsize = (15, 6))

df[df['Weather_Condition'] != 0]['Weather_Condition'].value_counts().iloc[:10].plot(

    kind='bar',color=['b','k','g','r','c','violet','lime','y','m','purple'])

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='Hour',data=df)

df.groupby('Hour')['Count'].value_counts().plot(color='k',linestyle='-',marker='.',linewidth=0.6)

plt.title('Count of Accidents by Hour')

plt.xticks(np.arange(0,24),np.arange(0,24),rotation=90)

plt.xlabel('Hour')

plt.plot()
x = pd.crosstab(index=df['Hour'],columns=df['Severity'])

x.plot(kind='bar',stacked=True, color=['b','k','g','r','c'],figsize=(12,6))

plt.show()
from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
severity_2 = df[df['Severity']==2]['Description']

severity_3 = df[df['Severity']==3]['Description']

severity_4 = df[df['Severity']==4]['Description']
desc_2 = severity_2.str.split("(").str[0].value_counts().keys()

wc_desc_2 = WordCloud(scale=5,max_words=100,colormap="rainbow",background_color="white").generate(" ".join(desc_2))



desc_3 = severity_3.str.split("!").str[0].value_counts().keys()

wc_desc_3 = WordCloud(scale=5,max_words=100,colormap="rainbow",background_color="white").generate(" ".join(desc_3))



desc_4 = severity_4.str.split("!").str[0].value_counts().keys()

wc_desc_4 = WordCloud(scale=5,max_words=100,colormap="rainbow",background_color="white").generate(" ".join(desc_4))
fig, axs = plt.subplots(1,3,sharey=True,figsize=(17,14))



axs[0].imshow(wc_desc_2,interpolation="bilinear")

axs[1].imshow(wc_desc_3,interpolation="bilinear")

axs[2].imshow(wc_desc_4,interpolation="bilinear")



axs[0].axis("off")

axs[1].axis("off")

axs[2].axis("off")



axs[0].set_title('Severity 2 Accidents')

axs[1].set_title('Severity 3 Accidents')

axs[2].set_title('Severity 4 Accidents')



plt.show()
import folium
df.sample(3)
w = df.groupby(['State'])['Count'].sum().reset_index()
state_geo = '/kaggle/input/usa-states/usa-states.json'
n = folium.Map(location=[39.381266, -97.922211],zoom_start=5)

folium.Choropleth(

 geo_data=state_geo,

 data=w,

 columns=['State', 'Count'],

 key_on='feature.id',

 fill_color='YlOrRd',

 fill_opacity=0.7,

 line_opacity=0.2,

 legend_name='Accidents'

).add_to(n)

n