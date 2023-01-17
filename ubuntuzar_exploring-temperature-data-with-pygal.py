# Import data processing and data visualization libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
# Read the csv file in pandas as a dataframe
df_city_temp = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv', dtype={"Region": str, "Country": str, "State": str, "City": str, 
                                                          "Month": int, "Day": int, "Year": int,
                                                         "AvgTemperature": float})
# Access the first 5 rows of the dataframe
df_city_temp.head()
# Access the last 5 rows of the dataframe
df_city_temp.tail()
# Find the shape of the dataframe
df_city_temp.shape
# Find the information of the dataframe
df_city_temp.info()
# Apply a quick statistical analysis on the dataframe
df_city_temp.describe()
# remove the above information
df_city_temp = df_city_temp[df_city_temp['Year'] != 200]
df_city_temp = df_city_temp[df_city_temp['Year'] != 2020]
# Redo the descriptive statistics and find the shape
df_city_temp.describe()
df_city_temp.shape
df_city_temp = df_city_temp[df_city_temp['Year'] != 201]
# Redo the descriptive statistics
df_city_temp.describe()
# Finding the missing temperatures in each country
missing_temp = pd.DataFrame(df_city_temp.loc[df_city_temp['AvgTemperature'] == -99, 'Country'].value_counts())
missing_temp['Total'] = df_city_temp.groupby('Country')['AvgTemperature'].count()
missing_temp['Percent_Missing'] = missing_temp.apply(lambda x: (x['Country']/x['Total'])*100, axis=1)
missing_temp.sort_values(by=['Percent_Missing'], inplace=True, ascending=False)
missing_temp.head(5)
# Replace all -99 F by NaN
df_city_temp.loc[df_city_temp['AvgTemperature'] == -99, 'AvgTemperature'] = np.nan

# Replace all -99 F by NaN
# Replace NaN with the mean of the city then check to see that it has been successfull.
df_city_temp['AvgTemperature'] = df_city_temp['AvgTemperature'].fillna(df_city_temp.groupby(['City'])['AvgTemperature']
                                                                       .transform('mean'))
df_city_temp['AvgTemperature'].isnull().sum()
# Define a function that allows us to create a table of missing values in df_city_temp and their percentages in 
# descending order
def missing_values(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    percentage_final = (round(percentage, 2) * 100)
    total_percent = pd.concat(objs=[total, percentage_final], axis = 1, keys=['Total', '%'])
    return total_percent
# Find all the missing values in the dataframe
missing_values(df_city_temp)
# I define a new table df-city_temp_US, which is basically the same dataframe. This is only for those interested in 
# working on US data.
df_city_temp_US = df_city_temp
# Changing the Year, Month and Day columns into a date-stamp.
date_data = pd.to_datetime(df_city_temp[['Year','Month', 'Day']], format='%Y%m%d', errors='coerce')
df_city_temp['Date'] = date_data
# Access the first 5 rows
df_city_temp.head()
# Fahrenheit to degrees Celcius then drop the original AvgTemperature in Fahrenheit's
df_city_temp['AvgTemperature_Celcius'] = round((df_city_temp['AvgTemperature'] - 32) * (5/9), 3)
df_city_temp = df_city_temp.drop(['AvgTemperature'], axis = 1)
# Access the df_city_temp
df_city_temp.sample(10)
# Create a table to find the mean value for each region to 3 s.f.
round(df_city_temp[['Region','AvgTemperature_Celcius']].groupby(['Region'], as_index=False).agg(np.mean), 3)
# Figure configuration
fig = plt.figure(figsize=(18,10))

# Using seaborn to visualize the average temperature for each region
sns.barplot(x="Region", y="AvgTemperature_Celcius", data=df_city_temp, ci=None)
plt.title("Average Temperature in Each Region", size=20)
plt.ylim(0, 30)
plt.xlabel('Region', size=20)
plt.ylabel('Average Temperature (C)', size=20)
plt.xticks(rotation=10, size=15)
plt.yticks(size=15)
# Figure configuration
plt.figure(figsize=(15,8))

# Visualize the global average temperature
sns.lineplot(x = 'Year', y = 'AvgTemperature_Celcius', data = df_city_temp, palette='hsv')
plt.title('Global Average Temperatures', size=20)
plt.ylabel('Average Temperature (°C)', size=15)
plt.xlabel('')
plt.xticks(size=15)
plt.yticks(size=15)
plt.ylim(15, 17)
# Figure configuration
plt.figure(figsize=(15,8))

# Visualize average temperature for each region
sns.lineplot(x = 'Year', y = 'AvgTemperature_Celcius', data = df_city_temp, hue="Region")
plt.title('Average Temperatures for all Regions', size=20)
plt.ylabel('Average Temperature (°C)', size=15)
plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=1)
plt.xlabel('')
plt.xticks(size=15)
plt.yticks(size=15)
plt.ylim(8, 27)
# Extract the monthly average temperatures to 3 s. f.
df_month_avg_temp = round(df_city_temp.groupby(['Month', 'Year'])['AvgTemperature_Celcius'].agg(np.mean)
                          .reset_index().sort_values(by=['Year']), 3)
# Access the data above
df_month_avg_temp.head()
# Pivot the data
df_month_avg_temp_pivoted = pd.pivot_table(data= df_month_avg_temp, index='Month', values='AvgTemperature_Celcius', 
                                           columns='Year')
# Access the data above
df_month_avg_temp_pivoted.head()
# Figure configuration
plt.figure(figsize=(15,8))

# Visualize the global glaverage temperature for each month each year
sns.heatmap(data=df_month_avg_temp_pivoted, cmap='coolwarm', annot = True, fmt=".1f", annot_kws={'size':11})
plt.xlabel('')
plt.ylabel('Month', size=20)
plt.title('Global Average Temperatures (°C)', size=20)
plt.xticks(size=10)
plt.yticks(size=15)
# Create a table to find the mean value for the monthly average temperatures for each region to 3 s.f.
df_month_avg_temp_region = round(df_city_temp[['Month', 'Year','Region','AvgTemperature_Celcius']]
                                 .groupby(['Region', 'Month', 'Year'])['AvgTemperature_Celcius'].agg(np.mean)
                                 .reset_index().sort_values(by=['Year', 'Region']), 3)
# Access the data above
df_month_avg_temp_region.head()
# Create a list for all unique regions
regions_list = df_month_avg_temp_region['Region'].unique().tolist()
# Access thelist
regions_list
# Pivot te data to created separate pivoted dataframes for each region
df_Africa = pd.pivot_table(
        data=df_month_avg_temp_region[df_month_avg_temp_region['Region'] == 'Africa'], 
                                       index='Month',values='AvgTemperature_Celcius',columns='Year')
df_Asia = pd.pivot_table(
        data=df_month_avg_temp_region[df_month_avg_temp_region['Region'] == 'Asia'], 
                                       index='Month',values='AvgTemperature_Celcius',columns='Year')
df_Aus = pd.pivot_table(
        data=df_month_avg_temp_region[df_month_avg_temp_region['Region'] == 'Australia/South Pacific'], 
                                       index='Month',values='AvgTemperature_Celcius',columns='Year')
df_ME = pd.pivot_table(
        data=df_month_avg_temp_region[df_month_avg_temp_region['Region'] == 'Middle East'], 
                                       index='Month',values='AvgTemperature_Celcius',columns='Year')
df_Europe = pd.pivot_table(
        data=df_month_avg_temp_region[df_month_avg_temp_region['Region'] == 'Europe'], 
                                       index='Month',values='AvgTemperature_Celcius',columns='Year')
df_NA = pd.pivot_table(
        data=df_month_avg_temp_region[df_month_avg_temp_region['Region'] == 'North America'], 
                                       index='Month',values='AvgTemperature_Celcius',columns='Year')
df_SA = pd.pivot_table(
        data=df_month_avg_temp_region[df_month_avg_temp_region['Region'] == 'South/Central America & Carribean'], 
                                       index='Month',values='AvgTemperature_Celcius',columns='Year')
# Figure configuration
fig = plt.figure(figsize=(25,15))

# Provide the specifications and visualize the average temperature per month for each region

fig.add_subplot(3,3,1)
sns.heatmap(data=df_Africa, cmap='coolwarm', cbar_kws={'shrink': .5}, annot=True, annot_kws={'fontsize': 12}, vmin=5, vmax=35)
plt.xlabel('')
plt.ylabel('Month', size=15)
plt.title('Africa (°C)', size=15)
plt.xticks(size=10)
plt.yticks(size=10)

fig.add_subplot(3,3,2)
sns.heatmap(data=df_Asia, cmap='coolwarm',  cbar_kws={'shrink': .5}, annot=True, annot_kws={'fontsize': 12}, vmin=5, vmax=35)
plt.xlabel('')
plt.ylabel('Month', size=15)
plt.title('Asia (°C)', size=15)
plt.xticks(size=10)
plt.yticks(size=10)

fig.add_subplot(3,3,3)
sns.heatmap(data=df_Aus, cmap='coolwarm', cbar_kws={'shrink': .5}, annot=True, annot_kws={'fontsize': 12}, vmin=5, vmax=35)
plt.xlabel('')
plt.ylabel('Month', size=15)
plt.title('Australia/South Pacific (°C)', size=15)
plt.xticks(size=10)
plt.yticks(size=10)

fig.add_subplot(3,3,4)
sns.heatmap(data=df_Europe, cmap='coolwarm', cbar_kws={'shrink': .5}, annot=True, annot_kws={'fontsize': 12}, vmin=5, vmax=35)
plt.xlabel('')
plt.ylabel('Month', size=15)
plt.title('Europe (°C)', size=15)
plt.xticks(size=10)
plt.yticks(size=10)


fig.add_subplot(3,3,5)
sns.heatmap(data=df_ME, cmap='coolwarm', cbar_kws={'shrink': .5}, annot=True, annot_kws={'fontsize': 12}, vmin=5, vmax=35)
plt.xlabel('')
plt.ylabel('Month', size=15)
plt.title('Middle East (°C)', size=15)
plt.xticks(size=10)
plt.yticks(size=10)

fig.add_subplot(3,3,6)
sns.heatmap(data=df_NA, cmap='coolwarm', cbar_kws={'shrink': .5}, annot=True, annot_kws={'fontsize': 12}, vmin=5, vmax=35)
plt.xlabel('')
plt.ylabel('Month', size=15)
plt.title('North America', size=15)
plt.xticks(size=10)
plt.yticks(size=10)

fig.add_subplot(3,3,7)
sns.heatmap(data=df_SA, cmap='coolwarm', cbar_kws={'shrink': .5}, annot=True, annot_kws={'fontsize': 12}, vmin=5, vmax=35)
plt.xlabel('')
plt.ylabel('Month', size=15)
plt.title('South/Central America & Carribean (°C)', size=15)
plt.xticks(size=10)
plt.yticks(size=10)

plt.subplots_adjust(wspace = 0.5)
plt.tight_layout()
from IPython.display import display, HTML

base_html = """
<!DOCTYPE html>
<html>
  <head>
  <script type="text/javascript" src="http://kozea.github.com/pygal.js/javascripts/svg.jquery.js"></script>
  <script type="text/javascript" src="https://kozea.github.io/pygal.js/2.0.x/pygal-tooltips.min.js""></script>
  </head>
  <body>
    <figure>
      {rendered_chart}
    </figure>
  </body>
</html>
"""
# Install pygal
!pip3 install pygal
# Install pygal_maps_world
!pip3 install pygal_maps_world
# Import pygal and it's features
import pygal
from pygal_maps_world.maps import World
from pygal_maps_world.i18n import COUNTRIES
from pygal.style import LightColorizedStyle, RotateStyle
# COUNTRIES is a dictionary of keys: values, print out the country_code (keys) and countries (values)
for country_code in sorted(COUNTRIES.keys()):
    print(country_code, COUNTRIES[country_code])
# Convert the dictionary into a dataframe of two columns
df_country_code = pd.DataFrame.from_dict(COUNTRIES.items())
df_country_code.columns = ['Code', 'Country']
# Access the first 5 rows to demonstrate proper conversion
df_country_code.head()
# Set Country as the index
df_country_code = df_country_code.set_index('Country')
df_country_code.head()
# Create a new dataframe for that contains the date as the index and then drop the Month & Day
df_city_temp_ = df_city_temp.set_index('Country')
df_city_temp_ = df_city_temp_.drop(["Month","Day"],axis = 1)
df_city_temp_.head()
# Certain Countries are presented differenly. This corrects the few countries based on the df_country_code to 
# minimize the number of Nan.
df_city_temp_ = df_city_temp_.rename(index = {"US":"United States","Ivory Coast":"Cote d'Ivoire",
                                              "South Korea":"Korea, Republic of",
                                              "North Korea": "Korea, Democratic People's Republic of",
                                              "Venezuela": "Venezuela, Bolivarian Republic of", 
                                              "Vietnam": "Viet Nam", "Taiwan": "Taiwan, Province of China",
                                              "Macedonia": "Macedonia, the former Yugoslav Republic of",
                                              "Tanzania": "Tanzania, United Republic of",
                                              "Laos": "Lao People's Democratic Republic",
                                              "Syria": "Syrian Arab Republic",
                                              "Russia": "Russian Federation", 
                                              "Bolivia": "Bolivia, Plurinational State of",
                                              "Equadon": "Ecuador",
                                              "The Netherlands": "Netherlands",
                                              "Serbia-Montenegro": "Serbia",            
                                              "Myanmar (Burma)":"Myanmar"})
df_city_temp_.head()
# Merge the two dataframes
df_countries_world_code = pd.merge(df_city_temp_, df_country_code, left_index = True , right_index = True , how = "left")
df_countries_world_code.head()
# Verify that the tail end of the data has also been modified to include the right country and code
df_countries_world_code.tail()
# Find all the missing values in the dataframe
missing_values(df_countries_world_code)
# Verify the United States has been chnaged appropriately
df_countries_world_code.loc['United States']
# Drop the State, since we are plotting the entire globe vs the United States. 
df_countries_world_code_ = df_countries_world_code.drop(["State"],axis = 1)
df_countries_world_code_.head()
# Find all the missing values in the dataframe
missing_values(df_countries_world_code_)
# Reset the index of the dataframe
df_countries_world_code__ = df_countries_world_code_.reset_index()
df_countries_world_code__.tail()
# Verify that df_countries_world_code__ will access the correct country format after the change above.
df_countries_world_code__.loc[df_countries_world_code__['Country'] == 'United States']
# Create a table to find the mean value for each year to 3 s.f.
df_countries_world_code__avg = round(df_countries_world_code__[['Country', 'Year', 'Code', 'AvgTemperature_Celcius']]
                                     .groupby(['Country', 'Year', 'Code'], as_index=False).agg(np.mean), 3)
# Access the first 5 rows
df_countries_world_code__avg.head()
# Create a dictionary of all the countries and their codes
country_code_dict = dict(zip(df_countries_world_code__avg['Country'], df_countries_world_code__['Code']))
# Print the dictionary
country_code_dict
# Make new tables for 1995, 2000, 2005, 2010, 2015 & 2019 in pandas
df_countries_world_1995 = df_countries_world_code__avg[df_countries_world_code__avg['Year'] == 1995]
df_countries_world_2000 = df_countries_world_code__avg[df_countries_world_code__avg['Year'] == 2000]
df_countries_world_2005 = df_countries_world_code__avg[df_countries_world_code__avg['Year'] == 2005]
df_countries_world_2010 = df_countries_world_code__avg[df_countries_world_code__avg['Year'] == 2010]
df_countries_world_2015 = df_countries_world_code__avg[df_countries_world_code__avg['Year'] == 2015]
df_countries_world_2019 = df_countries_world_code__avg[df_countries_world_code__avg['Year'] == 2019]

# Just to double check that we can access the first 5 rows of each dataframe for 1995
# change for the other years 2000, 2005, 2010, 2015 and 2019
df_countries_world_1995.head()
# Create a dictionary of all the codes and their AvgTemperature_Celcius for 1995, 2000, 2005, 2010, 2015 & 2019
code_temp_dict_1995 = dict(zip(df_countries_world_1995['Code'], df_countries_world_1995['AvgTemperature_Celcius']))
code_temp_dict_2000 = dict(zip(df_countries_world_2000['Code'], df_countries_world_2000['AvgTemperature_Celcius']))
code_temp_dict_2005 = dict(zip(df_countries_world_2005['Code'], df_countries_world_2005['AvgTemperature_Celcius']))
code_temp_dict_2010 = dict(zip(df_countries_world_2010['Code'], df_countries_world_2010['AvgTemperature_Celcius']))
code_temp_dict_2015 = dict(zip(df_countries_world_2010['Code'], df_countries_world_2010['AvgTemperature_Celcius']))
code_temp_dict_2019 = dict(zip(df_countries_world_2019['Code'], df_countries_world_2019['AvgTemperature_Celcius']))
len(code_temp_dict_1995), len(code_temp_dict_2000), len(code_temp_dict_2005), len(code_temp_dict_2010), len(code_temp_dict_2019)
# Group the temperatures into five categories for 1995.

cc_temp_1_1995, cc_temp_2_1995, cc_temp_3_1995, cc_temp_4_1995, cc_temp_5_1995 = {}, {}, {}, {}, {}
    
for cc, temp in code_temp_dict_1995.items():
    if temp < 0:
        cc_temp_1_1995[cc] = temp
    elif 0 <= temp < 10:
        cc_temp_2_1995[cc] = temp
    elif 10 <= temp < 20:
        cc_temp_3_1995[cc] = temp
    elif 20 <= temp < 30:
        cc_temp_4_1995[cc] = temp
    else:
        cc_temp_5_1995[cc] = temp
        
# Find the length of each of the categories
len(cc_temp_1_1995), len(cc_temp_2_1995), len(cc_temp_3_1995), len(cc_temp_4_1995), len(cc_temp_5_1995)
# Group the temperatures into five categories for 2000.

cc_temp_1_2000, cc_temp_2_2000, cc_temp_3_2000, cc_temp_4_2000, cc_temp_5_2000 = {}, {}, {}, {}, {}
    
for cc, temp in code_temp_dict_2000.items():
    if temp < 0:
        cc_temp_1_2000[cc] = temp
    elif 0 <= temp < 10:
        cc_temp_2_2000[cc] = temp
    elif 10 <= temp < 20:
        cc_temp_3_2000[cc] = temp
    elif 20 <= temp < 30:
        cc_temp_4_2000[cc] = temp
    else:
        cc_temp_5_2000[cc] = temp
        
# Find the length of each of the categories
len(cc_temp_1_2000), len(cc_temp_2_2000), len(cc_temp_3_2000), len(cc_temp_4_2000), len(cc_temp_5_2000)
# Group the temperatures into five categories for 2005.

cc_temp_1_2005, cc_temp_2_2005, cc_temp_3_2005, cc_temp_4_2005, cc_temp_5_2005 = {}, {}, {}, {}, {}
    
for cc, temp in code_temp_dict_2005.items():
    if temp < 0:
        cc_temp_1_2005[cc] = temp
    elif 0 <= temp < 10:
        cc_temp_2_2005[cc] = temp
    elif 10 <= temp < 20:
        cc_temp_3_2005[cc] = temp
    elif 20 <= temp < 30:
        cc_temp_4_2005[cc] = temp
    else:
        cc_temp_5_2005[cc] = temp
        
# Find the length of each of the categories
len(cc_temp_1_2005), len(cc_temp_2_2005), len(cc_temp_3_2005), len(cc_temp_4_2005), len(cc_temp_5_2005)
# Group the temperatures into five categories for 2010.

cc_temp_1_2010, cc_temp_2_2010, cc_temp_3_2010, cc_temp_4_2010, cc_temp_5_2010 = {}, {}, {}, {}, {}
    
for cc, temp in code_temp_dict_2010.items():
    if temp < 0:
        cc_temp_1_2010[cc] = temp
    elif 0 <= temp < 10:
        cc_temp_2_2010[cc] = temp
    elif 10 <= temp < 20:
        cc_temp_3_2010[cc] = temp
    elif 20 <= temp < 30:
        cc_temp_4_2010[cc] = temp
    else:
        cc_temp_5_2010[cc] = temp
        
# Find the length of each of the categories
len(cc_temp_1_2010), len(cc_temp_2_2010), len(cc_temp_3_2010), len(cc_temp_4_2010), len(cc_temp_5_2010)
# Group the temperatures into five categories for 2015.

cc_temp_1_2015, cc_temp_2_2015, cc_temp_3_2015, cc_temp_4_2015, cc_temp_5_2015 = {}, {}, {}, {}, {}
    
for cc, temp in code_temp_dict_2015.items():
    if temp < 0:
        cc_temp_1_2015[cc] = temp
    elif 0 <= temp < 10:
        cc_temp_2_2015[cc] = temp
    elif 10 <= temp < 20:
        cc_temp_3_2015[cc] = temp
    elif 20 <= temp < 30:
        cc_temp_4_2015[cc] = temp
    else:
        cc_temp_5_2015[cc] = temp
        
# Find the length of each of the categories
len(cc_temp_1_2015), len(cc_temp_2_2015), len(cc_temp_3_2015), len(cc_temp_4_2015), len(cc_temp_5_2015)
# Group the temperatures into five categories for 2019.

cc_temp_1_2019, cc_temp_2_2019, cc_temp_3_2019, cc_temp_4_2019, cc_temp_5_2019 = {}, {}, {}, {}, {}
    
for cc, temp in code_temp_dict_2019.items():
    if temp < 0:
        cc_temp_1_2019[cc] = temp
    elif 0 <= temp < 10:
        cc_temp_2_2019[cc] = temp
    elif 10 <= temp < 20:
        cc_temp_3_2019[cc] = temp
    elif 20 <= temp < 30:
        cc_temp_4_2019[cc] = temp
    else:
        cc_temp_5_2019[cc] = temp
        
# Find the length of each of the categories
len(cc_temp_1_2019), len(cc_temp_2_2019), len(cc_temp_3_2019), len(cc_temp_4_2019), len(cc_temp_5_2019)
# Generate Pygal visualizations for all years
# Interactive maps are saved in te folders.
# Year 1995
wm_style = RotateStyle('#336699', base_style=LightColorizedStyle)
wm = World(style=wm_style)
wm.title = 'World Temperatures in 1995, by Country'
wm.add('T < 0 °C', cc_temp_1_1995)
wm.add('0 <= T < 10 °C', cc_temp_2_1995)
wm.add('10 <= T < 20 °C', cc_temp_3_1995)
wm.add('20 <= T < 30 °C', cc_temp_4_1995)
wm.add('T < 30 °C', cc_temp_5_1995)

wm.add('1995', code_temp_dict_1995)

wm.render_to_file('world_temperatures_1995.svg')

# Year 2000
wm_style = RotateStyle('#336699', base_style=LightColorizedStyle)
wm = World(style=wm_style)
wm.title = 'World Temperatures in 2000, by Country'
wm.add('T < 0 °C', cc_temp_1_2000)
wm.add('0 <= T < 10 °C', cc_temp_2_2000)
wm.add('10 <= T < 20 °C', cc_temp_3_2000)
wm.add('20 <= T < 30 °C', cc_temp_4_2000)
wm.add('T < 30 °C', cc_temp_5_2000)

wm.add('2000', code_temp_dict_2000)

wm.render_to_file('world_temperatures_2000.svg')

#Year 2005
wm_style = RotateStyle('#336699', base_style=LightColorizedStyle)
wm = World(style=wm_style)
wm.title = 'World Temperatures in 2005, by Country'
wm.add('T < 0 °C', cc_temp_1_2005)
wm.add('0 <= T < 10 °C', cc_temp_2_2005)
wm.add('10 <= T < 20 °C', cc_temp_3_2005)
wm.add('20 <= T < 30 °C', cc_temp_4_2005)
wm.add('T < 30 °C', cc_temp_5_2005)

wm.add('2005', code_temp_dict_2005)

wm.render_to_file('world_temperatures_2005.svg')

#Year 2010
wm_style = RotateStyle('#336699', base_style=LightColorizedStyle)
wm = World(style=wm_style)
wm.title = 'World Temperatures in 2010, by Country'
wm.add('T < 0 °C', cc_temp_1_2010)
wm.add('0 <= T < 10 °C', cc_temp_2_2010)
wm.add('10 <= T < 20 °C', cc_temp_3_2010)
wm.add('20 <= T < 30 °C', cc_temp_4_2010)
wm.add('T < 30 °C', cc_temp_5_2010)

wm.add('2010', code_temp_dict_2010)

wm.render_to_file('world_temperatures_2010.svg')

#Year 2015
wm_style = RotateStyle('#336699', base_style=LightColorizedStyle)
wm = World(style=wm_style)
wm.title = 'World Temperatures in 2015, by Country'
wm.add('T < 0 °C', cc_temp_1_2015)
wm.add('0 <= T < 10 °C', cc_temp_2_2015)
wm.add('10 <= T < 20 °C', cc_temp_3_2015)
wm.add('20 <= T < 30 °C', cc_temp_4_2015)
wm.add('T < 30 °C', cc_temp_5_2015)

wm.add('2015', code_temp_dict_2015)

wm.render_to_file('world_temperatures_2015.svg')

#Year 2019
wm_style = RotateStyle('#336699', base_style=LightColorizedStyle)
wm = World(style=wm_style)
wm.title = 'World Temperatures in 2019, by Country'
wm.add('T < 0 °C', cc_temp_1_2019)
wm.add('0 <= T < 10 °C', cc_temp_2_2019)
wm.add('10 <= T < 20 °C', cc_temp_3_2019)
wm.add('20 <= T < 30 °C', cc_temp_4_2019)
wm.add('T < 30 °C', cc_temp_5_2019)

wm.add('2019', code_temp_dict_2019)

wm.render_to_file('world_temperatures_2019.svg')