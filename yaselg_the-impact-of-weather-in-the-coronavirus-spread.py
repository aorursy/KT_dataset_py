# Load libraries
import pandas as pd
import numpy as np
# Use the "glob" module to extract pathnames matching a specified pattern
import glob
import calendar
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
# Statistics
from scipy import stats

# import module we'll need to import our custom module
from shutil import copyfile
# copy our file into the working directory (make sure it has .py suffix)
from functions_py import *
# Load the "covid_19_data" dataset 
covid_2019=pd.read_csv("../input/novelcoronavirus2019dataset/covid_19_data.csv")
covid_2019.head()
# Describes the continuous variables
covid_2019.describe()
# Actual data types
covid_2019.dtypes
## Transform the data type to the correct format
# 'Last Update' and 'ObservationDate' to datetime
covid_2019['Last Update']=pd.to_datetime(covid_2019['Last Update'])
covid_2019['ObservationDate']=pd.to_datetime(covid_2019['ObservationDate'])
# 'Confirmed','Deaths','Recovered' to int
covid_2019[['Confirmed','Deaths','Recovered']]=covid_2019[['Confirmed','Deaths','Recovered']].astype('int')
# 'Province/State' and 'Country/Region' to category
covid_2019[['Province/State','Country/Region']]=covid_2019[['Province/State','Country/Region']].astype('category')
covid_2019.dtypes
print('Some general facts about our data:')
print('=> The first day reported in our data was {}.'.format(min(covid_2019['Last Update'])))
print('=> While the last day included is {}.'.format(max(covid_2019['Last Update'])))
print('=> Our data resume the information of the coronavirus spread in {}'.format(max(covid_2019['Last Update']) - min(covid_2019['Last Update'])))
print('=> During these days, a total of {} Province/States had reported at least one case of coronavirus.'.format(len(covid_2019['Province/State'].unique())))
print('=> These Province/States are distributed in {} countries or regions.'.format(len(covid_2019['Country/Region'].unique())))
# Extract the data of the last day
covid_2019_lastDay=covid_2019.loc[covid_2019['ObservationDate']==max(covid_2019['ObservationDate']),:]
covid_2019_lastDay.head()
# Compute the total number of cases by country
cases_by_countries=covid_2019_lastDay.pivot_table(index=['Country/Region'],
                                                  values='Confirmed',
                                                  aggfunc='sum').sort_values(by='Confirmed',
                                                                             ascending=False)
print('The countries with more cases are:\n {}'.format(cases_by_countries.head()))
# Select the city with more cases in the 3 countries with more cases.
countries=['US', 'Italy','Spain']
function = lambda country: covid_2019_lastDay.loc[covid_2019_lastDay['Country/Region']==country,:].sort_values(by='Confirmed',
                                                                             ascending=False).iloc[0,[2,5]]
# Stores the results in a dictionary
result={country: list(function(country)) for country in countries}
print('The cities with more cases for each of the top countries are:\n {}'.format(pd.DataFrame(result)))
# Slice the dataset to show only the information relative to Italy 
covid_2019.loc[covid_2019['Country/Region']=='Italy',:].sort_values(by='Confirmed',
                                                                             ascending=False).head()
# Drop all the information relative  to Italy from "covid_2019"
covid_2019=covid_2019.loc[covid_2019['Country/Region']!='Italy',:]
# Check that the information was droped
covid_2019.loc[covid_2019['Country/Region']=='Italy',:]
# Load the new dataframe with the information about Italy
italy=pd.read_csv("../input/novelcoronavirus2019dataset/covid19_italy_region.csv")
# Print the columns of this data frame
print(italy.columns)
# Create a new dataframe for Italy with only the necesary variables (listed above)
italy=italy[['SNo','Date','RegionName','Country','Date','TotalPositiveCases','Deaths','Recovered']]
# Name the columns as in covid_19
italy.columns=['SNo','ObservationDate','Province/State','Country/Region','Last Update',
               'Confirmed','Deaths','Recovered']
# Concat the two dataframes
covid_2019=pd.concat([covid_2019,italy])
# Rename ITA for Italy
covid_2019['Country/Region'].replace(to_replace='ITA',value='Italy',inplace=True)
covid_2019.loc[covid_2019['Country/Region']=='Italy',:].head()
# Transform data types
covid_2019=transform_dtypes(covid_2019)
# Extract the information about the cities with more cases
_ , cities=cases_country_city(covid_2019)
cities
# List the names of the coldest countries
coldest_countries=['Canada','Russia','Mongolia','Greenland','Sweden','Norway','Finland','Iceland','Austria']
# Pick only the information of the countries in "coldest_countries"
ind=(covid_2019_lastDay['Country/Region'].isin(set(coldest_countries))) 
# Subset and sort the dataframe using the number of confirmed cases
covid_2019_lastDay.loc[ind,:].sort_values('Confirmed',ascending=False).head()
# List of hottest countries 
hottest_countries=['Mali','Burkina Faso','Senegal','Mauritania','Djibouti','Benin','Ghana','Niger',
                  'Cambodia','South Sudan','Qatar','United Arab Emirates','Sudan',
                  'Saint Vincent and the Grenadines','Togo']
# Pick only the information of the countries in "hottest_countries"
ind=(covid_2019_lastDay['Country/Region'].isin(set(hottest_countries)))
# Subset and sort the dataframe using the number of confirmed cases
covid_2019_lastDay.loc[ind,:].sort_values('Confirmed',ascending=False).head()
weather_NewYork=pd.read_csv("../input/weather/NewYork_December2019_March_2020.csv")
weather_NewYork.head()
# Extract the directories
directories=glob.glob("../input/weather/*.csv")
# Create an empty dataframe to store the information
weather=pd.DataFrame()
# Include the new data in "weather" for each csv file in the directory
for file in directories:
    this_data=pd.read_csv(file)
    weather=pd.concat([weather,this_data],axis=0)
weather.head()
# Create a dictionary with the names of the months and the number that represent it.
d = dict((v,k) for k,v in enumerate(calendar.month_name))
# Replace the variable 'Month' using the dictionary
weather['Month']=weather['Month'].map(d)
weather.head()
# Create a new variable called "Infection Day" (note that I name this variable as in the 
# covid data frame to make clear that I am going the merge this dataframes using this variable)
weather['Infection Day']=pd.to_datetime(weather[['Year', 'Month', 'Day']]).dt.date
# Drop the information relative to the Day, Month and Year
weather.drop(columns=['Day','Month','Year'],inplace=True)
# Convert the 'Country' and 'State' features from objects to category variables
weather[['Country','State']]=weather[['Country','State']].astype('category')
weather.head()
# Print some basic exploration statistics
print('=> The data frame with the weather information is composed by {} rows and {} columns.'.format(weather.shape[0],
                                                                                                   weather.shape[1]))
print('=> The countries included in this dataframe are:\n {}'.format(weather['Country'].unique()))
print('=> The cities included in this dataframe are:\n {}'.format(weather['State'].unique()))
print('=> The total number of Missing Values are: {}'.format(weather.isna().sum().sum()))
# Filter only the observations of the selected countries
selected_countries=['US','Italy','Austria', 'Canada', 'Sweden', 'Qatar', 
                    'United Arab Emirates', 'Senegal', 'Spain']
covid_2019_countries=covid_2019.loc[covid_2019['Country/Region'].isin(selected_countries),:].copy()

# Include the cities in the selected countries without a city level information
countries_without_cities={'Austria': 'Vienna', 'Sweden': 'Stockholm',
                          'Qatar': 'Doha', 'United Arab Emirates': 'Dubai', 
                          'Senegal': 'Dakar', 'Spain':'Madrid'}
covid_2019_countries.loc[:,'Province/State'] = covid_2019_countries.apply(
    lambda row: countries_without_cities[row['Country/Region']] if 
    row['Country/Region'] in countries_without_cities.keys() else row['Province/State'],
    axis=1
)
# Check that we don't have missing information in the "Province/State" feature
print('The number of missing values in the Province/State feature is: {} ==> Great!!'.format(covid_2019_countries['Province/State'].isna().sum()))
# Select only the information relative to the selected province/state
cities=['New York','Madrid','Quebec','Lombardia','Vienna','Stockholm',
       'Doha','Dubai','Dakar']
covid_final=covid_2019_countries.loc[covid_2019_countries['Province/State'].isin(cities),:].copy()
print('=> The cities available in the reduced dataframe are:\n {} ==> Nice, everything looks fine'.format(covid_final['Province/State'].unique()))
print('=> The countries available in the reduced dataframe are:\n {} ==> Nice!'.format(list(covid_final['Country/Region'].unique())))
print('=> So far, the information about the cities of interests is contained in {} rows and {} columns.'.format(covid_final.shape[0],covid_final.shape[1]))
print('=> The new dataset has {} missing values'.format(covid_final.isna().sum().sum()))
covid_final.head()
# The number of new cases in the a day "d" (N_d) can be computed as [N_d - N_(d-1)]. 
# Remember that we need to do this by city.
# Iterate ove the cities and compute the number of new cases per day
covid_new_cases=pd.DataFrame()
for city in cities:
    # Subset the dataset to considder only one city
    temp=covid_final.loc[covid_final['Province/State']==city,:].sort_values(by='ObservationDate')
    # Transform the variable "Confirmed" to include only the information 
    # about the new infections by day (not the cumulative)
    temp.loc[temp['ObservationDate']>min(temp['ObservationDate']),
             'Confirmed'] = temp['Confirmed'][1:].values - temp['Confirmed'][:-1].values
    
    # Create a new variable "Days Since First Case" where 0 is the day when 
    # the first infection was reported and N is the last day where was 
    # recorded information about new cases in "city"
    diff_dates=temp.loc[:,'ObservationDate'].values - temp.iloc[0,1] # Difference between the first and k dates
    temp['Days Since First Case'] =[tt.days for tt in diff_dates] # Include only the information about the days
    
    # Concatenate the result with the "covid_new_cases" dataframe
    covid_new_cases=pd.concat([covid_new_cases,temp])
# Print a piece of "covid_new_cases" dataframe
covid_new_cases.head()
# Resume in test1 the sum of the new cases by cities
test1=covid_new_cases.pivot_table(index=['Province/State'],values='Confirmed',aggfunc='sum')
# Extract in test2 the number of cases the last day
test2=covid_final.loc[covid_final['ObservationDate']==max(covid_final['ObservationDate']),['Province/State','Confirmed']]
# Merge and show this information
pd.merge(test1,test2,on='Province/State',suffixes=('_cumulative (Last Day)', '_sum (new cases per day)'))
# Estimate the infection day
covid_new_cases['Infection Day']=covid_new_cases['ObservationDate'] -  pd.to_timedelta(8,'d')
# Shows the new results
covid_new_cases.head()
# Left Join the two data frames
covid_weather=pd.merge(covid_new_cases,weather,how='left',left_on=['Infection Day','Province/State'],
                                            right_on=['Infection Day','State'])
# Some variables like SNo, State (is a duplication of "Province/State"), 
# Country (is a duplication of "Country/Region") or "LastUpdate" are not 
# necessary to this study, so let's drop it from the data.
covid_weather.drop(columns=['SNo','State','Country','Last Update'],inplace=True)
covid_weather.head()
for city in cities:
    print('=> The data frames have a {} match between the number of observations in {}'.format(
        covid_weather.loc[covid_weather['Province/State']==city,:].shape[0]==
        covid_new_cases.loc[covid_new_cases['Province/State']==city,:].shape[0],city))
print('=> The final data frame that condense all the information about the coronavirus disease and the weather in the selected 9 cities has {} observations and {} features.'.
     format(covid_weather.shape[0],covid_weather.shape[1]))
print('=> The total number of missing values in the data frame is {} ==> Great!!'.format(covid_weather.isna().sum().sum()))
# PLot the Temperature Avg by day
px.line(covid_weather, x='Infection Day', y='TempAvg', color='Province/State',
       title='Average Temperature by Day')
# Scatter plot between the Average Temperature and the number of Cases by Province/State
px.scatter(covid_weather, x="TempAvg", y="Confirmed", color="Province/State",
                 marginal_y=None, marginal_x="box", trendline="o",
          title='New Infections vs Temperature')
# Import k-means from sklearn
from sklearn.cluster import KMeans
# Extract the information  about the temperatures
X=np.array(covid_weather['TempAvg'])
# Cluster
kmeans = KMeans(n_clusters=3, random_state=0).fit(X.reshape(-1,1))
# Include the labels in our data frame in the variable "Cluster_Temp"
covid_weather['Cluster_Temp']=kmeans.labels_
# Compute the min and max temperature values in each cluster
covid_weather.pivot_table(index='Cluster_Temp',values='TempAvg',aggfunc=['min','max'])
# Dictionary with the new labels
dic={0:'40-60 F', 1: '>60 F', 2: '<40 F'}
# Replace the labels
covid_weather['Cluster_Temp'].replace(dic,inplace=True)
# Plot the clusters
px.scatter(covid_weather, x="TempAvg", y="Cluster_Temp", color="Cluster_Temp",
                 marginal_y=None, marginal_x=None, trendline="o",
          width=900, height=300)
# Histogram of the number of infections by group of temperature
px.bar(covid_weather, x="Cluster_Temp", y="Confirmed", 
       color="Province/State", title='Temperature ranges and New Infections')
# Number of days that each Province/State had for each range of temperature
covid_weather.pivot_table(index='Province/State',columns='Cluster_Temp',
                          values='Days Since First Case',aggfunc='count')
# Create a data frame with the Region/State population and Land Area
region_state_density=pd.DataFrame({'Region/State':['Dakar', 'Doha', 'Dubai',
                             'Lombardia','Madrid','New York',
                             'Quebec','Stockholm','Vienna'], 
              'Population': [2956023,2382000,3331420, 10078012,
                             3223334,19453561,8164361,2377081,1888776 ],
             'Land Area (sq mi)': [211,51,1588,9206,233.3,54555,595391,2517,160.15]})
# Compute the population density as Area/population
region_state_density['Population Density']=region_state_density['Population']/region_state_density['Land Area (sq mi)']
region_state_density.sort_values(by=['Population Density'],ascending=False,inplace=True)
region_state_density
# Histogram of the new cases by cities
fig1, axs = plt.subplots(3, 3, figsize=(10,8), constrained_layout=True)
axs = trim_axs(axs, len(cities))
for ax, city in zip(axs, cities):
    X=covid_weather.loc[covid_weather['Province/State']==city,'Confirmed']
    ax.set_title('{} ({} days observed)'.format(city,len(X)))
    sns.distplot(X,kde=False,ax=ax,bins=40)
fig1.suptitle('New Infections Histogram by Province/State', fontsize=16);
## t-student Hypothesis tests ##
# Create a dictionary with the pairs of cities to be tested
cities2test=dict({'Dakar': ['New York','Madrid','Lombardia','Vienna','Stockholm','Quebec'],
                 'Doha': ['New York','Madrid','Lombardia','Vienna','Stockholm','Quebec'],
                 'Dubai': ['New York','Lombardia','Stockholm','Quebec']})
# Run the tests (use the function "t_test_byCities" available in "functions.py") 
results_pvalue, results_stat=t_test_byCities(cities2test,covid_weather)

print('The p-values are:')
results_pvalue.style.applymap(color_p_value)
print('The t-statistics are:')
results_stat.style.applymap(color_p_value)
# Extract only the information of Dakar, Doha, and Madrid
Dakar_Doha_Madrid=covid_weather.loc[covid_weather['Province/State'].isin(['Doha','Dakar','Madrid']),:]
# Plot the number of new cases and temperature for Madrid, Dakar and Doha
g=sns.pairplot(x_vars="Days Since First Case", aspect=3,
             y_vars=["Confirmed","TempAvg"], kind='scatter', hue="Province/State",
             data=Dakar_Doha_Madrid);
g.fig.suptitle("New Infections and Average Temperature by Day (Madrid, Dakar, Doha)");    
Dakar_Doha_Stockholm=covid_weather.loc[covid_weather['Province/State'].isin(['Doha','Dakar','Stockholm']),:]
# Plot the number of new cases and temperature in Stockholm, Dakar and Doha
g=sns.pairplot(x_vars="Days Since First Case", aspect=3,
             y_vars=["Confirmed","TempAvg"], kind='scatter', hue="Province/State",
             data=Dakar_Doha_Stockholm);
g.fig.suptitle("New Infections and Average Temperature by Day (Stockholm, Dakar, Doha)");  
# Plot the number of new infections by day and region
px.scatter(covid_weather, x="Days Since First Case",y="Confirmed",color="Province/State",
          title='New Infections by Day')
# Min and Max Humidity Average values by Province/State
covid_weather.pivot_table(index="Province/State",values="HumAvg", aggfunc=['min','max'])
# PLot the number of infections in relationship with the Days since the first infection and the Humidity Avg
px.scatter(covid_weather, x="Days Since First Case", 
           y="Confirmed", size="HumAvg",color="Province/State",
          title="Number of New Infections by Day and Humidity Average")
# Number of cases by Province/State
covid_weather.pivot_table(index="Province/State",values='HumAvg',aggfunc='count')
# Reduce each group to only the first 25 observations
reduced_data=covid_weather.loc[(covid_weather['Days Since First Case']>=0) & 
                 (covid_weather['Days Since First Case']<=24),:]
reduced_data.pivot_table(index='Province/State',values='Days Since First Case',aggfunc=['min','max'])
# Create a list of 1-D arrays with the information of the Average Humidity.  
data = [reduced_data.loc[ids, 'HumAvg'].values for ids in 
        reduced_data.groupby('Province/State').groups.values()]
# Run the Levene's test for the homeostasis of the variance
from scipy import stats
print(stats.levene(*data))
# Kruskal-Wallis H hypothesis test (analysis of the variance)
stats.kruskal(*data)
# Use the library scikit_posthocs to the posthoc test
import scikit_posthocs as sp
result=sp.posthoc_dunn(reduced_data,val_col='HumAvg',p_adjust='bonferroni',group_col='Province/State')
# Plot the results as a heatmap
sp.sign_plot(result);
# Boxplot of the pressure by Province/State
px.box(covid_weather, x='Province/State', y='Pressure_Avg',
       title='Boxplot Pressure by Province/State')
# Create a list of 1-D arrays with the information of the Average Humidity.  
data = [reduced_data.loc[ids, 'TempAvg'].values for ids in 
        reduced_data.groupby('Province/State').groups.values()]
# Levene's test
print(stats.levene(*data))
print('The test reveals that there are statistically significant differences between the variance of the temperature in different cities.')
# Kruskal-Wallis H hypothesis test (analysis of the variance)
print(stats.kruskal(*data))
print('The Kruskal-Wallis test shows that there are differences in the distribution of the temperature across different Province/State.')
# Dunn Posthoc test with Bonferroni correction 
result=sp.posthoc_dunn(covid_weather,val_col='TempAvg',p_adjust='bonferroni',group_col='Province/State')
sp.sign_plot(result);
# Create a dictionary with the combinations of cities
cities2test=dict({'Madrid': ['New York','Lombardia','Vienna']})
# Run the tests
results_pvalue, results_stat=t_test_byCities(cities2test,covid_weather)
results_pvalue['t-stats']=results_stat['Madrid']
results_pvalue=results_pvalue.rename(columns={'Madrid':'p-value'})
results_pvalue
# Plot the number of cases by day only for Madrid, Lombardia, New York and Vienna.
g=sns.relplot(x='Days Since First Case',y='Confirmed',hue='Province/State',
            data=covid_weather.loc[covid_weather['Province/State'].isin(['Madrid',
                                                                         'New York','Lombardia',
                                                                         'Vienna']),:]);
g.fig.suptitle("New Infections by Day (Madrid, New York, Lombardia, Vienna)");