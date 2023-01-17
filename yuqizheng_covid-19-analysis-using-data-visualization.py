!pip install pycountry_convert
import pandas as pd # Load libaries

import numpy as np

import pycountry_convert as pc

import plotly.express as px

import os

import plotly

import plotly.graph_objs as go

from IPython.display import Image
Image(filename='../input/data-viz-files/screenshot_visualization1.JPG')
Image(filename='../input/data-viz-files/screenshot_visualization2.JPG')
Image(filename='../input/data-viz-files/screenshot_visualization2b.JPG')
Image(filename='../input/data-viz-files/screenshot_visualization3.JPG')
Image(filename='../input/data-viz-files/screenshot_visualization4.JPG')
Image(filename='../input/data-viz-files/screenshot_visualization4b.JPG')
Image(filename='../input/data-viz-files/screenshot_visualization5b.jpg')
Image(filename='../input/data-viz-files/screenshot_visualization5.JPG')
Image(filename='../input/data-viz-files/screenshot_visualization6.JPG')
Image(filename='../input/data-viz-files/screenshot_visualization6b.JPG')
Image(filename='../input/data-viz-files/11.jpg')
Image(filename='../input/data-viz-files/12.jpg')
Image(filename='../input/data-viz-files/13.jpg')
Image(filename='../input/data-viz-files/14.JPG')
Image(filename='../input/data-viz-files/15.JPG')
Image(filename='../input/data-viz-files/16.JPG')
Image(filename='../input/data-viz-files/17.JPG')
Image(filename='../input/data-viz-files/18.JPG')
Image(filename='../input/data-viz-files/19.JPG')
Image(filename='../input/data-viz-files/20.JPG')
Image(filename='../input/data-viz-files/21.JPG')
Image(filename='../input/data-viz-files/22.JPG')
Image(filename='../input/data-viz-files/23.JPG')
Image(filename='../input/data-viz-files/24.JPG')
Image(filename='../input/data-viz-files/25.JPG')
Image(filename='../input/data-viz-files/26.JPG')
Image(filename='../input/data-viz-files/27.JPG')
Image(filename='../input/data-viz-files/28.JPG')
Image(filename='../input/data-viz-files/29.JPG')
Image(filename='../input/data-viz-files/30.JPG')
import pandas as pd # Load libaries

import numpy as np

import pycountry_convert as pc

import plotly.express as px

import os

import plotly

import plotly.graph_objs as go



ts_confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv') # Load data



# Remove certain counts for convenience

ts_confirmed = ts_confirmed[~ts_confirmed['Country/Region'].isin(['Congo (Brazzaville)'])]

ts_confirmed = ts_confirmed[~ts_confirmed['Country/Region'].isin(['Diamond Princess'])]

ts_confirmed = ts_confirmed[~ts_confirmed['Country/Region'].isin(['MS Zaandam'])]

ts_confirmed = ts_confirmed[~ts_confirmed['Country/Region'].isin(['Holy See'])]

ts_confirmed = ts_confirmed[~ts_confirmed['Country/Region'].isin(['Western Sahara'])]

ts_confirmed = ts_confirmed[~ts_confirmed['Country/Region'].isin(['Kosovo'])]

ts_confirmed = ts_confirmed[~ts_confirmed['Country/Region'].isin(['Cruise Ship'])]

ts_confirmed = ts_confirmed[~ts_confirmed['Country/Region'].isin(['Timor-Leste'])]

ts_confirmed = ts_confirmed[~ts_confirmed['Country/Region'].isin(['West Bank and Gaza'])]



# Change country name to full name for US

ts_confirmed.iloc[:,1][ts_confirmed.iloc[:,1] == 'US'] = 'United States of America'



iso_list = [] # Create ISO column for location plotting

for i in range(0,len(ts_confirmed)): # Manual ISO labeling

    if ts_confirmed.iloc[i,1] == 'Korea, South':

        iso_list.append('KOR')

    elif ts_confirmed.iloc[i,1] == 'Taiwan*':

        iso_list.append('TWN')

    elif ts_confirmed.iloc[i,1] == 'Congo (Kinshasa)':

        iso_list.append('COD')

        ts_confirmed.iloc[i,1] = 'Congo'

#    elif ts_confirmed.iloc[i,1] == 'Congo (Brazzaville)':

#        iso_list.append('COD')

#        ts_confirmed.iloc[i,1] = 'Congo'

    elif ts_confirmed.iloc[i,1] == 'Cote d\'Ivoire':

        iso_list.append('CIV')

    elif ts_confirmed.iloc[i,1] == 'Gambia, The':

        iso_list.append('GMB')

    elif ts_confirmed.iloc[i,1] == 'Bahamas, The':

        iso_list.append('BHS')

    elif ts_confirmed.iloc[i,1] == 'West Bank and Gaza':

        iso_list.append('PS')

    elif ts_confirmed.iloc[i,1] == 'Burma':

        iso_list.append('MMR')

    else:

        iso_list.append(pc.country_name_to_country_alpha3(ts_confirmed.iloc[i,1], cn_name_format="default"))



geospatial = ts_confirmed.copy() # Create new df to work with

geospatial['ISO'] = iso_list

geospatial.head()



# References: https://stackoverflow.com/questions/28654047/pandas-convert-some-columns-into-rows

# Change dataframe from long format to wide format

geospatial_melt = geospatial.melt(id_vars=geospatial.columns[[0,1,2,3,(len(geospatial.columns) - 1)]],

                 var_name='Date',

                 value_name='Confirmed Cases')



# Change some country names for continent labels

#geospatial_melt.loc[geospatial_melt['Country/Region'] == 'Congo (Brazzaville)','Country/Region'] = 'Congo'

geospatial_melt.loc[geospatial_melt['Country/Region'] == 'Congo (Kinshasa)','Country/Region'] = 'Congo'

geospatial_melt.loc[geospatial_melt['Country/Region'] == 'Cote d\'Ivoire','Country/Region'] = 'Ivory Coast'

geospatial_melt.loc[geospatial_melt['Country/Region'] == 'Korea, South','Country/Region'] = 'South Korea'

geospatial_melt.loc[geospatial_melt['Country/Region'] == 'Taiwan*','Country/Region'] = 'Taiwan'

geospatial_melt.loc[geospatial_melt['Country/Region'] == 'Kosovo','Country/Region'] = 'Serbia'

geospatial_melt.loc[geospatial_melt['Country/Region'] == 'Burma','Country/Region'] = 'Myanmar'

geospatial_melt.drop(geospatial_melt[geospatial_melt['Country/Region'] == 0].index, inplace=True)



# Reference: https://stackoverflow.com/questions/55910004/get-continent-name-from-country-using-pycountry

continents = { # Add continent column

    'NA': 'North America',

    'SA': 'South America', 

    'AS': 'Asia',

    'OC': 'Australia',

    'AF': 'Africa',

    'EU': 'Europe'

}

geospatial_melt['Continent'] = [continents[pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))] for country in geospatial_melt['Country/Region']]



# Combine province/state from countries into single rows with groupby

# Reference: https://stackoverflow.com/questions/33068007/pandas-keeping-dates-in-order-when-using-groupby-or-pivot-table

geo_groupby = geospatial_melt.groupby(['Date', 'Continent', 'Country/Region', 'ISO'],

                                      sort=False)['Confirmed Cases'].sum().reset_index()



fig = px.scatter_geo(geo_groupby, locations="ISO", color="Continent",

                     hover_name="Country/Region", size="Confirmed Cases",

                     animation_frame="Date",

                     height = 1000,

                     template='plotly_dark',

                     title='Timeseries of COVID-19 Pandemic by Country',

                     projection="natural earth")

fig.update(layout=dict(title=dict(x=0.5)))

fig.show()



# Reference: https://community.plot.ly/t/proper-way-to-save-a-plot-to-html/7063/8

cwd = os.getcwd()

#fig.write_html(cwd + '\\visualization1.html')
import pandas as pd # Load libaries

import numpy as np

import pycountry_convert as pc

import plotly.express as px

import os

import plotly

import plotly.graph_objs as go



df_viz2 = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv') # Load data



### The melt_groupby_lowerbound function will melt the original timeseries

### dataframe from long format to wide format. It will also do a groupby

### to keep only Country/Region and Date columns. It also adds the columns

### of 'Days' and 'Log Confirmed Cases'. There's also the option to subset

### countries with a lower bound number of cases on the latest date.

def melt_groupby_lowerbound(untouched_data=df_viz2, lower_bound_cases=5000):

    # Drop Lat and Long columns

    untouched_data.drop(['Lat', 'Long'], axis=1, inplace=True)

    

    # Melt from long to wide format keeping only ['Country/Region','Date']

    subset_melt = untouched_data.melt(id_vars=untouched_data.columns[[0,1]],

                 var_name='Date',

                 value_name='Confirmed Cases')

    

    # Reference: https://stackoverflow.com/questions/40553002/pandas-group-by-two-columns-to-get-sum-of-another-column

    # Reference: https://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-output-from-series-to-dataframe

    # Groupby ['Country/Region','Date'] and sum the rows to combine states and provinces

    df_groupby_country_date = subset_melt.groupby(['Country/Region','Date']).agg({'Confirmed Cases': 'sum'}).reset_index()



    # Reference: https://stackoverflow.com/questions/28161356/sort-pandas-dataframe-by-date

    # Reference: https://stackoverflow.com/questions/17141558/how-to-sort-a-dataframe-in-python-pandas-by-two-or-more-columns

    # Re-sort values after doing groupby

    df_groupby_country_date['Date'] = pd.to_datetime(df_groupby_country_date.Date, format='%m/%d/%y')

    df_groupby_country_date.sort_values(by=['Country/Region', 'Date'], ascending=[True, True], inplace=True)

    df_groupby_country_date.reset_index(inplace=True, drop=True)



    # Reference: https://stackoverflow.com/questions/59642338/creating-new-column-based-on-condition-on-other-column-in-pandas-dataframe

    # Add 'Days' column which repeats 1:n_dates per country by mapping dates to corresponding nth day

    unique_dates = df_groupby_country_date['Date'].unique()

    unique_dates_df = pd.DataFrame({'Dates': unique_dates})

    unique_dates_df['Days'] = [i for i in range(1, len(unique_dates_df) + 1)]

    df_groupby_country_date['Days'] = [unique_dates_df[x == unique_dates_df['Dates']]['Days'].values[0]

                                       for x in df_groupby_country_date['Date']]

    

    # Add log confirmed cases

    df_groupby_country_date['Log Confirmed Cases'] = np.log(df_groupby_country_date['Confirmed Cases'])

    

    # Subset countries with lower bound confirmed cases at current date

    # Reference: https://stackoverflow.com/questions/22591174/pandas-multiple-conditions-while-indexing-data-frame-unexpected-behavior

    lower_bound_countries = df_groupby_country_date[(df_groupby_country_date['Days'] == int(df_groupby_country_date['Days'].tail(1))) &

                            (df_groupby_country_date['Confirmed Cases'] >= lower_bound_cases)]['Country/Region']

    # Reference: https://stackoverflow.com/questions/17071871/how-to-select-rows-from-a-dataframe-based-on-column-values

    lower_bound_subset = df_groupby_country_date.loc[df_groupby_country_date['Country/Region'].isin(lower_bound_countries)]

    lower_bound_subset.reset_index(drop=True, inplace=True)



    return lower_bound_subset



# Subset countries with at least 50,000 cases

df_line_chart = melt_groupby_lowerbound(untouched_data=df_viz2, lower_bound_cases=50000)



df_line_chart['Country/Region'].replace('US', 'United States of America', inplace=True) # edit US to United States...



# df_line_chart.to_csv('python_melt.csv', index=False) # Save data to file for use in RStudio



### The k_lag_subset function is used to subset the data so that it creates a k'th lag in the data

### from the date of the first infection. It modifies the dataset so that the countries only

### show confirmed cases for 'k' days after their first infection. This helps to stagger the

### countries on a similar start date so that they can be compared.

def k_lag_subset(full_line_chart_df=df_line_chart, k_lags=0):

    df = full_line_chart_df.copy()

    subset_countries = df['Country/Region'].unique()

    total_days = np.max(df['Days'])

    lag_list = []

    for i in subset_countries:

        # Subset by i'th country

        country_subset = df[df['Country/Region'] == i]

        # Find row index of k'th day lag

        first_case_lag = np.where(country_subset['Confirmed Cases'] > 0)[0][0] + k_lags

        # Subset df by k'th row

        lag_subset = country_subset.iloc[first_case_lag:total_days]

        lag_subset['Lag Days'] = [i for i in range(1, lag_subset['Days'].tail(1).values[0] -

                                                   lag_subset['Days'].head(1).values[0] + 2)]

        # Append subset df

        lag_list.append(lag_subset)

    

    k_lag_df = pd.concat(lag_list) # Concatenate back to single df

    

    return k_lag_df



# Lag up to k=10 days

lag_df_k10 = k_lag_subset(full_line_chart_df=df_line_chart, k_lags=10)



# lag_df_k10.to_csv('lag_data.csv', index=False) # save to df to work with in R



fig = px.line(lag_df_k10, x='Lag Days', y='Confirmed Cases', color='Country/Region',

             hover_name='Country/Region', height=1000, title='Confirmed Cases vs. Lag Days')

fig.update(layout=dict(title=dict(x=0.5)))

fig.show()



# Reference: https://community.plot.ly/t/proper-way-to-save-a-plot-to-html/7063/8

cwd = os.getcwd()

# fig.write_html(cwd + '\\visualization2.html')
import pandas as pd # Load libaries

import numpy as np

import pycountry_convert as pc

import plotly.express as px

import os

import plotly

import plotly.graph_objs as go



df_viz3 = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv') # Load data



### The melt_groupby_lowerbound function will melt the original timeseries

### dataframe from long format to wide format. It will also do a groupby

### to keep only Country/Region and Date columns. It also adds the columns

### of 'Days' and 'Log Confirmed Cases'. There's also the option to subset

### countries with a lower bound number of cases on the latest date.

def melt_groupby_lowerbound(untouched_data=df_viz3, lower_bound_cases=5000):

    # Drop Lat and Long columns

    untouched_data.drop(['Lat', 'Long'], axis=1, inplace=True)

    

    # Melt from long to wide format keeping only ['Country/Region','Date']

    subset_melt = untouched_data.melt(id_vars=untouched_data.columns[[0,1]],

                 var_name='Date',

                 value_name='Confirmed Cases')

    

    # Reference: https://stackoverflow.com/questions/40553002/pandas-group-by-two-columns-to-get-sum-of-another-column

    # Reference: https://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-output-from-series-to-dataframe

    # Groupby ['Country/Region','Date'] and sum the rows to combine states and provinces

    df_groupby_country_date = subset_melt.groupby(['Country/Region','Date']).agg({'Confirmed Cases': 'sum'}).reset_index()



    # Reference: https://stackoverflow.com/questions/28161356/sort-pandas-dataframe-by-date

    # Reference: https://stackoverflow.com/questions/17141558/how-to-sort-a-dataframe-in-python-pandas-by-two-or-more-columns

    # Re-sort values after doing groupby

    df_groupby_country_date['Date'] = pd.to_datetime(df_groupby_country_date.Date, format='%m/%d/%y')

    df_groupby_country_date.sort_values(by=['Country/Region', 'Date'], ascending=[True, True], inplace=True)

    df_groupby_country_date.reset_index(inplace=True, drop=True)



    # Reference: https://stackoverflow.com/questions/59642338/creating-new-column-based-on-condition-on-other-column-in-pandas-dataframe

    # Add 'Days' column which repeats 1:n_dates per country by mapping dates to corresponding nth day

    unique_dates = df_groupby_country_date['Date'].unique()

    unique_dates_df = pd.DataFrame({'Dates': unique_dates})

    unique_dates_df['Days'] = [i for i in range(1, len(unique_dates_df) + 1)]

    df_groupby_country_date['Days'] = [unique_dates_df[x == unique_dates_df['Dates']]['Days'].values[0]

                                       for x in df_groupby_country_date['Date']]

    

    # Add log confirmed cases

    df_groupby_country_date['Log Confirmed Cases'] = np.log(df_groupby_country_date['Confirmed Cases'])

    

    # Subset countries with lower bound confirmed cases at current date

    # Reference: https://stackoverflow.com/questions/22591174/pandas-multiple-conditions-while-indexing-data-frame-unexpected-behavior

    lower_bound_countries = df_groupby_country_date[(df_groupby_country_date['Days'] == int(df_groupby_country_date['Days'].tail(1))) &

                            (df_groupby_country_date['Confirmed Cases'] >= lower_bound_cases)]['Country/Region']

    # Reference: https://stackoverflow.com/questions/17071871/how-to-select-rows-from-a-dataframe-based-on-column-values

    lower_bound_subset = df_groupby_country_date.loc[df_groupby_country_date['Country/Region'].isin(lower_bound_countries)]

    lower_bound_subset.reset_index(drop=True, inplace=True)



    return lower_bound_subset



# Subset countries with at least 50,000 cases

df_line_chart = melt_groupby_lowerbound(untouched_data=df_viz3, lower_bound_cases=50000)

df_line_chart['Country/Region'].replace('US', 'United States of America', inplace=True) # edit US to United States...



### The k_lag_subset function is used to subset the data so that it creates a k'th lag in the data

### from the date of the first infection. It modifies the dataset so that the countries only

### show confirmed cases for 'k' days after their first infection. This helps to stagger the

### countries on a similar start date so that they can be compared.

def k_lag_subset(full_line_chart_df=df_line_chart, k_lags=0):

    df = full_line_chart_df.copy()

    subset_countries = df['Country/Region'].unique()

    total_days = np.max(df['Days'])

    lag_list = []

    for i in subset_countries:

        # Subset by i'th country

        country_subset = df[df['Country/Region'] == i]

        # Find row index of k'th day lag

        first_case_lag = np.where(country_subset['Confirmed Cases'] > 0)[0][0] + k_lags

        # Subset df by k'th row

        lag_subset = country_subset.iloc[first_case_lag:total_days]

        lag_subset['Lag Days'] = [i for i in range(1, lag_subset['Days'].tail(1).values[0] -

                                                   lag_subset['Days'].head(1).values[0] + 2)]

        # Append subset df

        lag_list.append(lag_subset)

    

    k_lag_df = pd.concat(lag_list) # Concatenate back to single df

    

    return k_lag_df



# Lag up to k=10 days

lag_df_k10 = k_lag_subset(full_line_chart_df=df_line_chart, k_lags=10)



# Reference: https://github.com/plotly/plotly_express/issues/52

fig = px.line(lag_df_k10, x='Lag Days', y='Log Confirmed Cases', color='Country/Region',

             hover_name='Country/Region', height=1000, title='Log Confirmed Cases vs. Lag Days')

fig.update(layout=dict(title=dict(x=0.5)))

fig.show()



# Reference: https://community.plot.ly/t/proper-way-to-save-a-plot-to-html/7063/8

cwd = os.getcwd()

# fig.write_html(cwd + '\\visualization3.html')
import pandas as pd # Load libaries

import numpy as np

import pycountry_convert as pc

import plotly.express as px

import os

import plotly

import plotly.graph_objs as go



sunburst_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv') # Load data



latest_date = max(sunburst_data['ObservationDate']) # Find latest date



# Subset 'Province/State', 'Country/Region','Confirmed' from the latest date

sunburst_sub = sunburst_data[sunburst_data['ObservationDate'] == latest_date][['Province/State', 'Country/Region','Confirmed']]



# Remove certain counts for convenience

sunburst_sub = sunburst_sub[~sunburst_sub['Country/Region'].isin(['Congo (Brazzaville)'])]

sunburst_sub = sunburst_sub[~sunburst_sub['Country/Region'].isin(['Diamond Princess'])]

sunburst_sub = sunburst_sub[~sunburst_sub['Country/Region'].isin(['MS Zaandam'])]

sunburst_sub = sunburst_sub[~sunburst_sub['Country/Region'].isin(['Holy See'])]

sunburst_sub = sunburst_sub[~sunburst_sub['Country/Region'].isin(['Western Sahara'])]

sunburst_sub = sunburst_sub[~sunburst_sub['Country/Region'].isin(['Kosovo'])]

sunburst_sub = sunburst_sub[~sunburst_sub['Country/Region'].isin(['Cruise Ship'])]

sunburst_sub = sunburst_sub[~sunburst_sub['Country/Region'].isin(['Timor-Leste'])]

sunburst_sub = sunburst_sub[~sunburst_sub['Country/Region'].isin(['West Bank and Gaza'])]



# Change country name to alternate name for US, UK, and China

sunburst_sub.iloc[:,1][sunburst_sub.iloc[:,1] == 'US'] = 'United States of America'

sunburst_sub.iloc[:,1][sunburst_sub.iloc[:,1] == 'UK'] = 'United Kingdom'

sunburst_sub.iloc[:,1][sunburst_sub.iloc[:,1] == 'Mainland China'] = 'China'



iso_list = [] # Create ISO column for location plotting

for i in range(0,len(sunburst_sub)): # Manual ISO labeling

    if sunburst_sub.iloc[i,1] == 'Korea, South':

        iso_list.append('KOR')

    elif sunburst_sub.iloc[i,1] == 'Taiwan*':

        iso_list.append('TWN')

    elif sunburst_sub.iloc[i,1] == 'Congo (Kinshasa)':

        iso_list.append('COD')

        sunburst_sub.iloc[i,1] = 'Congo'

#    elif sunburst_sub.iloc[i,1] == 'Congo (Brazzaville)':

#        iso_list.append('COD')

#        sunburst_sub.iloc[i,1] = 'Congo'

    elif sunburst_sub.iloc[i,1] == 'Cote d\'Ivoire':

        iso_list.append('CIV')

    elif sunburst_sub.iloc[i,1] == 'Gambia, The':

        iso_list.append('GMB')

    elif sunburst_sub.iloc[i,1] == 'Bahamas, The':

        iso_list.append('BHS')

    elif sunburst_sub.iloc[i,1] == 'West Bank and Gaza':

        iso_list.append('PS')

    elif sunburst_sub.iloc[i,1] == 'Burma':

        iso_list.append('MMR')

    else:

        iso_list.append(pc.country_name_to_country_alpha3(sunburst_sub.iloc[i,1], cn_name_format="default"))

sunburst_sub['ISO'] = iso_list # Save ISO column



# Change some country names for continent labels

sunburst_sub.loc[sunburst_sub['Country/Region'] == 'Congo (Kinshasa)','Country/Region'] = 'Congo'

sunburst_sub.loc[sunburst_sub['Country/Region'] == 'Cote d\'Ivoire','Country/Region'] = 'Ivory Coast'

sunburst_sub.loc[sunburst_sub['Country/Region'] == 'Korea, South','Country/Region'] = 'South Korea'

sunburst_sub.loc[sunburst_sub['Country/Region'] == 'Taiwan*','Country/Region'] = 'Taiwan'

sunburst_sub.loc[sunburst_sub['Country/Region'] == 'Burma','Country/Region'] = 'Myanmar'

sunburst_sub.drop(sunburst_sub[sunburst_sub['Country/Region'] == 0].index, inplace=True)



# Reference: https://stackoverflow.com/questions/55910004/get-continent-name-from-country-using-pycountry

continents = { # Add continent to data frame

    'NA': 'North America',

    'SA': 'South America', 

    'AS': 'Asia',

    'OC': 'Australia',

    'AF': 'Africa',

    'EU': 'Europe'

}



sunburst_sub['Continent'] = [continents[pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))] for country in sunburst_sub['Country/Region']]



# sunburst_sub.to_csv('sunburst.csv', index=False) # Save data



sunburst_edit = pd.read_csv('../input/data-viz-files/sunburst_edit.csv') # Load edited data from R



# Reference: https://stackoverflow.com/questions/14162723/replacing-pandas-or-numpy-nan-with-a-none-to-use-with-mysqldb

sunburst_edit = sunburst_edit.where(pd.notnull(sunburst_edit), None) # change NaN to None



provinces = list(sunburst_edit['Province.State']) # make dataframe version of dictionary

countries = list(sunburst_edit['Country.Region'])

continents = list(sunburst_edit['Continent'])

confirmed_cases = list(sunburst_edit['Confirmed'])

sunburst_dict = pd.DataFrame(

    dict(Provinces=provinces, Countries=countries, Continents=continents, Cases=confirmed_cases)

)



# change countries with extra None Leaves to match Parent node

non_none_leaves = sunburst_dict[sunburst_dict['Provinces'].notnull()]['Countries'].unique()

for i in non_none_leaves:

    # sum to see if any None

    # Reference: https://stackoverflow.com/questions/45271309/check-for-none-in-pandas-dataframe

    num_none = sum(sunburst_dict[sunburst_dict['Countries'] == i].applymap(lambda x: x is None)['Provinces'])

    # if there's a None, change it to country

    if num_none != 0:

        none_province = sunburst_dict[sunburst_dict['Countries'] == i]

        none_index = none_province['Provinces'][none_province['Provinces'].isnull()].index

        # References: https://stackoverflow.com/questions/13842088/set-value-for-particular-cell-in-pandas-dataframe-using-index

        sunburst_dict.at[none_index, 'Provinces'] = i + ' (country)'

        

# Reference: https://plot.ly/python/sunburst-charts/#sunburst-of-a-rectangular-dataframe-with-continuous-color-argument-in-pxsunburst

fig = px.sunburst(sunburst_dict,

                  path=['Continents', 'Countries', 'Provinces'],

                  values='Cases',

                  color='Continents',

                  hover_data=['Countries'],

                  title="Sunburst Hierarchy of Confirmed Cases",

                  height=1000,

                  width=1000,

                  color_continuous_scale=px.colors.sequential.Viridis)

fig.update(layout=dict(title=dict(x=0.5)))

fig.update_layout(legend_title='<b> Total Cases </b>')

fig.show()



cwd = os.getcwd()

# fig.write_html(cwd + '\\visualization6.html')