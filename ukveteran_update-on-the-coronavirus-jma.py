import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
confirmed = pd.read_csv('../input/coronavirus-latlon-dataset/time_series_19-covid-Confirmed.csv')

deaths = pd.read_csv('../input/coronavirus-latlon-dataset/time_series_19-covid-Deaths.csv')

recovered = pd.read_csv('../input/coronavirus-latlon-dataset/time_series_19-covid-Recovered.csv')
confirmed.head()
def melt_data(df):

    '''Returns a new DataFrame with three columns:

                date, county, and count

    Count refers to the number of confirmed cases

    on a specific date within a specific county.'''

    new_df = df.T #turn date columns into rows

    new_df.columns = pd.Index([i for i in new_df.iloc[0,].values], dtype='str') #rename columns by county

    new_df = new_df.iloc[4:, :] #only select dates

    new_df = new_df.rename_axis(index='date')

    new_df = new_df.reset_index()

    new_df['date'] = pd.to_datetime(new_df['date'])

    new_df = new_df.melt(id_vars='date').rename(columns={'variable':'county','value':'count'})

    return new_df
united_states = confirmed[confirmed['Country/Region']=='US']

for county in united_states[united_states['Province/State'].str.contains("CA")]['Province/State'].value_counts().index:

    print(county)
def select_bay_area_counties(df):

    '''Selects only counties that are considered

    to be in the Bay Area. This will be updated as

    new counties get news of confirmed cases.'''

    new_df = (df[(df['Province/State']=="Alameda County, CA") | 

                 (df['Province/State']=="Santa Clara County, CA") | 

                 (df['Province/State']=="San Francisco County, CA") | 

                 (df['Province/State']=="San Mateo, CA") | 

                 (df['Province/State']=="Contra Costa County, CA") | 

                 (df['Province/State']=="Grand Princess Cruise Ship") |

                (df['Province/State']=="Sonoma County, CA") |

                (df['Province/State']=="Solano, CA")])

    return new_df
bay_area_confirmed = melt_data(select_bay_area_counties(confirmed))

bay_area_deaths = melt_data(select_bay_area_counties(deaths))

bay_area_recovered = melt_data(select_bay_area_counties(recovered))
bay_area_confirmed.head()
plt.figure(figsize=(15,8))

bay_area_confirmed.groupby('date').sum().loc[:, 'count'].plot(kind="bar")

plt.title("Number of Confirmed Cases in the Bay Area", fontsize=20)

plt.xlabel("Date", fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.show()
grouped = bay_area_confirmed.groupby(['date','county']).sum().reset_index()

grouped = grouped[grouped['date']==grouped.date.values[-1]]

grouped.groupby('county').sum().reset_index()
plt.figure(figsize=(18,6))

sns.barplot(x='county', y='count', data = grouped.groupby('county').sum().reset_index())

plt.title("Number of Confirmed Cases by County", fontsize=20)

plt.xlabel("County", fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.show()