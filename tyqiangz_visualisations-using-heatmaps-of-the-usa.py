# install the necessary libraries

!pip install us

!pip install gmaps
# import the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt # for plotting graphs

import math



import us # for converting state names to numeric geolocation

import plotly.graph_objects as go # to plot heatmaps

import gmaps # to plot heatmaps using Google Maps



# libraries for folium heatmaps

import folium

from folium import plugins



# libraries for displaying image files in a notebook

from IPython.display import display

from PIL import Image
farmers_mkt = pd.read_csv('/kaggle/input/farmers-markets-in-the-united-states/farmers_markets_from_usda.csv')



print(farmers_mkt.shape)

print('-'*40)

farmers_mkt.isna().sum()
farmers_mkt.head()
farmers_mkt = farmers_mkt.drop(columns=['Season1Date',

       'Season1Time', 'Season2Date', 'Season2Time', 'Season3Date',

       'Season3Time', 'Season4Date', 'Season4Time'])
# if products are not shown, we assign a value 1, otherwise assign 0

farmers_mkt['No products'] = farmers_mkt['Bakedgoods'].isna().astype(int)



farmers_mkt.loc[:,['MarketName','No products']].head()
farmers_mkt['Is market'] = [1]*farmers_mkt.shape[0]
# function that formats the variables corresponding to each social media platform

# if a social media platform is present, we assign it a value 1, otherwise value 0 is assigned

def format_social_media(df):

    vars = ['Website', 'Facebook', 'Twitter', 'Youtube', 'OtherMedia']

    

    # if a social media site is present, we assign 1, otherwise 0

    for var in vars:

        df[var] = (df[var].notnull()).astype('int')

    

    return df



# apply the function to the data set

farmers_mkt = format_social_media(farmers_mkt)



# add a media score, a sum of the values of the 5 media types

farmers_mkt['Media'] = farmers_mkt.loc[:, 'Website':'OtherMedia'].sum(1)



# drop the columns of each media channel

farmers_mkt = farmers_mkt.drop(columns=['Website', 'Facebook', 'Twitter', 'Youtube',

       'OtherMedia', 'updateTime'])
# products that need feature engineering

products = ['Organic', 'Bakedgoods', 'Cheese', 'Crafts', 'Flowers', 'Eggs',

       'Seafood', 'Herbs', 'Vegetables', 'Honey', 'Jams', 'Maple', 'Meat',

       'Nursery', 'Nuts', 'Plants', 'Poultry', 'Prepared', 'Soap', 'Trees',

       'Wine', 'Coffee', 'Beans', 'Fruits', 'Grains', 'Juices', 'Mushrooms',

       'PetFood', 'Tofu', 'WildHarvested']



# changing all the 'Y' to 1 and 'N' to 0

for product in products:

    try:

        farmers_mkt[product] = farmers_mkt[product].replace(to_replace=['Y', 'N'], value=[1,0])

    except:

        continue
# temporary dataframe to store our new variables

products_df = pd.DataFrame()



# dictionary of new variables

new_products = {'Plants':['Trees', 'Plants', 'Nursery', 'Flowers'],\

                'Meat':['Meat', 'Poultry', 'Seafood'],\

                'Dairy':['Cheese', 'Eggs', 'Tofu'],\

                'Fresh produce':['Organic', 'Herbs', 'Vegetables', 'Mushrooms', 'WildHarvested', 'Beans', 'Fruits', 'Grains', 'Nuts'],\

                'Confectionery':['Bakedgoods', 'Honey', 'Jams', 'Maple', 'Coffee', 'Juices', 'Wine'],\

                'Others':['Crafts', 'Prepared', 'Soap', 'PetFood']}



# creating new product categories

for product in new_products.keys():

    try:

        products_df[product] = farmers_mkt.loc[:,new_products[product]].sum(1)

    except:

        print(product)



# simply sum the columns up to obtain number of categories present        

products_df['Number of products'] = products_df.sum(1)



# drop the product categories from the main dataset and add new product categories

farmers_mkt = farmers_mkt.drop(columns=products)

farmers_mkt = pd.concat([farmers_mkt, products_df], axis=1)
payment_modes = ['Credit', 'WIC', 'WICcash', 'SFMNP', 'SNAP']

for payment_mode in payment_modes:

    try:

        farmers_mkt[payment_mode] = farmers_mkt[payment_mode].replace(to_replace=['Y', 'N'], value=[1,0])

    except:

        continue

farmers_mkt['Low income friendly'] = farmers_mkt.loc[:, 'WIC':'SNAP'].sum(1)
farmers_mkt.head()
county_info = pd.read_csv('/kaggle/input/farmers-markets-in-the-united-states/wiki_county_info.csv')



print(county_info.shape)

print('-'*20)

county_info.isna().sum()
county_info.head()
# the variable 'number' is redundant, so we drop them

county_info = county_info.drop(columns = 'number')



# drop rows with missing values

county_info = county_info.dropna()
# the figures are strings, have a dollar sign and commas, the following changes them to integers

def remove_char(df):

    bad_var = ['per capita income', 'median household income', 'median family income', 'population', 'number of households']

    bad_tokens = ['$', r',']

    

    for var in bad_var:

        df[var] = df[var].replace('[\$,]', '', regex=True).astype(int)

    return df
county_info = remove_char(county_info)
# list of all the unique states in the data set

states = list(county_info['State'].unique())



# check which U.S. state isn't named correctly

for state in states:

    if (us.states.lookup(state) == None):

        print(state)
county_info.loc[county_info['State'] == 'U.S. Virgin Islands', 'State'] = 'Virgin Islands'
# list of all the unique states in the data set

states = list(county_info['State'].unique())



states_coded = []



# obtains the FIPS code from state name

for state in states:

    states_coded.append(us.states.lookup(state).abbr)
# ready to create state-level data set

state_info = pd.DataFrame()



# retain state names in state-level data set for reference

state_info['State'] = states



# FIPS code of each state

state_info['State code'] = states_coded



# variables to be included in new data set

cols = ['Per capita income', 'Population', 'Number of households', 'Number of markets']



# initialisation

for var in cols:

    state_info[var] = ''



temp = []



# computation for state-level variables

for i in range(len(states)):

    num_household = 0

    

    # dataframe of all counties in state state[i]

    state_df = pd.DataFrame(county_info.loc[county_info['State'] == states[i], :]).reset_index()

    

    total_popn = sum(state_df['population'])

    state_info.loc[i, 'Population'] = total_popn

    state_info.loc[i, 'Number of households'] = state_df['number of households'].sum()

    state_info.loc[i, 'Number of markets'] = farmers_mkt[farmers_mkt['State'] == states[i]].shape[0]

    temp += [round(state_df['per capita income'].dot(state_df['population'] / total_popn))]



state_info['Per capita income'] = temp

state_info['Per capita income'] = state_info['Per capita income'].astype(int)



state_info.head()
# many of the steps required to plot a gmaps heatmap are repetitive

# the following function eases this process



# input: a dataframe containing longitude and latitude values and the variable you wish to visualise

# output: a corresponding heatmap

def plot_gmaps(df, var):

    # obtain your own API key with the link above

    API_KEY = YOUR_API_KEY



    gmaps.configure(api_key=API_KEY)

    

    # a dataframe of longitude and latitudes, this dataframe cannot have missing values

    valid_df = df.loc[~df['x'].isnull(), ['x', 'y', var]]

    

    m = gmaps.Map()

    

    # adding a heatmap layer on top on Google Maps

    heatmap_layer = gmaps.heatmap_layer(

        valid_df[['y','x']], 

        

        # we divide the variable by its max value to ensure all variable have a scale of [0,1]

        # this prevents the heatmap from looking more saturated for a variables with larger scale

        weights=valid_df[var] / valid_df[var].max(),

        max_intensity=100, 

        point_radius=20.0

    )

    m.add_layer(heatmap_layer)

    

    return m
# uncomment out the following line if you're using a jupyter notebook

# plot_gmaps(farmers_mkt, 'Is market')



# comment out the following line if you're using a jupyter notebook

display(Image.open("/kaggle/input/figures/gmaps_is_market.png"))
# uncomment out the following line if you're using a jupyter notebook

# plot_gmaps(farmers_mkt, 'Low income friendly')



# comment out the following line if you're using a jupyter notebook

display(Image.open("/kaggle/input/figures/gmaps_low_income_friendly.png"))
# uncomment out the following line if you're using a jupyter notebook

# plot_gmaps(farmers_mkt, 'No products')



# comment out the following line if you're using a jupyter notebook

display(Image.open("/kaggle/input/figures/gmaps_no_products.png"))
# uncomment out the following line if you're using a jupyter notebook

# plot_gmaps(farmers_mkt, 'Media')



# comment out the following line if you're using a jupyter notebook

display(Image.open("/kaggle/input/figures/gmaps_media.png"))
## Distribution of markets that sell plants



# uncomment out the following line if you're using a jupyter notebook

# plot_gmaps(farmers_mkt, 'Plants')



# comment out the following line if you're using a jupyter notebook

display(Image.open("/kaggle/input/figures/gmaps_plants.png"))
## Distribution of markets that sell meat



# uncomment out the following line if you're using a jupyter notebook

# plot_gmaps(farmers_mkt, 'Meat')



# comment out the following line if you're using a jupyter notebook

display(Image.open("/kaggle/input/figures/gmaps_meat.png"))
## Distribution of markets that sell fresh produce



# uncomment out the following line if you're using a jupyter notebook

# plot_gmaps(farmers_mkt, 'Fresh produce')



# comment out the following line if you're using a jupyter notebook

display(Image.open("/kaggle/input/figures/gmaps_fresh_produce.png"))
## Distribution of markets that sell dairy products



# uncomment out the following line if you're using a jupyter notebook

# plot_gmaps(farmers_mkt, 'Dairy')



# comment out the following line if you're using a jupyter notebook

display(Image.open("/kaggle/input/figures/gmaps_dairy.png"))
## Distribution of markets that sell confectionery products



# uncomment out the following line if you're using a jupyter notebook

# plot_gmaps(farmers_mkt, 'Confectionery')



# comment out the following line if you're using a jupyter notebook

display(Image.open("/kaggle/input/figures/gmaps_confectionery.png"))
## Distribution of markets that sell other miscellaneous products



# uncomment out the following line if you're using a jupyter notebook

# plot_gmaps(farmers_mkt, 'Others')



# comment out the following line if you're using a jupyter notebook

display(Image.open("/kaggle/input/figures/gmaps_others.png"))
## Distribution of number of products sold in the markets



# uncomment out the following line if you're using a jupyter notebook

# plot_gmaps(farmers_mkt, 'Number of products')



# comment out the following line if you're using a jupyter notebook

display(Image.open("/kaggle/input/figures/gmaps_num_products.png"))
## Distribution of markets that use social media



# uncomment out the following line if you're using a jupyter notebook

# plot_gmaps(farmers_mkt, 'Media')



# comment out the following line if you're using a jupyter notebook

display(Image.open("/kaggle/input/figures/gmaps_media.png"))
state_info['Number of markets per capita'] = state_info['Number of markets'] / state_info['Population']



# many of the steps for plotting heatmaps are repetitive, this function takes care of that

# input: a dataframe df and a variable in df that you wish to plot

# output: None. A heatmap will be plotted directly.

def plot_heatmap(df, var): 

    # plotting the heatmap by states

    fig = go.Figure(data=go.Choropleth(

        locations=df['State code'], # Spatial coordinates

        z = df[var].astype(float), # Data to be color-coded

        locationmode = 'USA-states', # set of locations match entries in `locations`

        colorscale = 'Reds',

        colorbar_title = var,

        text = df['State']

    ))



    fig.update_layout(

        title_text = var + ' by state<br>(Hover over the states for details)',

        geo_scope='usa', # limit map scope to USA

    )



    fig.show()
plot_heatmap(state_info, 'Population')
plot_heatmap(state_info, 'Per capita income')
# sort the dataframe by the 'per capita income' variable, ascending order

state_info = state_info.sort_values(by=['Per capita income'])



fig = plt.figure()

ax = fig.add_axes([0,0,2,1])

ax.bar(state_info['State'], state_info['Per capita income'])

plt.xticks(rotation=90)

plt.show()
state_info['Per capita income'].describe()
plot_heatmap(state_info, 'Number of households')
plot_heatmap(state_info, 'Number of markets')