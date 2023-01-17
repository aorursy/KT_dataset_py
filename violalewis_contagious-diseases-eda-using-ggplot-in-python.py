# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = "15, 15"



import seaborn as sns

from plotnine import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Set theme for plots

theme_set(

        theme_minimal(base_size = 12, base_family = 'sans') \

      + theme(axis_text_x = element_text(angle = 0, vjust = 0.5, hjust = 0.5),\

          axis_text = element_text(size = 8), \

          axis_title = element_text(size = 14), \

          panel_grid_major = element_line(color = "grey"), \

          panel_grid_minor = element_blank(), \

          panel_background = element_rect(fill = "aliceblue"), \

          strip_background = element_rect(fill = "navy", color = "navy", size = 1), \

          strip_text = element_text(face = "bold", size = 10, color = "white"), \

          legend_position = "right", \

          legend_background = element_blank(), \

          panel_border = element_rect(color = "grey" , fill = "NA", size = 0.5))  

)
# Read input data and concat into a single dataframe

polio = pd.read_csv("../input/polio.csv")

mumps = pd.read_csv("../input/mumps.csv")

measles = pd.read_csv("../input/measles.csv")

pertussis = pd.read_csv("../input/pertussis.csv")

rubella = pd.read_csv("../input/rubella.csv")

hepatitis = pd.read_csv("../input/hepatitis.csv")

smallpox = pd.read_csv("../input/smallpox.csv")



polio.disease = 'Polio'

mumps.disease = 'Mumps'

measles.disease = 'Measles'

pertussis.disease = 'Pertussis'

rubella.disease = 'Rubella'

hepatitis.disease = 'Hepatitis'

smallpox.disease = 'Smallpox'



diseasedf = pd.concat([polio, mumps, measles,pertussis, rubella, hepatitis, smallpox], ignore_index=True)



diseasedf.rename(columns={'week':'date'}, inplace=True)

diseasedf.date = diseasedf.date.astype(str)

diseasedf['year'] = diseasedf['date'].str.extract('(\d\d\d\d)', expand=True)

diseasedf['week'] = diseasedf['date'].str.extract('(\\d[0-9])$', expand=True)



diseasedf['cases'] = pd.to_numeric(diseasedf['cases'], errors='coerce')



diseasedf['year'] = pd.to_numeric(diseasedf['year'])



diseasedf['week'] = pd.to_numeric(diseasedf['week'])



diseasedf.info()



diseasedf.head()
# Remove NA rows and rows with number of cases = 0

((diseasedf.isnull() | diseasedf.isna()).sum() * 100 / diseasedf.index.size).round(2)



diseasedf.dropna(how='any', inplace =True)



diseasedf = diseasedf.loc[diseasedf['cases'] > 0]



((diseasedf.isnull() | diseasedf.isna()).sum() * 100 / diseasedf.index.size).round(2)
# Create a dataframe for US population data

us_pop = {'year': [2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000, 1999, 1998

                   , 1997, 1996, 1995, 1994, 1993, 1992, 1991, 1990, 1989, 1988, 1987, 1986, 1985, 1984

                   , 1983, 1982, 1981, 1990, 1979, 1978, 1977, 1976, 1975, 1974, 1973, 1972, 1971, 1970

                   , 1969, 1968, 1967, 1966, 1965, 1964, 1963, 1962, 1961, 1960, 1959, 1958, 1957, 1956

                   , 1955, 1954, 1953, 1952, 1951, 1950, 1949, 1948, 1947, 1946, 1945, 1944, 1943, 1942

                   , 1941, 1940, 1939, 1938, 1937, 1936, 1935, 1934, 1933, 1932, 1931, 1930, 1929, 1928]

          , 'population':[310500000, 308110000, 306770000, 304090000, 301230000, 298380000, 295520000, 292810000

                          , 290110000, 287630000, 284970000, 282160000, 279040000, 275850000, 272650000, 269390000

                          , 266280000, 263130000, 259920000, 256510000, 252980000, 249620000, 246820000, 244500000

                          , 242290000, 240130000, 237920000, 235820000, 233790000, 231660000, 229470000, 227220000

                          , 225060000, 222580000, 220240000, 218040000, 215970000, 213850000, 211910000, 209900000

                          , 207660000, 205050000, 202680000, 200710000, 198710000, 196560000, 194300000, 191890000

                          , 189240000, 186540000, 183690000, 180670000, 177830000, 174880000, 171990000, 168900000

                          , 165930000, 163030000, 160180000, 157550000, 154880000, 152270000, 149190000, 146630000

                          , 144130000, 141390000, 139930000, 138400000, 136740000, 134860000, 133400000, 132120000

                          , 130880000, 129820000, 128820000, 128050000, 127250000, 126370000, 125580000, 124840000

                          , 124040000, 123080000, 121770000, 120510000]}

populationdf = pd.DataFrame(us_pop)

populationdf.head()
# Disease outbreak through the years - All Diseases

import warnings

warnings.filterwarnings('ignore')

(

    ggplot(data=diseasedf) \

      + aes(y='cases', x='year') \

      + aes(color='disease') \

      + geom_line(aes(color='disease'),size=1) \

      + labs(title = "Disease Outbreak Through The Years") \

      + labs(x = "Year") \

      + labs(y = "Number of Cases") 

)

# Disease outbreak through the years - Facet Grid

import warnings

warnings.filterwarnings('ignore')

(

    ggplot(data=diseasedf)

      + aes(y='cases', x='year')

      + aes(color='disease')

      + geom_line(aes(color='disease'),size=1) 

      + facet_wrap('~disease', nrow=3, ncol=3)

      + labs(title = "Disease Outbreak Through The Years")

      + labs(x = "Year") 

      + labs(y = "Number of Cases")

)

# Create new column 'decade': 1920 = all years from 1920 to 1929

bins = np.array([1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])

diseasedf['decade'] = pd.cut(diseasedf.year, bins

                             , labels=['1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010']

                            , right=False)



# Get the mean cases 

diseasedf['mean_cases_year'] = diseasedf.groupby(['year', 'disease', 'week'])['cases'].transform('mean')
# Disease density (by decade)

log10cases = np.log10(diseasedf.mean_cases_year)



import warnings

warnings.filterwarnings('ignore')

(

    ggplot(data=diseasedf)

      + aes(x=log10cases, fill='disease')

      + aes(color='disease')

      + geom_density(alpha = 0.3, size = 1)

      + geom_rug()

      + facet_wrap('~decade', nrow=4, ncol=3, scales="free_y")

      + theme(axis_text_x = element_text(angle = 0, vjust = 0.5, hjust = 0.1),\

          axis_text_y = element_text(angle = 0, vjust = 0.5, hjust = 0.1),\

          axis_text = element_text(size = 6), \

          subplots_adjust={'wspace': 5.8, 'hspace': 0.2})

      + labs(title = "Disease Density Through the Decades")

      + labs(x = "Log transformation of the mean number of cases") 

      + labs(y = "Density")

)
# Join the disease and population data 



popaffdf = pd.merge(diseasedf, populationdf, left_on = 'year', right_on = 'year', how = 'left')



popaffdf['per_pop_aff'] = (popaffdf['mean_cases_year'] / popaffdf['population']) * 100

# Stacked Bar Plot - Population affected by contagious diseases over the years

import warnings

warnings.filterwarnings('ignore')

(

    ggplot(data=popaffdf) \

    + aes(x='year', y='per_pop_aff', color='disease') \

    + geom_line(size=1) \

    + labs(title = "Disease Rate vs Entire US Population") \

    + labs(x = "Years") \

    + labs(y = "Percentage of Popuation Affected") 

)
# Calculate mean cases per disease, state, decade

diseasedf['state_name'] = diseasedf.state_name.str.title()



diseasedf['mean_cases_state_decade'] = 0



diseasedf['mean_cases_state_decade'] = diseasedf.groupby(['disease', 'state_name', 'decade'])['cases'].transform('mean')
# Heatmap - Measles over Decades

import warnings

warnings.filterwarnings('ignore')

(

    ggplot(data=diseasedf.loc[diseasedf.disease == 'Measles']) \

    + aes(x='decade', y='state_name') \

    + geom_tile(aes(fill='mean_cases_state_decade')) \

    + scale_fill_gradientn(colors=['#9ebcda','#8c6bb1','#88419d','#6e016b']) \

    + theme(axis_text = element_text(size = 6)) \

    + labs(title = "Measles Over Decades Heatmap") \

    + labs(x = "Decades") \

    + labs(y = "State") 

)
# Heatmap - Hepatitis over Decades

import warnings

warnings.filterwarnings('ignore')

(

    ggplot(data=diseasedf.loc[diseasedf.disease == 'Hepatitis']) \

    + aes(x='decade', y='state_name') \

    + geom_tile(aes(fill='mean_cases_state_decade')) \

    + scale_fill_gradientn(colors=['#9ebcda','#8c6bb1','#88419d','#6e016b']) \

    + theme(axis_text = element_text(size = 6)) \

    + labs(title = "Hepatitis Over Decades Heatmap") \

    + labs(x = "Decades") \

    + labs(y = "State") 

)
# Heatmap - Polio over Decades

import warnings

warnings.filterwarnings('ignore')

(

    ggplot(data=diseasedf.loc[diseasedf.disease == 'Polio']) \

    + aes(x='decade', y='state_name') \

    + geom_tile(aes(fill='mean_cases_state_decade')) \

    + scale_fill_gradientn(colors=['#9ebcda','#8c6bb1','#88419d','#6e016b']) \

    + theme(axis_text = element_text(size = 6)) \

    + labs(title = "Polio Over Decades Heatmap") \

    + labs(x = "Decades") \

    + labs(y = "State") 

)
# Heatmap - Pertussis over Decades

import warnings

warnings.filterwarnings('ignore')

(

    ggplot(data=diseasedf.loc[diseasedf.disease == 'Pertussis']) \

    + aes(x='decade', y='state_name') \

    + geom_tile(aes(fill='mean_cases_state_decade')) \

    + scale_fill_gradientn(colors=['#9ebcda','#8c6bb1','#88419d','#6e016b']) \

    + theme(axis_text = element_text(size = 6)) \

    + labs(title = "Pertussis Over Decades Heatmap") \

    + labs(x = "Decades") \

    + labs(y = "State") 

)
# Heatmap - Rubella over Decades

import warnings

warnings.filterwarnings('ignore')

(

    ggplot(data=diseasedf.loc[diseasedf.disease == 'Rubella']) \

    + aes(x='decade', y='state_name') \

    + geom_tile(aes(fill='mean_cases_state_decade')) \

    + scale_fill_gradientn(colors=['#9ebcda','#8c6bb1','#88419d','#6e016b']) \

    + theme(axis_text = element_text(size = 6)) \

    + labs(title = "Rubella Over Decades Heatmap") \

    + labs(x = "Decades") \

    + labs(y = "State") 

)
# Heatmap - Smallpox over Decades

import warnings

warnings.filterwarnings('ignore')

(

    ggplot(data=diseasedf.loc[diseasedf.disease == 'Smallpox']) \

    + aes(x='decade', y='state_name') \

    + geom_tile(aes(fill='mean_cases_state_decade')) \

    + scale_fill_gradientn(colors=['#9ebcda','#8c6bb1','#88419d','#6e016b']) \

    + theme(axis_text = element_text(size = 6)) \

    + labs(title = "Smallpox Over Decades Heatmap") \

    + labs(x = "Decades") \

    + labs(y = "State") 

)
# Heatmap - Mumps over Decades

import warnings

warnings.filterwarnings('ignore')

(

    ggplot(data=diseasedf.loc[diseasedf.disease == 'Mumps']) \

    + aes(x='decade', y='state_name') \

    + geom_tile(aes(fill='mean_cases_state_decade')) \

    + scale_fill_gradientn(colors=['#9ebcda','#8c6bb1','#88419d','#6e016b']) \

    + theme(axis_text = element_text(size = 6)) \

    + labs(title = "Mumps Over Decades Heatmap") \

    + labs(x = "Decades") \

    + labs(y = "State") 

)