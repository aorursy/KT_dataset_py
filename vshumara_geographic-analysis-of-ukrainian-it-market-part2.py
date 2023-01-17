# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter



import folium

from folium import plugins

from folium.plugins import HeatMap



from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
#Importing data of Private Entrepreneurs and geodata of postal codes

df_codes = pd.read_csv("../input/ukrain-postal-codes/ua_post_utf8_v2.csv").drop_duplicates(subset="postal_code")

df_fop = pd.read_csv("../input/ukraine-register-of-individual-entrepreneurs/edr_fop_utf-8.csv")



#Extracting only whose which are registered under IT-related Industry classification code

df_fop = df_fop.loc[(df_fop['KVED'].str.startswith('63')) | (df_fop['KVED'].str.startswith('62'))]



#Extracting postal code from the full address 

df_fop['postal_code'] = df_fop['ADDRESS'].str[:5]

df_codes['postal_code'] = df_codes['postal_code'].astype(str).apply(lambda x: '0' * (5 - len(x)) + x if len(x) < 5 else x)
#Geocoding by postal codes

df_full = pd.merge(df_fop.assign(postal_code=df_fop.postal_code.astype(str)), 

          df_codes.assign(postal_code=df_codes.postal_code.astype(str)), 

          how='left', on='postal_code')



print('NAN count = ' + str(df_full['longitude'].isna().sum()))
#Deal with those which are not matched

df_nan = df_full[pd.isnull(df_full['latitude'])][['postal_code', 'ADDRESS']]

df_nan['postal4'] = df_nan.postal_code.str[:4]

df_codes['postal4'] = df_codes.postal_code.str[:4]

df_codes = df_codes.drop_duplicates(subset="postal4")



df_nan_merge = pd.merge(df_nan, df_codes, how='left', on='postal4')



df_full.loc[pd.isnull(df_full['latitude']),['latitude', 'longitude', 'admin_name1']] = df_nan_merge[['latitude', 'longitude', 'admin_name1']].values



print(df_full['latitude'].isna().sum())
#Distinguishing the active records and the Entrepreneurs who stopped the business

df_full = df_full.dropna(subset = ['longitude'])



df_act = df_full[df_full['STAN'] == 'зареєстровано']

df_inact = df_full[df_full['STAN'] == 'припинено']



print('NA count after cleanup = ' + str(df_full['latitude'].isna().sum()))

print('Total count after cleanup = ' + str(len(df_full.index)))
#preparing an array with distributions by 'latitude', 'longitude'

df_cln = df_act[['latitude', 'longitude']].copy()

df_cln['count'] = 0;

res = df_cln.groupby(['latitude', 'longitude'])['count'].size().reset_index()

res['count'].sum()



#res['count'] = (res['count'] / res['count'].max())
# import gmaps



# gmaps.configure(api_key="")

# locations = res[['latitude', 'longitude']]

# weights = res['count']

# figure_layout = {'height': '1024px', 'margin': '0 auto 0 auto'}

# fig = gmaps.figure(map_type='HYBRID', layout=figure_layout, zoom_level=6.6, center=(48.7,31.0))

# heatmap_layer = gmaps.heatmap_layer(locations, weights=weights, max_intensity = 100, point_radius = 16)



# fig.add_layer(heatmap_layer)

# fig
#let's try to do the same with Open Street Maps



#removing low frequency data

res = res[res['count'] > 10]



hmap = folium.Map(location=[48.7, 31.0], #height=1024,

                    zoom_start = 6.4) # Uses lat then lon. The bigger the zoom number, the closer in you get



# Plot it on the map

HeatMap(res.values, radius = 10, min_opacity = 0.4, max_zoom=1, blur=8, max_val=150.).add_to(hmap)



# Display the map

hmap
#creating wordcloud to highlite most frequent cities

df_cln = df_act[['place_name']].copy()

df_cln['count'] = 0;

res = df_cln.groupby('place_name')['count'].size().reset_index()

dic = res.set_index('place_name').T.to_dict('records')



plt.figure(figsize=(18, 16), dpi= 80)



wordcloud = WordCloud(width=1024,height=768, max_words=500, background_color="white").generate_from_frequencies(dic[0])



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
