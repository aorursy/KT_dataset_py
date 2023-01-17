import json
import os
from datetime import date
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, rgb2hex
from mpl_toolkits.basemap import Basemap

%matplotlib inline
with open('../input/sf-restaurant-scores-lives-standard/socrata_metadata.json', 'r') as f:
    metadata = json.loads(f.read())
    
data = pd.read_csv('../input/sf-restaurant-scores-lives-standard/restaurant-scores-lives-standard.csv')
print('Starting with {} records.'.format(len(data)))

# Select only records where the inspection score is not null
data_clean = data[~pd.isna(data.inspection_score)].copy()
print('After cleaning rows with no score: {}'.format(len(data_clean)))

# Zip codes are 'dirty'. A quick google search returns the 'allowed' zip codes
allowed_zips = {'94151', '94159', '94158', '94102', '94104', '94103', '94105', '94108',
                '94177', '94107', '94110', '94109', '94112', '94111', '94115', '94114',
                '94117', '94116', '94118', '94121', '94123', '94122', '94124', '94127',
                '94126', '94129', '94131', '94133', '94132', '94134', '94139', '94143'}

data_clean = data_clean[data_clean.business_postal_code.isin(allowed_zips)].copy()
print('After cleaning weird zip codes: {}'.format(len(data_clean)))

# Convert zip codes to numeric and dates to datetimes
# data_clean.loc[:, 'business_postal_code'] = pd.to_numeric(data_clean.business_postal_code)
data_clean.loc[:, 'inspection_date'] = pd.to_datetime(data_clean.inspection_date)

# Select only the columns that are important
data_clean = data_clean[['business_id', 'business_postal_code', 'inspection_date', 'inspection_score']].copy()
# Select only the latest inspection per business
data_clean = data_clean.sort_values('inspection_date').groupby('business_id').last()
print('After selecting the last inspection only: {}'.format(len(data_clean)))
# Get the average score per zipcode
avg_scores = data_clean.groupby('business_postal_code').mean()
avg_scores.head()
plt.figure(figsize=(16, 12))

# SF coordinates.
lowerlon = -122.52
upperlon = -122.34
lowerlat = 37.70
upperlat = 37.84

m = Basemap(
    llcrnrlon=lowerlon,
    llcrnrlat=lowerlat,
    urcrnrlon=upperlon,
    urcrnrlat=upperlat,
    resolution='c',
    projection='lcc',
    lat_0=lowerlat,
    lat_1=upperlat,
    lon_0=lowerlon,
    lon_1=upperlon
    )

shp_info = m.readshapefile('../input/2017-zip-code-shapefiles/cb_2017_us_zcta510_500k', 
                           'zips', drawbounds=True)

scores_dict = avg_scores.to_dict()['inspection_score']
colormap = plt.cm.RdYlGn
min_score = avg_scores.inspection_score.min()
max_score = avg_scores.inspection_score.max()
edge_color = '#000000'

for coords, info in zip(m.zips, m.zips_info):
    zip_code = info['ZCTA5CE10']
    if zip_code in scores_dict:
        i, j = zip(*coords)
        score_norm = (scores_dict.get(zip_code, min_score) - min_score) / (max_score - min_score)
        zip_color = rgb2hex(colormap(score_norm)[:3])
        plt.fill(i, j, color=zip_color, edgecolor='k')

# Colorbar
mm = plt.cm.ScalarMappable(cmap=colormap)
mm.set_array([min_score, max_score])
plt.colorbar(mm, ticks=np.arange(int(min_score), int(max_score+1), 1), orientation="vertical")
plt.title('Average Restaurant Score per Zip Code -- San Francisco')
plt.gca().axis('off')
plt.show()
