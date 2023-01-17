import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns
def multiliner(string_list, n):

    length = len(string_list)

    for i in range(length):

        rem = i % n

        string_list[i] = '\n' * rem + string_list[i]

    return string_list
df = pd.read_csv('../input/boston-crime-incident-reports-20152019/crime.csv')
df.shape
df.head(20)
#df.set_index('incident_number', inplace = True)

df['shooting'].fillna(0, inplace = True)

df.drop(columns = ['location'], inplace = True)
df.head(5)
df.dtypes
df.isna().sum()
df_od = df.groupby('offense_description').size().reset_index(name = 'counts').set_index('offense_description').sort_values(by = 'counts', ascending = False)
df_od
df_ucr = df.groupby('ucr_part').size().reset_index(name = 'counts').set_index('ucr_part')
df_ucr
df_p3 = df.loc[df['ucr_part'] == 'Part Three']
df_3d = df_p3.groupby('offense_description').size().reset_index(name = 'counts').set_index('offense_description')
df_3d
df_other = df.loc[df['ucr_part'] == 'Other']
df_otherd = df_other.groupby('offense_description').size().reset_index(name = 'counts').set_index('offense_description')
df_otherd
df_ucrna = df.loc[df['ucr_part'].isnull()]
df_nad = df_ucrna.groupby('offense_description').size().reset_index(name = 'counts').set_index('offense_description')
df_nad
df_clean = df.loc[(df['ucr_part'] != 'Part Three') & (df['offense_description'] != 'INVESTIGATE PERSON')]
df_clean.shape
df_day = df_clean.groupby('day_of_week').size().reset_index(name = 'counts').set_index('day_of_week')
df_day
df_day.reset_index(inplace = True)
df_day['day_of_week'] = pd.Categorical(df_day['day_of_week'], categories = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'], ordered = True)
df_day.set_index('day_of_week', inplace = True)
df_day.sort_values(by = 'day_of_week', inplace = True)
fig = plt.figure(figsize = (10,7))

ax = plt.subplot(111)

ind = np.arange(7)

crimes_by_day = df_day['counts']

rects = ax.bar(ind, crimes_by_day, width = 0.8, color = ['green','olive','olive','olive','olive','red','green'])

ax.set_xticks(ind)

ax.set_xticklabels(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])

ax.set_title('Crimes in Boston by days of the week')

ax.set_ylabel('Amount of crimes')

for rect in rects:

    height = rect.get_height()

    ax.text(rect.get_x() + 0.15, 1.01 * height, height, fontsize = 12)
df_year = df_clean.groupby('year').size().reset_index(name = 'counts').set_index('year')
df_year
df_year.drop(labels = [2015,2019], inplace = True)
fig2 = plt.figure(figsize = (5,7))

ind2 = np.arange(3)

ax2 = plt.subplot(111)

rects = ax2.bar(ind2, df_year['counts'], width = 0.4, color = ['red','green','blue'])

ax2.set_xticks(ind2)

ax2.set_xticklabels([2016,2017,2018])

ax2.set_xlabel('Year')

ax2.set_ylabel('Amount of crimes')

ax2.set_title('Crimes in Boston by year')

for rect in rects:

    height = rect.get_height()

    ax2.text(rect.get_x() - 0.01, 1.01 * height, height, fontsize = 14)
df_1618 = df_clean.loc[(df_clean['year'] > 2015) & (df_clean['year'] < 2019)]
df_1618.shape
df_month = df_1618.groupby('month').size().reset_index(name = 'counts').set_index('month')
df_month
fig3 = plt.figure(figsize = (10,7))

ind3 = np.arange(12)

ax3 = plt.subplot(111)

rects = ax3.bar(ind3, df_month['counts'], width = 0.8,color = ['yellow','lime','yellow','yellow','red','red','red','darkred','orange','orange','green','green'])

ax3.set_xticks(ind3)

ax3.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

ax3.set_xlabel('Month')

ax3.set_ylabel('Amount of crimes')

ax3.set_title('Crimes in Boston by month')

for rect in rects:

    height = rect.get_height()

    ax3.text(rect.get_x() - 0.13, 1.01 * height, height, fontsize = 14)
df_ocg = df_clean.groupby('offense_code_group').size().reset_index(name = 'counts').set_index('offense_code_group').sort_values(by = 'counts', ascending = False)
df_ocg
fig41 = plt.figure(figsize = (20,8))

ind41 = np.arange(10)

ax41 = plt.subplot(111)

y_data = df_ocg['counts'].head(10)

df_riocg = df_ocg.reset_index()

rects = ax41.bar(ind41, y_data, width = 0.8,color = 'r')

ax41.set_xticks(ind41)

ax41.set_xticklabels(multiliner(df_ocg.index.tolist()[:10], 2))

ax41.set_xlabel('Offense Code Group')

ax41.set_ylabel('Amount of crimes')

ax41.set_title('Crimes in Boston by offense code group')

for rect in rects:

    height = rect.get_height()

    ax41.text(rect.get_x() + 0.2, 1.02 * height, height, fontsize = 14)
df_od = df_clean.groupby('offense_description').size().reset_index(name = 'counts').set_index('offense_description').sort_values(by = 'counts', ascending = False)
df_od
fig42 = plt.figure(figsize = (20,8))

ind42 = np.arange(10)

ax42 = plt.subplot(111)

y_data = df_od['counts'].head(10)

df_riod = df_od.reset_index()

rects = ax42.bar(ind42, y_data, width = 0.8,color = 'r')

ax42.set_xticks(ind42)

ax42.set_xticklabels(multiliner(df_od.index.tolist()[:10], 3))

ax42.set_xlabel('Offense Description')

ax42.set_ylabel('Amount of crimes')

ax42.set_title('Crimes in Boston by offense description')

for rect in rects:

    height = rect.get_height()

    ax42.text(rect.get_x() + 0.2, 1.02 * height, height, fontsize = 14)
df_districts = df_clean.groupby('district').size().reset_index(name = 'counts').set_index('district').sort_values('counts', ascending = False)
df_districts
fig5 = plt.figure(figsize = (10,7))

ind5 = np.arange(12)

ax5 = plt.subplot(111)

rects = ax5.bar(ind5, df_districts['counts'], width = 0.8,color = 'r')

ax5.set_xticks(ind5)

ax5.set_xticklabels(df_districts.index)

ax5.set_xlabel('District')

ax5.set_ylabel('Amount of crimes')

ax5.set_title('Crimes in Boston by district')

for rect in rects:

    height = rect.get_height()

    if height > 9999:

        hor = rect.get_x() - 0.13

    else:

        hor = rect.get_x() - 0.03

    ax5.text(hor, 1.01 * height, height, fontsize = 14)
df_hour = df_clean.groupby('hour').size().reset_index(name = 'counts').set_index('hour')
df_hour
fig6 = plt.figure(figsize = (20,7))

ind6 = np.arange(24)

ax6 = plt.subplot(111)

color = []

for i in range(24):

    amount = df_hour.loc[i, 'counts']

    if amount > 12000:

        color.append('darkred')

    elif amount < 4000:

        color.append('lime')

    elif amount > 10000:

        color.append('r')

    elif amount < 6000:

        color.append('g')

    else:

        color.append('olive')

rects = ax6.bar(ind6, df_hour['counts'], width = 0.8,color = color)

ax6.set_xticks(ind6)

ax6.set_xticklabels(df_hour.index)

ax6.set_xlabel('Hour')

ax6.set_ylabel('Amount of crimes')

ax6.set_title('Crimes in Boston by hour')

for rect in rects:

    height = rect.get_height()

    if height > 9999:

        hor = rect.get_x() - 0.13

    else:

        hor = rect.get_x() - 0.03

    ax6.text(hor, 1.01 * height, height, fontsize = 14)
df_shooting = df_clean.groupby('shooting').size().reset_index(name = 'counts').set_index('shooting')
df_shooting
shooting_rate = df_shooting.loc['Y', 'counts']/df_clean.shape[0]
shooting_rate
fig7 = plt.figure(figsize = (10,5))

labels = ['Shooting', 'No shooting']

ax7 = plt.subplot(111)

size = [shooting_rate, 1 - shooting_rate]

ax7.pie(size, explode = [0.5,0], labels = labels, autopct = '%1.2f%%', shadow = True, colors = ['red','blue'])

ax7.axis('equal')

ax7.legend()

ax7.set_title('What percentage of crimes involve shooting?')
import json

with open('../input/boston-neighborhoods-geojson/Boston_Neighborhoods.geojson', 'r') as f:

    boston_geojson = json.load(f)

features = boston_geojson['features']

nbh_list = []

for feature in features:

    nbh_list.append(feature['properties']['Name'])

print(nbh_list)
boston_geojson['features'][0]['geometry']
nbh_list.sort()
nbh_list
nbh_list.remove('Chinatown')

nbh_list.remove('Bay Village')

nbh_list.remove('Leather District')
nbh_list
nbh_pop = {

    'Allston':22312,

 'Back Bay':16622,

 'Beacon Hill':9023,

 'Brighton':52685,

 'Charlestown':16439,

 'Dorchester':114249,

 'Downtown':15992,

 'East Boston':40508,

 'Fenway':33895,

 'Harbor Islands':535,

 'Hyde Park':32317,

 'Jamaica Plain':35541,

 'Longwood':4861,

 'Mattapan':22500,

 'Mission Hill':16874,

 'North End':8608,

 'Roslindale':26368,

 'Roxbury':49111,

 'South Boston':31110,

 'South Boston Waterfront':2564,

 'South End':29612,

 'West End':5423,

 'West Roxbury':30445

}
df_nb = pd.DataFrame.from_dict(data = nbh_pop, orient = 'index', columns = ['population'])
df_nb
df_nb.to_csv('population.csv')
from shapely.geometry import Point, shape
def point_to_neighborhood (lat, long, geojson):

    point = Point(long, lat)

    features = geojson['features']

    for feature in features:

        polygon = shape(feature['geometry'])

        neighborhood = feature['properties']['Name']

        if polygon.contains(point):

            if neighborhood == 'Chinatown' or neighborhood == 'Leather District':

                return 'Downtown'

            elif neighborhood == 'Bay Village':

                return 'South End'

            else:

                return neighborhood

    print(f'Point ({long},{lat}) is not in Boston.')

    return None
boston_geojson
point_to_neighborhood(42.372269, -71.039015, boston_geojson)
df_nafree = df_clean.dropna(subset = ['lat','long'])
df_nafree.shape
df_nafree.shape[0]/df_clean.shape[0]
for index, row in df_nafree.iterrows():

    lat = df_nafree.at[index, 'lat']

    long = df_nafree.at[index, 'long']

    #print(index)

    #print(lat)

    #print(long)

    neighborhood = point_to_neighborhood(lat, long, boston_geojson)

    #print(neighborhood)

    df_nafree.at[index, 'Neighborhood'] = neighborhood

df_nafree.tail(10)
df_nafree.to_csv('crimes_with_neighborhoods.csv')
df_nbh = df_nafree.groupby('Neighborhood').size().reset_index(name = 'count').set_index('Neighborhood')
df_nbh
df_nafree['Neighborhood'].isna().sum()
df_nbh.index.size
df_nbh_full = pd.concat([df_nb,df_nbh],axis = 1, sort = True)
df_nbh_full
df_nbh_full.at['Harbor Islands','count'] = 0
for index, row in df_nbh_full.iterrows():

    df_nbh_full.at[index, 'Crime rate'] = df_nbh_full.at[index, 'count'] / df_nbh_full.at[index, 'population']
df_nbh_full = df_nbh_full.sort_values('Crime rate', ascending = False)
df_nbh_full
df.tail(5)
df.sort_values(by = 'occurred_on_date').head(5)
df.sort_values(by = 'occurred_on_date').tail(5)
for index, row in df_nbh_full.iterrows():

    df_nbh_full.at[index, 'Crime rate'] = df_nbh_full.at[index, 'Crime rate'] * 365 / 1322
for index, row in df_nbh_full.iterrows():

    df_nbh_full.at[index, 'Crime rate'] = df_nbh_full.at[index, 'Crime rate'] * 100000
df_nbh_full
df_nbh_full.index
multiliner(['a','b','c'],3)
fig0 = plt.figure(figsize = (20,10))

ax0 = plt.subplot(111)

ind0 = np.arange(23)

crime_rate_by_neighborhood = df_nbh_full['Crime rate']

rects = ax0.bar(ind0, crime_rate_by_neighborhood, width = 0.8, color = 'r')

ax0.set_xticks(ind0)

ax0.set_xticklabels(multiliner(df_nbh_full.index.tolist(),3))

ax0.set_title('Crime rate (per 100,000 residents) of neighborhoods of Boston')

ax0.set_ylabel('Crime rate')

for rect in rects:

    height = rect.get_height()

    if height >= 9999.5:

        hor = rect.get_x() - 0.02

    else:

        hor = rect.get_x() + 0.05

    ax0.text(hor, 1.01 * height + 250, int(round(height)), fontsize = 12)
df_particulars = df_nafree.dropna(subset = ['Neighborhood'])
df_particulars.shape
df_particulars['offense_description'].unique().tolist()
df_particulars['offense_code_group'].unique().tolist()
df_p2 = df_particulars.loc[df['ucr_part'] == 'Part Two']
df_2p = df_p2.groupby('offense_code_group').size().reset_index(name = 'count').set_index('offense_code_group')
df_2p
df_p1 = df_particulars.loc[df['ucr_part'] == 'Part One']
df_1p = df_p1.groupby('offense_code_group').size().reset_index(name = 'count').set_index('offense_code_group')
df_1p
df_1d = df_p1.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')
df_1d
l_op = df.groupby('offense_code_group').size().reset_index(name = 'counts')['offense_code_group'].tolist()
l_op
l_od = df.groupby('offense_description').size().reset_index(name = 'counts')['offense_description'].tolist()
l_od
df_murder = df_particulars.loc[df_particulars['offense_description'] == 'MURDER, NON-NEGLIGIENT MANSLAUGHTER']
df_murdern = df_murder.groupby('Neighborhood').size().reset_index(name = 'murder_and_nonnegligent_manslaughter').set_index('Neighborhood')
df_murdern
df_aa = df_particulars.loc[df_particulars['offense_code_group'] == 'Aggravated Assault']
df_aa_d = df_aa.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')
df_aa_d
df_aan = df_aa.groupby('Neighborhood').size().reset_index(name = 'aggravated_assault').set_index('Neighborhood')
df_aan
df_robbery = df_particulars.loc[df_particulars['offense_code_group'] == 'Robbery']
df_r_d = df_robbery.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')
df_r_d
df_robberyn = df_robbery.groupby('Neighborhood').size().reset_index(name = 'robbery').set_index('Neighborhood')
df_robberyn
df_arson = df_particulars.loc[df_particulars['offense_code_group'] == 'Arson']
df_arson_d = df_arson.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')
df_arson_d
df_arsonn = df_arson.groupby('Neighborhood').size().reset_index(name = 'arson').set_index('Neighborhood')
df_arsonn
df_at = df_particulars.loc[df_particulars['offense_code_group'] == 'Auto Theft']
df_at_d = df_at.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')
df_at_d
df_atn = df_at.groupby('Neighborhood').size().reset_index(name = 'auto_theft').set_index('Neighborhood')
df_atn
df_larceny = df_particulars.loc[(df_particulars['offense_code_group'] == 'Larceny') | (df_particulars['offense_code_group'] == 'Larceny From Motor Vehicle')]
df_larceny_d = df_larceny.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')
df_larceny_d
df_larcenyn = df_larceny.groupby('Neighborhood').size().reset_index(name = 'larceny').set_index('Neighborhood')
df_larcenyn
df_burglary = df_particulars.loc[(df_particulars['offense_code_group'] == 'Other Burglary') | (df_particulars['offense_code_group'] == 'Commercial Burglary') | (df_particulars['offense_code_group'] == 'Residential Burglary') | (df_particulars['offense_code_group'] == 'Burglary - No Property Taken')]
df_burglary_d = df_burglary.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')
df_burglary_d
df_burglaryn = df_burglary.groupby('Neighborhood').size().reset_index(name = 'burglary').set_index('Neighborhood')
df_burglaryn
df_nbh_crimes = pd.concat([df_nbh_full, df_murdern, df_aan, df_robberyn, df_arsonn, df_atn, df_larcenyn, df_burglaryn], axis = 1, sort = True)
df_nbh_crimes
df_nbh_crimes.fillna(0, inplace = True)
df_nbh_crimes
for index, row in df_nbh_crimes.iterrows():

    df_nbh_crimes.at[index, 'violent_crimes'] = df_nbh_crimes.at[index, 'murder_and_nonnegligent_manslaughter'] + df_nbh_crimes.at[index, 'aggravated_assault'] + df_nbh_crimes.at[index, 'robbery']

    df_nbh_crimes.at[index, 'property_crimes'] = df_nbh_crimes.at[index, 'arson'] + df_nbh_crimes.at[index, 'auto_theft'] + df_nbh_crimes.at[index, 'larceny'] + df_nbh_crimes.at[index, 'burglary']

    df_nbh_crimes.at[index, 'murder_rate'] = df_nbh_crimes.at[index, 'murder_and_nonnegligent_manslaughter'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000

    df_nbh_crimes.at[index, 'aggravated_assault_rate'] = df_nbh_crimes.at[index, 'aggravated_assault'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000

    df_nbh_crimes.at[index, 'robbery_rate'] = df_nbh_crimes.at[index, 'robbery'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000

    df_nbh_crimes.at[index, 'arson_rate'] = df_nbh_crimes.at[index, 'arson'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000

    df_nbh_crimes.at[index, 'auto_theft_rate'] = df_nbh_crimes.at[index, 'auto_theft'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000

    df_nbh_crimes.at[index, 'larceny_rate'] = df_nbh_crimes.at[index, 'larceny'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000

    df_nbh_crimes.at[index, 'burglary_rate'] = df_nbh_crimes.at[index, 'burglary'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000

    df_nbh_crimes.at[index, 'violent_crime_rate'] = df_nbh_crimes.at[index, 'violent_crimes'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000

    df_nbh_crimes.at[index, 'property_crime_rate'] = df_nbh_crimes.at[index, 'property_crimes'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000
df_nbh_crimes
df_nbh_crimes.to_csv('crime_rates_by_neighborhood.csv')
fig01 = plt.figure(figsize = (20,10))

ax01 = plt.subplot(111)

ind01 = np.arange(23)

murder_rate_by_neighborhood = df_nbh_crimes['murder_rate']

rects = ax01.bar(ind01, murder_rate_by_neighborhood, width = 0.8, color = 'r')

ax01.set_xticks(ind01)

ax01.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))

ax01.set_title('Murder rate (per 100,000 residents) of neighborhoods of Boston')

ax01.set_ylabel('Murder rate')

for rect in rects:

    height = rect.get_height()

    if height >= 9.95:

        hor = rect.get_x() + 0.08

    else:

        hor = rect.get_x() + 0.12

    ax01.text(hor, 1.01 * height + 0.1, "{:.1f}".format(height), fontsize = 12)
fig02 = plt.figure(figsize = (20,10))

ax02 = plt.subplot(111)

ind02 = np.arange(23)

aa_rate_by_neighborhood = df_nbh_crimes['aggravated_assault_rate']

rects = ax02.bar(ind02, aa_rate_by_neighborhood, width = 0.8, color = 'r')

ax02.set_xticks(ind02)

ax02.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))

ax02.set_title('Aggravated assault rate (per 100,000 residents) of neighborhoods of Boston')

ax02.set_ylabel('Aggravated assault rate')

for rect in rects:

    height = rect.get_height()

    if height >= 999.95:

        hor = rect.get_x() - 0.04

    elif height >= 99.95:

        hor = rect.get_x()

    elif height >= 9.95:

        hor = rect.get_x() + 0.08

    else:

        hor = rect.get_x() + 0.12

    ax02.text(hor, 1.01 * height + 10, "{:.1f}".format(height), fontsize = 12)
fig03 = plt.figure(figsize = (20,10))

ax03 = plt.subplot(111)

ind03 = np.arange(23)

r_rate_by_neighborhood = df_nbh_crimes['robbery_rate']

rects = ax03.bar(ind03, r_rate_by_neighborhood, width = 0.8, color = 'r')

ax03.set_xticks(ind03)

ax03.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))

ax03.set_title('Robbery rate (per 100,000 residents) of neighborhoods of Boston')

ax03.set_ylabel('Robbery rate')

for rect in rects:

    height = rect.get_height()

    if height >= 999.95:

        hor = rect.get_x() - 0.04

    elif height >= 99.95:

        hor = rect.get_x()

    elif height >= 9.95:

        hor = rect.get_x() + 0.08

    else:

        hor = rect.get_x() + 0.12

    ax03.text(hor, 1.01 * height + 7, "{:.1f}".format(height), fontsize = 12)
fig04 = plt.figure(figsize = (20,10))

ax04 = plt.subplot(111)

ind04 = np.arange(23)

ar_rate_by_neighborhood = df_nbh_crimes['arson_rate']

rects = ax04.bar(ind04, ar_rate_by_neighborhood, width = 0.8, color = 'r')

ax04.set_xticks(ind04)

ax04.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))

ax04.set_title('Arson rate (per 100,000 residents) of neighborhoods of Boston')

ax04.set_ylabel('Arson rate')

for rect in rects:

    height = rect.get_height()

    if height >= 999.95:

        hor = rect.get_x() - 0.04

    elif height >= 99.95:

        hor = rect.get_x()

    elif height >= 9.95:

        hor = rect.get_x() + 0.08

    else:

        hor = rect.get_x() + 0.16

    ax04.text(hor, 1.01 * height + 0.02, "{:.1f}".format(height), fontsize = 12)
fig05 = plt.figure(figsize = (20,10))

ax05 = plt.subplot(111)

ind05 = np.arange(23)

at_rate_by_neighborhood = df_nbh_crimes['auto_theft_rate']

rects = ax05.bar(ind05, at_rate_by_neighborhood, width = 0.8, color = 'r')

ax05.set_xticks(ind05)

ax05.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))

ax05.set_title('Auto theft rate (per 100,000 residents) of neighborhoods of Boston')

ax05.set_ylabel('Auto theft rate')

for rect in rects:

    height = rect.get_height()

    if height >= 999.95:

        hor = rect.get_x() - 0.04

    elif height >= 99.95:

        hor = rect.get_x()

    elif height >= 9.95:

        hor = rect.get_x() + 0.08

    else:

        hor = rect.get_x() + 0.16

    ax05.text(hor, 1.01 * height + 3, "{:.1f}".format(height), fontsize = 12)
fig06 = plt.figure(figsize = (20,10))

ax06 = plt.subplot(111)

ind06 = np.arange(23)

la_rate_by_neighborhood = df_nbh_crimes['larceny_rate']

rects = ax06.bar(ind06, la_rate_by_neighborhood, width = 0.8, color = 'r')

ax06.set_xticks(ind06)

ax06.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))

ax06.set_title('Larceny rate (per 100,000 residents) of neighborhoods of Boston')

ax06.set_ylabel('Larceny rate')

for rect in rects:

    height = rect.get_height()

    if height >= 999.95:

        hor = rect.get_x() - 0.06

    elif height >= 99.95:

        hor = rect.get_x() + 0.02

    elif height >= 9.95:

        hor = rect.get_x() + 0.08

    else:

        hor = rect.get_x() + 0.16

    ax06.text(hor, 1.01 * height + 30, "{:.1f}".format(height), fontsize = 12)
fig07 = plt.figure(figsize = (20,10))

ax07 = plt.subplot(111)

ind07 = np.arange(23)

b_rate_by_neighborhood = df_nbh_crimes['burglary_rate']

rects = ax07.bar(ind07, b_rate_by_neighborhood, width = 0.8, color = 'r')

ax07.set_xticks(ind07)

ax07.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))

ax07.set_title('Burglary rate (per 100,000 residents) of neighborhoods of Boston')

ax07.set_ylabel('Burglary rate')

for rect in rects:

    height = rect.get_height()

    if height >= 999.95:

        hor = rect.get_x() - 0.06

    elif height >= 99.95:

        hor = rect.get_x() + 0.02

    elif height >= 9.95:

        hor = rect.get_x() + 0.08

    else:

        hor = rect.get_x() + 0.16

    ax07.text(hor, 1.01 * height + 3, "{:.1f}".format(height), fontsize = 12)