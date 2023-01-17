import pandas as pd

from shapely.geometry import  Point

import geopandas as gpd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import seaborn as sns

from matplotlib import cm

import urllib.request

import shutil

import zipfile

import os

import re

import contextily as ctx

import geoplot as gplt

import lightgbm as lgb

import eli5

from eli5.sklearn import PermutationImportance

from lightgbm import LGBMClassifier

from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots

import shap
train = pd.read_csv('data/train.csv', parse_dates=['Dates'])

test = pd.read_csv('data/test.csv', parse_dates=['Dates'], index_col='Id')
train.Dates.describe()
train.shape
train.head()
train.dtypes
train.duplicated().sum()
def create_gdf(df):

    gdf = df.copy()

    gdf['Coordinates'] = list(zip(gdf.X, gdf.Y))

    gdf['Coordinates'] = gdf['Coordinates'].apply(Point)

    gdf = gpd.GeoDataFrame(gdf, geometry='Coordinates',

                          crs={'init': 'epsg:4326'})

    return gdf



train_gdf = create_gdf(train)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))



f, ax = plt.subplots(1, figsize=(9,9))

ax = world.plot(color='white', edgecolor='black', axes=ax)

train_gdf.plot(ax=ax, color='red');
train_gdf[train_gdf.Y > 70]
train_gdf[train_gdf.Y > 70].count()[0]
train.drop_duplicates(inplace=True)
train.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)

test.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
imputer = SimpleImputer(strategy='mean')
for district in train['PdDistrict'].unique():

    train.loc[train['PdDistrict']==district, ['X', 'Y']] = imputer.fit_transform(

        train.loc[train['PdDistrict']==district, ['X', 'Y']])

    # fit은 train 데이터로 해주었기때문에 transform만 적용

    test.loc[test['PdDistrict']==district, ['X', 'Y']] = imputer.transform(

        test.loc[test['PdDistrict']==district, ['X', 'Y']])    

    

train_gdf = create_gdf(train)
train['Date'] = train['Dates'].dt.date

train['Year'] = train['Dates'].dt.year

train['Month'] = train['Dates'].dt.month

train['Day'] = train['Dates'].dt.day

train['Hour'] = train['Dates'].dt.hour
year_series = train.groupby('Year').count().iloc[:, 0]

g = sns.barplot(x=year_series.index, y=year_series)

g.set_xticklabels(g.get_xticklabels(), rotation=45);
month_series = train.groupby('Month').count().iloc[:, 0]

sns.barplot(x=month_series.index, y=month_series);
hour_series = train.groupby('Hour').count().iloc[:, 0]

sns.barplot(x=hour_series.index, y=hour_series);
palette = sns.color_palette()



plt.figure(figsize=(10, 6))

date_count = train.groupby('Date').count().iloc[:, 0]

sns.kdeplot(data=date_count, shade=True)

plt.axvline(x=date_count.median(), ymax=0.95, linestyle='--', color=palette[1])

plt.annotate('Median ' + str(date_count.median()),

             xy=(date_count.median(), 0.004),

             xytext=(200, 0.005),

             arrowprops=dict(arrowstyle='->', color=palette[1], shrinkB=10))

plt.title('Distribution of number of incidents per day',fontdict={'fontsize':16})

plt.xlabel('Crime Incidents')

plt.ylabel('Density')

plt.legend().remove()

plt.show()
weekday_series = train.groupby('DayOfWeek').count().iloc[:,0]

weekday_series = weekday_series.reindex([

    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',

    'Sunday'])



with sns.axes_style("whitegrid"):

    f, ax = plt.subplots(1, figsize=(10, 6))

    sns.barplot(

        weekday_series.index, (weekday_series.values / weekday_series.values.sum()) * 100,

        palette=cm.ScalarMappable(cmap='Blues').to_rgba(weekday_series.values))



plt.title('Incidents per Weekday', fontdict={'fontsize':16})

plt.xlabel('Weekday')

plt.ylabel('Percent of Incients (%)');
category_counts = train.groupby('Category').count().iloc[:, 0].sort_values(ascending=False)
category_counts
# OTHER OFFENSES를 제일 아래 두기 위해

category_counts = category_counts.reindex(

    np.append(np.delete(category_counts.index, 1), 'OTHER OFFENSES'))
with sns.axes_style("whitegrid"):

    f, ax = plt.subplots(1, figsize=(10, 10))

    sns.barplot(

        category_counts.values / category_counts.values.sum() * 100,

        category_counts.index,

        orient='h',

        palette='Blues_d')

plt.title('Incidents per Crime Category', fontdict={'fontsize': 16})

plt.xlabel('Incidents (%)');
# Downloading the shapefile of the area 

url = 'https://data.sfgov.org/api/geospatial/wkhw-cjsf?method=export&format=Shapefile'

with urllib.request.urlopen(url) as response, open('pd_data.zip', 'wb') as out_file:

    shutil.copyfileobj(response, out_file)

    

# Unzipping it

with zipfile.ZipFile('pd_data.zip', 'r') as zip_ref:

    zip_ref.extractall('pd_data')

    

# Loading to a geopandas dataframe

for filename in os.listdir('./pd_data/'):

    if re.match(".+\.shp", filename):

        pd_districts = gpd.read_file('./pd_data/'+filename)

        break

        

# Merging our train dataset with the geo-dataframe

pd_districts = pd_districts.merge(

    train.groupby('PdDistrict').count().iloc[:, [0]].rename(

        columns={'Dates': 'Incidents'}),

    left_on='district',

    right_index=True)



# Transforming the coordinate system to Spherical Mercator for

# compatibility with the tiling background

pd_districts = pd_districts.to_crs({'init': 'epsg:3857'})



# Calculating the incidents per day for every district

train_days = train.groupby('Date').count().shape[0]

pd_districts['inc_per_day'] = pd_districts.Incidents/train_days
# Ploting the data

fig, ax = plt.subplots(figsize=(10, 10))

pd_districts.plot(

    column='inc_per_day',

    cmap='Reds',

    alpha=0.6,

    edgecolor='r',

    linestyle='-',

    linewidth=1,

    legend=True,

    ax=ax);
fig, ax = plt.subplots(figsize=(10, 10))

pd_districts.plot(

    column='inc_per_day',

    cmap='Reds',

    alpha=0.6,

    edgecolor='r',

    linestyle='-',

    linewidth=1,

    legend=True,

    ax=ax);



def add_basemap(ax, zoom, url='http://tile.stamen.com/terrain/tileZ/tileX/tileY.png'):

    """Function that add the tile background to the map"""

    xmin, xmax, ymin, ymax = ax.axis()

    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)

    ax.imshow(basemap, extent=extent, interpolation='bilinear')

    # restore original x/y limits

    ax.axis((xmin, xmax, ymin, ymax))



# Adding the background

add_basemap(ax, zoom=11, url=ctx.sources.ST_TONER_LITE)



# Adding the name of the districts

for index in pd_districts.index:

    plt.annotate(

        pd_districts.loc[index].district,

        (pd_districts.loc[index].geometry.centroid.x,

         pd_districts.loc[index].geometry.centroid.y),

        color='#353535',

        fontsize='large',

        fontweight='heavy',

        horizontalalignment='center'

    )



ax.set_axis_off()

plt.show()
crimes = train['Category'].unique().tolist()

crimes.remove('TREA')



pd_districts = pd_districts.to_crs({'init': 'epsg:4326'})



# geometry containing the union of all geometries 

sf_land = pd_districts.unary_union
sf_land
sf_land = gpd.GeoDataFrame(gpd.GeoSeries(sf_land), crs={'init':'epsg:4326'})

sf_land = sf_land.rename(columns={0:'geometry'}).set_geometry('geometry')
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(12,12))

for i, crime in enumerate(np.random.choice(crimes, size=9, replace=False)):

    data = train_gdf.loc[train_gdf['Category'] == crime]

    ax = fig.add_subplot(3, 3, i+1)

    gplt.kdeplot(data,

                shade=True,

                shade_lowest=False, # False일 때, 0에 가까운 빈도를 가진 구역은 진하기를 표현하지 않음

                clip=sf_land.geometry,# 주어진 구역만 시각화

                cmap='Reds',

                ax=ax)

    gplt.polyplot(sf_land, ax=ax)

    ax.set_title(crime)

plt.suptitle('Geographic Density of Diffenrent Crimes')

fig.tight_layout(rect=[0, 0.03, 1, 0.95]);
# as_index=False: Hour, Date, Category를 index로 지정하지 않음

data = train.groupby(['Hour', 'Date', 'Category'],

                    as_index=False).count().iloc[:, :4]

data
data.rename(columns={'Dates': 'Incidents'}, inplace=True)

data = data.groupby(['Hour', 'Category'], as_index=False).mean()

data = data.loc[data['Category'].isin(

    ['ROBBERY', 'GAMBLING', 'BUGLARY', 'ARSON', 'PROSTITUTION'])]



data
sns.set_style('whitegrid')

fig, ax = plt.subplots(figsize=(14,4))

ax = sns.lineplot(data=data, x='Hour', y='Incidents', hue='Category')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6)

plt.suptitle('Average number of incidents per hour')

fig.tight_layout(rect=[0, 0, 1, 0.95])
naive_vals = train.groupby('Category').count().iloc[:, 0] / train.shape[0]

n_rows = test.shape[0]



naive_vals
submission = pd.DataFrame(

    np.repeat(np.array(naive_vals), n_rows).reshape(39, n_rows).transpose(),

    columns=naive_vals.index)
submission
def feature_engineering(data):

    # object type -> datetime type으로

    data['Date'] = pd.to_datetime(data['Dates'].dt.date) 

    # timedelta type -> int type으로

    data['n_days'] = (

        data['Date'] - data['Date'].min()).apply(lambda x: x.days)

    data['Day'] = data['Dates'].dt.day

    data['DayOfWeek'] = data['Dates'].dt.weekday

    data['Month'] = data['Dates'].dt.month

    data['Year'] = data['Dates'].dt.year

    data['Hour'] = data['Dates'].dt.hour

    data['Minute'] = data['Dates'].dt.minute

    data['Block'] = data['Address'].str.contains('block', case=False)

    

    data.drop(columns=['Dates', 'Date', 'Address'], inplace=True)

    

    return data
train = feature_engineering(train)

train.drop(columns=['Descript', 'Resolution'], inplace=True)

test = feature_engineering(test)

train.head()
PdDistrict_le = LabelEncoder()

train['PdDistrict'] = PdDistrict_le.fit_transform(train['PdDistrict'])

test['PdDistrict'] = PdDistrict_le.transform(test['PdDistrict'])



Category_le = LabelEncoder()

y = Category_le.fit_transform(train.pop('Category'))



train_X, val_X, train_y, val_y = train_test_split(train, y)

model = LGBMClassifier(objective='multiclass', num_class=39).fit(train_X, train_y)



# 하나의 column을 섞어 성능을 구했을 때 성능 감소량이 그 feature의 중요도임

perm = PermutationImportance(model).fit(val_X, val_y)

eli5.show_weights(perm, feature_names=val_X.columns.tolist())
# Loading the data

train = pd.read_csv('./data/train.csv', parse_dates=['Dates'])

test = pd.read_csv('./data/test.csv', parse_dates=['Dates'], index_col='Id')



# Data cleaning

train.drop_duplicates(inplace=True)

train.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)

test.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)



imp = SimpleImputer(strategy='mean')



for district in train['PdDistrict'].unique():

    train.loc[train['PdDistrict'] == district, ['X', 'Y']] = imp.fit_transform(

        train.loc[train['PdDistrict'] == district, ['X', 'Y']])

    test.loc[test['PdDistrict'] == district, ['X', 'Y']] = imp.transform(

        test.loc[test['PdDistrict'] == district, ['X', 'Y']])

train_data = lgb.Dataset(

    train, label=y, categorical_feature=['PdDistrict'], free_raw_data=False)



# Feature Engineering

train = feature_engineering(train)

train.drop(columns=['Descript','Resolution'], inplace=True)

test = feature_engineering(test)



# Encoding the Categorical Variables

le1 = LabelEncoder()

train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])

test['PdDistrict'] = le1.transform(test['PdDistrict'])



le2 = LabelEncoder()

X = train.drop(columns=['Category'])

y= le2.fit_transform(train['Category'])



# Creating the model

train_data = lgb.Dataset(

    X, label=y, categorical_feature=['PdDistrict'])



params = {'boosting':'gbdt',

          'objective':'multiclass',

          'num_class':39,

          'max_delta_step':0.9,

          'min_data_in_leaf': 21,

          'learning_rate': 0.4,

          'max_bin': 465,

          'num_leaves': 41

         }



bst = lgb.train(params, train_data, 100)



predictions = bst.predict(test)



# Submitting the results

submission = pd.DataFrame(

    predictions,

    columns=le2.inverse_transform(np.linspace(0, 38, 39, dtype='int16')),

    index=test.index)

submission.to_csv(

    'LGBM_final.csv', index_label='Id')