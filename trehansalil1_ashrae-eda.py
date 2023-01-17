import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import stats



%matplotlib inline
test = pd.read_csv('../input/ashrae-energy-prediction/test.csv',parse_dates=['timestamp'])
train = pd.read_csv('../input/ashrae-energy-prediction/train.csv',parse_dates=['timestamp'])
weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv',parse_dates=['timestamp'])
weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv',parse_dates=['timestamp'])
building_metadata = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')
sample_submission = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')
train.head()
weather_train.head()
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
test = reduce_mem_usage(test)

train = reduce_mem_usage(train)

weather_test = reduce_mem_usage(weather_test)

weather_train = reduce_mem_usage(weather_train)

building_metadata = reduce_mem_usage(building_metadata)
weather_train.isnull().sum()
weather_train.head()
weather_train = weather_train.groupby('site_id').apply(lambda x: x.set_index('timestamp').interpolate(method='time',limit_direction='both').reset_index()).drop(columns='site_id').reset_index().drop(columns='level_1')

weather_test = weather_test.groupby('site_id').apply(lambda x: x.set_index('timestamp').interpolate(method='time',limit_direction='both').reset_index()).drop(columns='site_id').reset_index().drop(columns='level_1')



weather_train.isnull().sum()
weather_train.head()


weather_train['cloud_coverage'] = weather_train['cloud_coverage'].round()

weather_train['wind_direction'] = weather_train['wind_direction'].round()

weather_train['precip_depth_1_hr'] = weather_train['precip_depth_1_hr'].round()
weather_train.groupby('site_id').mean().sort_values(by='cloud_coverage')
train = train.merge(building_metadata, on='building_id', how='left')

test = test.merge(building_metadata, on='building_id', how='left')



train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')

test = test.merge(weather_test, on=['site_id', 'timestamp'], how='left')
del weather_train

del weather_test

del building_metadata
train['aday']=train['timestamp'].dt.day

train['amonth']=train['timestamp'].dt.month

train['ayear']=train['timestamp'].dt.year

train['aquarter']=train['timestamp'].dt.quarter

train['adayofweek']=train['timestamp'].dt.dayofweek

train['adayofyear']=train['timestamp'].dt.dayofyear

train['adate']=train['timestamp'].dt.date

train['ahour']=train['timestamp'].dt.hour



test['aday']=test['timestamp'].dt.day

test['amonth']=test['timestamp'].dt.month

test['ayear']=test['timestamp'].dt.year

test['aquarter']=test['timestamp'].dt.quarter

test['adayofweek']=test['timestamp'].dt.dayofweek

test['adayofyear']=test['timestamp'].dt.dayofyear

test['adate']=test['timestamp'].dt.date

test['ahour']=test['timestamp'].dt.hour
test['source']='test'

train['source']='train'
train.head()
train.columns
train.shape, test.shape
train.head()
train.info()
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
train.info()
train['ayear'].value_counts()
sns.countplot(train['meter'])

plt.show() # The meter count varies from meter to meter
train['site_id'].value_counts().sort_index().plot.bar()

plt.show()
plt.figure(figsize=(10,6))

train['primary_use'].value_counts().plot(kind='bar')

plt.show()
plt.figure(figsize=(12,6))

sns.distplot(train['square_feet'])

plt.show()
plt.figure(figsize=(18,6))

train['year_built'].value_counts().sort_index(ascending= False).plot(kind='bar')

plt.show()
train['year_built'].value_counts().to_frame().iloc[:2013]
plt.figure(figsize=(12,6))

train['floor_count'].value_counts().plot(kind='bar')

plt.show()
sns.boxplot(train['air_temperature'].dropna(), showfliers=True)



plt.show()
sns.distplot(train['air_temperature'].dropna())

plt.show()
d= train['cloud_coverage'].value_counts().sort_index().to_frame()
(d).plot.bar()

plt.show()
d.T
sns.stripplot(d.index,d.cloud_coverage)

plt.show()
sns.boxplot(train['dew_temperature'].dropna(), showfliers=True)

plt.show()
train['precip_depth_1_hr'].dropna().value_counts().sort_index().plot.density()

plt.show()
sns.boxplot(train['precip_depth_1_hr'].dropna(), showfliers = False)

plt.show()
m=train['precip_depth_1_hr'].dropna().value_counts().sort_index()

m

# Suspecting precip_depth_1_hr values under 0 to be incorrect entries


plt.figure(figsize=(24,7))

sns.barplot(m.index,m.values)

plt.xticks(rotation=85)

plt.ylim((0,553352))

plt.show()
sns.distplot(m.to_frame()['precip_depth_1_hr'])

plt.ylim((0,0.0000004))

plt.show()
plt.figure(figsize=(24,7))

sns.boxplot(train['sea_level_pressure'].dropna(), showfliers=True)

plt.show() # varies from 995 to 1034
train['sea_level_pressure'].dropna().value_counts().sort_index().plot.hist(bins=50)

plt.show()
sns.boxplot(train['wind_direction'].dropna(), showfliers=True)

plt.show() # varies from 0-360
sns.boxplot(train['wind_speed'].dropna(), showfliers=True)

plt.show() # the plot is right skewed
sns.boxplot(train.groupby('building_id')['meter_reading'].mean(), showfliers=False)

plt.show() # the plot is right skewed
'''plt.figure(figsize=(24,7))

#sns.boxplot(train['building_id'], train['meter_reading'], showfliers=False)

sns.barplot(train['building_id'], train['meter_reading'])

plt.xticks(rotation=90)

plt.show()'''
plt.figure(figsize=(24,7))

sns.boxplot(train['site_id'], train['meter_reading'], showfliers=False)

plt.show() # site: 5,8, 12,3,11 are the most efficient sites
train['amonth'].value_counts().sort_index()
plt.figure(figsize=(24,7))

sns.boxplot(train['site_id'], train['air_temperature'], showfliers=True)

plt.show()
train.groupby(['site_id'])['air_temperature'].mean().sort_index()
m=train.groupby(['site_id','building_id'])['air_temperature'].mean()

m[0][0]
d={}

for i in range(16):

    n = np.array(m[i])

    d.update({i:[i,np.mean(n),np.std(n)]})

    print('For site_id = ', i,', mean value is: ',np.mean(n), 'standard dev value is: ', np.std(n))
sns.distplot(train.loc[train.site_id == 5, 'air_temperature'].dropna())

plt.show()
pd.DataFrame(d, index=['site_index','mean', 'std']).T.sort_values(by='mean').iloc[:,[0,1]].set_index('site_index').plot(kind='bar')

plt.show()
pd.DataFrame(d, index=['site_index','mean', 'std']).T.sort_values(by='mean').iloc[:,[0,1]].set_index('site_index').plot(kind='bar')

pd.DataFrame(d, index=['site_index','mean', 'std']).T.sort_values(by='std').iloc[:,[0,2]].set_index('site_index').plot(kind='bar')

plt.grid(True)

plt.show()
train.columns
sns.boxplot(train['meter'], train['meter_reading'], showfliers=False)

plt.show()
plt.figure(figsize=(20,6))

train.groupby(['site_id','building_id'])['meter_reading'].mean().sort_values()[-50:].plot.bar()

plt.show()
plt.figure(figsize=(20,6))

train.groupby(['site_id','building_id'])['meter_reading'].mean().sort_values()[:50].plot.bar()

plt.show()
plt.figure(figsize=(24,7))

sns.boxplot(train['primary_use'], train['meter_reading'], showfliers=False)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(24,7))

sns.boxplot(train['primary_use'], train['meter_reading'], showfliers=True)

plt.xticks(rotation=90)

plt.show()
m=train.groupby('square_feet')['meter_reading'].mean().to_frame()

plt.figure(figsize=(18,7))

sns.scatterplot(m.index,m['meter_reading']);

plt.ylim((0,5000))

m=train.groupby('square_feet')['meter_reading'].median().to_frame()

sns.scatterplot(m.index,m['meter_reading']);

plt.ylim((0,5000))

plt.grid(True)

plt.show()
plt.figure(figsize=(24,7))

sns.boxplot(train.groupby('year_built')['meter_reading'].mean(), showfliers=False)

plt.show()
plt.figure(figsize=(24,7))

train.groupby('year_built')['meter_reading'].mean().plot.bar()

plt.xticks(rotation=90)

plt.show()
m=train.groupby('year_built')['meter_reading'].mean().to_frame()

plt.figure(figsize=(18,7))

sns.scatterplot(m.index,m['meter_reading'])



m=train.groupby('year_built')['meter_reading'].median().to_frame()

sns.scatterplot(m.index,m['meter_reading'])



plt.grid(True)

plt.show()
plt.figure(figsize=(26,7))

sns.barplot(train['year_built'], train['meter_reading'])

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(26,7))

sns.boxplot(train['floor_count'], train['meter_reading'], showfliers=False)

plt.show()
plt.figure(figsize=(26,7))

train.groupby(['floor_count'])['meter_reading'].mean().sort_values().plot.bar()

plt.show() # To analyze floor count & meter reading by grouping on the basis of site_id
plt.figure(figsize=(26,7))

sns.barplot(train['floor_count'], train['meter_reading'])

plt.show()
m=train.groupby('air_temperature')['meter_reading'].mean().to_frame()

plt.figure(figsize=(18,7))

sns.scatterplot(m.index,m['meter_reading'])



plt.show()
m=train.groupby('air_temperature')['meter_reading'].mean().to_frame()

plt.figure(figsize=(18,7))

sns.scatterplot(m.index,m['meter_reading'])

m=train.groupby('dew_temperature')['meter_reading'].mean().to_frame()

sns.scatterplot(m.index,m['meter_reading'])

plt.show()
m=train.groupby('dew_temperature')['meter_reading'].mean().to_frame()

plt.figure(figsize=(18,7))

sns.scatterplot(m.index,m['meter_reading'])

m=train.groupby('dew_temperature')['meter_reading'].median().to_frame()

sns.scatterplot(m.index,m['meter_reading'])

plt.show()
plt.figure(figsize=(24,7))

sns.boxplot(train['cloud_coverage'], train['meter_reading'], showfliers=False)

plt.show() # To analyze cloud_coverage & meter reading by grouping on the basis of site_id
m=train.groupby('precip_depth_1_hr')['meter_reading'].mean().to_frame()

plt.figure(figsize=(18,7))

sns.scatterplot(m.index,m['meter_reading'])

m=train.groupby('precip_depth_1_hr')['meter_reading'].median().to_frame()

sns.scatterplot(m.index,m['meter_reading'])

plt.show()
plt.figure(figsize=(18,7))

m=train.groupby('sea_level_pressure')['meter_reading'].mean().to_frame()

sns.scatterplot(m.index,m['meter_reading'])

m=train.groupby('sea_level_pressure')['meter_reading'].median().to_frame()

sns.scatterplot(m.index,m['meter_reading'])

plt.show()
plt.figure(figsize=(18,7))

m=train.groupby('wind_direction')['meter_reading'].mean().to_frame()

sns.scatterplot(m.index,m['meter_reading'])

m=train.groupby('wind_direction')['meter_reading'].median().to_frame()

sns.scatterplot(m.index,m['meter_reading'])

plt.show()
plt.figure(figsize=(18,7))

m=train.groupby('wind_speed')['meter_reading'].mean().to_frame()

sns.scatterplot(m.index,m['meter_reading'])

m=train.groupby('wind_speed')['meter_reading'].median().to_frame()

sns.scatterplot(m.index,m['meter_reading'])

plt.show()
m = train.groupby('site_id')['air_temperature', 'cloud_coverage'

                         , 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure'

                         , 'wind_direction','wind_speed'].median()
n=[i for i in m.columns]

n

# site_id, month, hour
train.isnull().sum()[train.isnull().sum()!=0]
train.drop(columns=['year_built', 'floor_count'],inplace=True)
train.isnull().sum()[train.isnull().sum()!=0]
m.isnull().sum()[m.isnull().sum()!=0]
for i in n:  

    train[i]=train.groupby(['site_id','amonth', 'ahour'])[i].apply(lambda x: x.fillna(x.median()))

train.isnull().sum()[train.isnull().sum()!=0]
m.isnull().sum()[m.isnull().sum()!=0]
train['wind_speed'].value_counts().sort_index()
d
train.columns
train.groupby('site_id')['timestamp', 'air_temperature', 'cloud_coverage'

                         , 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure'

                         , 'wind_direction','wind_speed',].mean().sort_values(by='air_temperature')

train[train['air_temperature'].isnull()]
plt.figure(figsize=(14, 6))

sns.distplot(train['meter_reading'])

plt.show()
plt.figure(figsize=(15,6))

train.groupby('building_id')['square_feet'].mean().plot(kind='bar')

plt.show()
train.iloc[:,[9,10,11,12,13,14,15]][train['air_temperature'].isnull()]
fig, axes = plt.subplots(1, 1, figsize=(14, 6))

train['timestamp'].dt.floor('d').value_counts().sort_index().plot(ax=axes).set_xlabel('Date', fontsize=14);

test['timestamp'].dt.floor('d').value_counts().sort_index().plot(ax=axes).set_ylabel('Number of training examples', fontsize=14);

axes.set_title('Number of training examples by day', fontsize=16);

axes.legend(['Train', 'Test']);
fig, axes = plt.subplots(1, 1, figsize=(32, 4))

train.iloc[:,[2,3]].set_index(['timestamp']).resample('H').mean()['meter_reading'].plot(ax=axes

                                                                                        , label='By hour').set_ylabel('Meter reading', fontsize=14);

train.iloc[:

           ,[2,3]].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes

                                                                                      , label='By day'

                                                                                     , alpha=1).set_ylabel('Meter reading', fontsize=14);
plt.figure(figsize=(14, 6))

sns.boxplot(x='meter', y='meter_reading', data=train, showfliers=False);



plt.show()
plt.figure(figsize=(14, 6))

sns.boxplot(x='meter', y='air_temperature', data=train);



plt.show()
sample_submission.head()