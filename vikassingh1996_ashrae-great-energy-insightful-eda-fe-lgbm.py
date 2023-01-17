'''Importing Data Manipulattion Moduls'''

import numpy as np

import pandas as pd

from scipy import stats

import os, gc



'''Seaborn and Matplotlib Visualization'''

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")                    

%matplotlib inline



'''plotly Visualization'''

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

init_notebook_mode(connected = True)



'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))
'''Read the dataset from csv file'''

building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')

train = pd.read_csv('../input/ashrae-energy-prediction/train.csv') 

test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')
'''Train and test data at a glance.'''

bold('**Preview of building data**')

display(building.head(3))

bold('**Preview of Weather Train Data:**')

display(weather_train.head(3))

bold('**Preview of Weather Test Data:**')

display(weather_test.head(3))

bold('**Preview of Train Data:**')

display(train.head(3))

bold('**Preview of Test Data:**')

display(test.head(3))
'''Dimension of train and test data'''

bold('**Shape of our train and test data**')

print('Dimension of building:', building.shape) 

print('Dimension of Weather train:',weather_train.shape) 

print('Dimension of Weather test:', weather_test.shape)

print('Dimension of train:',train.shape) 

print('Dimension of test:',test.shape)
'''Merging datasets'''

train = train.merge(building, on = 'building_id', how = 'left')

test = test.merge(building, on = 'building_id', how = 'left')



train = train.merge(weather_train, on = ['site_id', 'timestamp'], how = 'left')

test = test.merge(weather_test, on = ['site_id', 'timestamp'], how = 'left')



del weather_train, weather_test,building
'''Function to reduce the DF size'''

# source: https://www.kaggle.com/kernels/scriptcontent/3684066/download



def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

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

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
'''Variable Description'''

def description(df):

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.iloc[0].values

    summary['Second Value'] = df.iloc[1].values

    summary['Third Value'] = df.iloc[2].values

    return summary
bold('**Variable Description of  train Data:**')

description(train)
bold('**Variable Description of  train Data:**')

description(test)
train["timestamp"] = pd.to_datetime(train["timestamp"])

train["hour"] = np.uint8(train["timestamp"].dt.hour)

train["day"] = np.uint8(train["timestamp"].dt.day)

train["weekday_name"] = train["timestamp"].dt.weekday_name 

train["weekday"] = np.uint8(train["timestamp"].dt.weekday)

train["month"] = np.uint8(train["timestamp"].dt.month)



test["timestamp"] = pd.to_datetime(test["timestamp"])

test["hour"] = np.uint8(test["timestamp"].dt.hour)

test["day"] = np.uint8(test["timestamp"].dt.day)

test["weekday"] = np.uint8(test["timestamp"].dt.weekday)

test["month"] = np.uint8(test["timestamp"].dt.month)
train['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)

test['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)
'''Function to distribution plot'''

def distplot(variable, color):

    global ax

    font_size = 16

    title_size = 20

    plt.rcParams['figure.figsize'] = (18, 10)

    ax = sns.distplot(variable, color = color)

    plt.xlabel('%s' %variable.name, fontsize = font_size)

    plt.ylabel('Count ', fontsize = font_size)

    plt.xticks(fontsize = font_size)

    plt.yticks(fontsize = font_size)

    plt.title(' Distribution of '+'%s' %variable.name, fontsize = title_size)

    plt.show()
'''Distribution of the Meter Reading'''

distplot(train['meter_reading'], 'teal')
'''Summary of meter reading'''

train['meter_reading'].describe()
'''Log tranformation of meter_reading'''

train['meter_reading'] = np.log1p(train['meter_reading'])



bold('**Distribution after log tranformation**')

distplot(train['meter_reading'], 'teal')
bold('**ELECTRICITY THE MOST FREQUENT METER TYPE MEASURED**')

plt.rcParams['figure.figsize'] = (18, 10)

ax = sns.countplot(data = train, x ='meter', palette = 'CMRmap', alpha = 0.5)

ax.set_ylabel('Count', fontsize = 20)

ax.set_xlabel('Meter Type', fontsize = 20)

plt.show()
bold('**THE STEAM METER TYPE IS THE LEAST EFFICIENT, ELECTRICITY THE MOST EFFICIENT**')

plt.rcParams['figure.figsize'] = (18, 10)



temp_df = train[train["meter"]=="Electricity"]

ax = sns.kdeplot(temp_df['meter_reading'], shade = True, label="electricity")

temp_df = train[train["meter"]=="ChilledWater"]

ax = sns.kdeplot(temp_df['meter_reading'], shade = True, label="chill water", color = 'm')

temp_df = train[train["meter"]=="Steam"]

ax = sns.kdeplot(temp_df['meter_reading'], shade = True, label="steam", color = 'lime')

temp_df = train[train["meter"]=="HotWater"]

ax = sns.kdeplot(temp_df['meter_reading'], shade = True, label="hot water", color = 'k')

ax.set_xlabel('Log(Meter Reading)', fontsize = 20)

plt.show()
bold('**SUNDAYS HAVE THE LOWEST READINGS**')

plt.rcParams['figure.figsize'] = (18, 10)

ax = sns.boxplot(data = train, x ='weekday_name', y = 'meter_reading', color = 'teal', boxprops=dict(alpha=.3))

ax.set_ylabel('Log(Meter Reading)', fontsize = 20)

ax.set_xlabel('weekdays', fontsize = 20)

plt.show()
bold('**READINGS HIGHEST DURING THE MIDDLE OF THE DAY**')

plt.rcParams['figure.figsize'] = (18,10)

temp_df = train.groupby('hour').meter_reading.sum()

temp_df.plot(linewidth = 5, color = 'teal')

plt.xlabel('Reading Hour', fontsize = 15)

plt.ylabel('Meter Reading')

plt.show()
bold('**MONTHLY READINGS ARE HIGHEST CHANGES BASED ON BUILDING TYPE**')

temp_df = train.groupby(['month', 'primary_use']).meter_reading.sum().reset_index()

ax = sns.FacetGrid(temp_df, col="primary_use", col_wrap=2, height=4, aspect=2,  sharey=False)

ax.map(plt.plot, 'month', 'meter_reading', color="teal", linewidth = 3)

plt.subplots_adjust(hspace=0.45)

plt.show()
bold('**UTILITIES AND HEALTHCARE HAVE THE HIGHEST READINGS**')

plt.rcParams['figure.figsize'] = (18, 15)

ax = sns.boxplot(data = train, y ='primary_use', x = 'meter_reading', color = 'teal', boxprops=dict(alpha=.3))

ax.set_xlabel('Log(Meter Reading)', fontsize = 20)

ax.set_ylabel('primary_use', fontsize = 20)

plt.show()
bold('**PLACES OF INDUSTRY HIGHEST READINGS ON WEEKDAYS**')

ax = sns.FacetGrid(train, col="primary_use", col_wrap=4, height=4, aspect=1,  sharex=False)

ax.map(sns.boxplot, 'meter_reading', 'weekday_name', color="teal",   boxprops=dict(alpha=.3))

plt.subplots_adjust(hspace=0.45)

plt.show()
bold('**READINGS REALLY PEAKED FROM MAY TO OCTOBER**')

plt.rcParams['figure.figsize'] = (18,10)

temp_df = train.groupby(['timestamp', 'month']).meter_reading.sum().reset_index()

ax = sns.lineplot(data = temp_df, x = 'timestamp', y = 'meter_reading', color = 'teal')

plt.xlabel('Timestamp', fontsize = 15)

plt.ylabel('Meter Reading')

plt.show()
bold('**MANUFACTURING REALLY BUCKED THE GENERAL TREND**')

temp_df = train.groupby(['timestamp', "primary_use"]).meter_reading.sum().reset_index()

ax = sns.FacetGrid(temp_df, col="primary_use", col_wrap=2, height=4, aspect=2,  sharey=False)

ax.map(sns.lineplot,'timestamp',  'meter_reading', color="teal")

plt.subplots_adjust(hspace=0.45)

plt.show()
plt.rcParams['figure.figsize'] = (18,10)

sns.heatmap(train.corr(), vmin=-1, vmax=1, center=0,

            square=True, cmap = sns.diverging_palette(20, 220, n=200))

plt.show()
'''Distribution of the Meter Reading'''

distplot(train['square_feet'], 'darkgreen')
'''Log tranformation of meter_reading'''

train['square_feet'] = np.log1p(train['square_feet'])

test['square_feet'] = np.log1p(test['square_feet'])



bold('**Distribution after log tranformation**')

distplot(train['square_feet'], 'darkgreen')
plt.rcParams['figure.figsize'] = (18,10)

temp_df = train.groupby('year_built').building_id.sum().reset_index()

ax = sns.lineplot(data = temp_df, x = 'year_built', y = 'building_id', color = 'black', linewidth = 3.5)

plt.xlabel('Year Built', fontsize = 15)

plt.ylabel('Building_ID', fontsize = 15)

plt.show()
'''Distribution of the Meter Reading'''

distplot(train['floor_count'].dropna(), 'darkred')
plt.rcParams['figure.figsize'] = (18,10)

sns.kdeplot(train['air_temperature'].dropna(), shade = True, color = 'gold')

plt.xlabel('Air Temperature', fontsize = 15)

plt.ylabel('Density', fontsize = 15)

plt.show()
plt.rcParams['figure.figsize'] = (18,10)

sns.kdeplot(train['dew_temperature'].dropna(), shade = True, color = 'indigo')

plt.xlabel('Dew Temperature', fontsize = 15)

plt.ylabel('Density', fontsize = 15)

plt.show()
plt.rcParams['figure.figsize'] = (18,10)

sns.kdeplot(train['wind_speed'].dropna(), shade = True, color = 'peru')

plt.xlabel('Wind Speed', fontsize = 15)

plt.ylabel('Density', fontsize = 15)

plt.show()
def speed_labels(bins:list, units:str) -> list:   

    labels = list()

    for left, right in zip(bins[:-1], bins[1:]):

        if left == bins[0]:

            labels.append('calm'.format(right))

        elif np.isinf(right):

            labels.append('>{} {}'.format(left, units))

        else:

            labels.append('{} - {} {}'.format(left, right, units))

    return labels



def _convert_dir(directions, N=None):

    if N is None:

        N = directions.shape[0]

    barDir = directions * np.pi/180. - np.pi/N

    barWidth = 2 * np.pi / N

    return barDir, barWidth



spd_bins = [-1, 0, 5, 10, 15, 20, 25, 30, np.inf]

spd_labels = speed_labels(spd_bins, units='m/s')



dir_bins = np.arange(-7.5, 370, 15)

dir_labels = (dir_bins[:-1] + dir_bins[1:]) / 2
calm_count = train[train['wind_speed'] == 0].shape[0]

total_count = len(train)

rose = (train.assign(WindSpd_bins=lambda df:

            pd.cut(df['wind_speed'], bins=spd_bins, labels=spd_labels, right=True)).assign(WindDir_bins=lambda df: pd.cut(df['wind_direction'], bins=dir_bins, labels=dir_labels, right=False)).replace({'WindDir_bins': {360: 0}}).groupby(by=['WindSpd_bins', 'WindDir_bins']).size().unstack(level='WindSpd_bins').fillna(0).assign(calm=lambda df: calm_count / df.shape[0]).sort_index(axis=1).applymap(lambda x: x / total_count * 100))

rose.drop(rose.index[0], inplace=True)

directions = np.arange(0, 360, 15)
def wind_rose(rosedata, wind_dirs, palette=None):

    if palette is None:

        palette = sns.color_palette('inferno', n_colors=rosedata.shape[1])



    bar_dir, bar_width = _convert_dir(wind_dirs)



    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.set_theta_direction('clockwise')

    ax.set_theta_zero_location('N')



    for n, (c1, c2) in enumerate(zip(rosedata.columns[:-1], rosedata.columns[1:])):

        if n == 0:

            # first column only

            ax.bar(bar_dir, rosedata[c1].values, 

                   width=bar_width,

                   color=palette[0],

                   edgecolor='none',

                   label=c1,

                   linewidth=0)

            # all other columns

        ax.bar(bar_dir, rosedata[c2].values, 

               width=bar_width, 

               bottom=rosedata.cumsum(axis=1)[c1].values,

               color=palette[n+1],

               edgecolor='none',

               label=c2,

               linewidth=0)



    leg = ax.legend(loc=(0.75, 0.95), ncol=2)

    xtl = ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    

    return fig

fig = wind_rose(rose, directions)
'''Imputing missing value of year build'''

train['year_built'] = np.uint8(train['year_built']-1900, inplace = True)

test['year_built'] = np.uint8(test['year_built']-1900, inplace = True)
'''The following variables are either discrete numerical or continuous numerical variables.So the will be imputed by median'''

to_impute_by_median = train.loc[:, ['floor_count','air_temperature', 'cloud_coverage', 'dew_temperature',

                      'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction','wind_speed']]

for i in to_impute_by_median.columns:

    train[i].fillna(train[i].median(), inplace = True)



to_impute_by_median = test.loc[:, ['floor_count','air_temperature', 'cloud_coverage', 'dew_temperature',

                      'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction','wind_speed']]

for i in to_impute_by_median.columns:

    test[i].fillna(test[i].median(), inplace = True)
'''Using sklearn's label encoder method'''

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

train['primary_use'] = le.fit_transform(train['primary_use'])

test['primary_use'] = le.fit_transform(test['primary_use'])
'''Now extract the nominal variables for one hot encoding of train and test data.'''

one_hot_train = pd.get_dummies(train['meter'])



one_hot_test = pd.get_dummies(test['meter'])
'''Droping variable'''

train.drop(columns=['meter', 'timestamp', 'weekday_name'], axis = 1, inplace = True)

test.drop(columns=['meter', 'timestamp'], axis = 1, inplace = True)
"""Let's concate one hot encoded, other variables together."""

train_processed = pd.concat([one_hot_train, train], axis = 1)

test_processed = pd.concat([one_hot_test, test], axis = 1)
"""Let's look at our final train and test data for modelling."""

bold('**Updated train data for modelling:**')

display(train_processed.head(3))

bold('**Updated test data for modelling:**')

display(test_processed.head(3))
'''Setting train, test and target for model'''

target = train_processed['meter_reading']

train = train_processed.drop(['meter_reading'], axis = 1)

test = test_processed.drop(['row_id'], axis = 1)
"""Let's have a final look at our data"""

bold('**Data Dimension for Model Building:**')

print('Input matrix dimension:', train.shape)

print('Output vector dimension:',target.shape)

print('Test data dimension:', test.shape)
cat_feat = ['ChilledWater', 'Electricity', 'HotWater', 'Steam',"site_id", "building_id", "primary_use", "hour", "weekday", "wind_direction"]
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold

from tqdm import tqdm



params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse'},

            'subsample': 0.25,

            'subsample_freq': 1,

            'learning_rate': 0.3,

            'num_leaves': 20,

            'feature_fraction': 0.9,

            'lambda_l1': 1,  

            'lambda_l2': 1

            }



folds = 4

seed = 55

kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

# oof_pred = np.zeros(train.shape[0])  # out of fold predictions

models = []



## stratify data by building_id

for train_index, val_index in tqdm(kf.split(train, train['building_id']), total=folds):

    train_X = train.iloc[train_index]

    val_X = train.iloc[val_index]

    train_y = target.iloc[train_index]

    val_y = target.iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=cat_feat)

    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=cat_feat)

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=(lgb_train, lgb_eval),

                early_stopping_rounds=100,

                verbose_eval = 100)

    models.append(gbm)
plt.rcParams['figure.figsize'] = (18,10)

lgb.plot_importance(models[0], importance_type='gain')

plt.show()
i=0

result=[]

step_size = 50000

for j in tqdm(range(int(np.ceil(test.shape[0]/50000)))):

    result.append(np.expm1(sum([model.predict(test.iloc[i:i+step_size]) for model in models])/folds))

    i+=step_size
'''Submission'''

result = np.concatenate(result)

submission = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")

submission["meter_reading"] = result

submission.to_csv("submission.csv", index = False)