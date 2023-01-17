'''Ignore deprecation and future, and user warnings.'''

import warnings as wrn

wrn.filterwarnings('ignore', category = DeprecationWarning) 

wrn.filterwarnings('ignore', category = FutureWarning) 

wrn.filterwarnings('ignore', category = UserWarning) 



'''Import basic modules.'''

import pandas as pd

import numpy as np

from scipy import stats



'''Customize visualization

Seaborn and matplotlib visualization.'''

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline



'''Plotly visualization .'''

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook



'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))
'''Read in export and import data from CSV file'''

df_train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv')

df_test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv')
'''Train and Test data at a glance.'''

bold('**Preview of Train Data:**')

display(df_train.head())

bold('**Preview of Test Data:**')

display(df_test.head())
'''Dimension of train and test data'''

bold('**Shape of our train and test data**')

print('Dimension of train:',df_train.shape) 

print('Dimension of test:',df_test.shape)
'''Funtion for Variable Description'''

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

display(description(df_train))

bold('**Variable Description of  test Data:**')

display(description(df_test))
'''Visulization of IntersectionID'''

plt.figure(figsize=(15,6))

df_train.IntersectionId.value_counts()[:50].plot(kind='bar', color = 'teal')

plt.xlabel("Intersection Number", fontsize=18)

plt.ylabel("Count", fontsize=18)

plt.title("TOP 50 most commmon IntersectionID's ", fontsize=22)

plt.show()
'''Visulization of Entry/Exit StreetNames'''

plt.figure(figsize=(15,6))

df_train['EntryStreetName'].value_counts()[:50].plot(kind='bar', color = 'darkred')

plt.xlabel("Entry Street Names", fontsize=18)

plt.ylabel("Count", fontsize=18)

plt.title("TOP 50 most Entry Street Names ", fontsize=22)

plt.show()



plt.figure(figsize=(15,6))

df_train['ExitStreetName'].value_counts()[:50].plot(kind='bar', color = 'darkgreen')

plt.xlabel("Exit Street Names", fontsize=18)

plt.ylabel("Count", fontsize=18)

plt.title("TOP 50 most Exit Street Names ", fontsize=22)

plt.show()
'''Visulization of Path'''

plt.figure(figsize=(15,6))

df_train.Path.value_counts()[:50].plot(kind='bar', color = 'teal')

plt.xlabel("Path", fontsize=18)

plt.ylabel("Count", fontsize=18)

plt.title("TOP 50 most commmon Paths", fontsize=22)

plt.show()
sns.set_style("dark")

fig, ax = plt.subplots(2,1, figsize=[15, 12])



sns.countplot(data = df_train, x = 'EntryHeading', ax = ax[0], palette = 'YlOrRd_r')

ax[0].set_title('Count plot of Entry Heading', fontsize = 22)

ax[0].set_xlabel('Entry Heady', fontsize = 18)



sns.countplot(data = df_train, x = 'ExitHeading', ax = ax[1], palette = 'YlGnBu')

ax[1].set_title('Count plot of Exit Heading', fontsize = 22)

ax[1].set_xlabel('Exit Heady', fontsize = 18)



plt.subplots_adjust(hspace = 0.3)

plt.show()
total = len(df_train)

plt.figure(figsize=(15,6))

sns.set_style("white")



ax = sns.countplot(x = "City", data = df_train, palette = 'Dark2')

ax.set_title("City Count Distribution", fontsize=20)

ax.set_ylabel("Count",fontsize= 17)

ax.set_xlabel("City Names", fontsize=17)

sizes=[]

for p in ax.patches:

    height = p.get_height()

    sizes.append(height)

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=14) 

ax.set_ylim(0, max(sizes) * 1.15)



plt.show()
fig, ax = plt.subplots(2,1, figsize=[15, 12])



sns.countplot(data = df_train[df_train['Weekend']==0], x = 'Hour', hue = 'City', ax = ax[0], palette = 'Dark2')

ax[0].legend()

ax[0].set_title('Count Distribution of Hour by Week days ', fontsize = 22)

ax[0].set_xlabel('Hour', fontsize = 18)



sns.countplot(data = df_train[df_train['Weekend']==1], x = 'Hour', hue = 'City', ax = ax[1], palette = 'Dark2')

ax[1].legend()

ax[1].set_title('Count Distribution of Hour by Weekend days', fontsize = 22)

ax[1].set_xlabel('Hour', fontsize = 18)



plt.subplots_adjust(hspace = 0.3)

plt.show()
plt.figure(figsize=(15,6))

ax = sns.countplot(x = "Month", data = df_train, hue = 'City',palette = 'Dark2')

ax.set_title("Month Distribution by cities", fontsize=20)

ax.set_ylabel("Count",fontsize= 17)

ax.set_xlabel("Month", fontsize=17)

plt.show()
'''Visualition of Map Plot of Atlanta city'''

import mplleaflet

plt.figure(figsize=(10,10))

map1 = df_train[df_train['City']=='Atlanta'].groupby(['Latitude', 'Longitude'])['RowId'].count().reset_index()

plt.scatter(map1['Longitude'], map1['Latitude'], alpha=0.5)



mplleaflet.display()
'''Visualition of Map Plot of Boston city'''

plt.figure(figsize=(10,10))

map2 = df_train[df_train['City']=='Boston'].groupby(['Latitude', 'Longitude'])['RowId'].count().reset_index()

plt.scatter(map2['Longitude'], map2['Latitude'], alpha=0.5)



mplleaflet.display()
'''Visualition of Map Plot of Philadelphia city'''

plt.figure(figsize=(10,10))

map3 = df_train[df_train['City']=='Philadelphia'].groupby(['Latitude', 'Longitude'])['RowId'].count().reset_index()

plt.scatter(map3['Longitude'], map3['Latitude'], alpha=0.5)



mplleaflet.display()
'''Visualition of Map Plot of Chicago city'''

plt.figure(figsize=[10, 10])

map4 = df_train[df_train['City']=='Chicago'].groupby(['Latitude', 'Longitude'])['RowId'].count().reset_index()

sns.kdeplot(map4['Longitude'], map4['Latitude'])



mplleaflet.display()
fig, ax = plt.subplots(nrows=2, ncols=2)

sns.set_style("whitegrid")



df_train[df_train['City']=='Atlanta'].groupby('Hour')['TotalTimeStopped_p80'].mean().plot(

    ax=ax[0,0],title="Atlanda's Total Stoppage Time in Hours", color='r', figsize=(18,15))



df_train[df_train['City']=='Boston'].groupby('Hour')['TotalTimeStopped_p80'].mean().plot(

    ax=ax[0,1],title="Boston's Total Stoppage Time in Hours", color='r', figsize=(18,15))





df_train[df_train['City']=='Chicago'].groupby('Hour')['TotalTimeStopped_p80'].mean().plot(

    ax=ax[1,0],title="Chicago's Total Stoppage Time in Hours", color='r', figsize=(18,15))





df_train[df_train['City']=='Philadelphia'].groupby('Hour')['TotalTimeStopped_p80'].mean().plot(

    ax=ax[1,1],title="Philadelphia's Total Stoppage Time in Hours", color='r', figsize=(18,15))



plt.show()
df_train['Intersection'] = df_train['IntersectionId'].astype(str) + df_train['City']

df_test['Intersection'] = df_test['IntersectionId'].astype(str) + df_test['City']

print(df_train['Intersection'].sample(6).values)
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()

le.fit(pd.concat([df_train['Intersection'], df_test['Intersection']]).drop_duplicates().values)

df_train['Intersection'] = le.transform(df_train['Intersection'])

df_test['Intersection'] = le.transform(df_test['Intersection'])

print(df_train['Intersection'].sample(6).values)
# Reference: https://www.kaggle.com/bgmello/how-one-percentile-affect-the-others



'''Let's use the following road types: Street, Avenue, Road, Boulevard, Broad and Drive'''

road_encoding = {

    'Road': 1,

    'Street': 2,

    'Avenue': 2,

    'Drive': 3,

    'Broad': 3,

    'Boulevard': 4

}
def encode(x):

    if pd.isna(x):

        return 0

    for road in road_encoding.keys():

        if road in x:

            return road_encoding[road]

        

    return 0
df_train['EntryTypeStreet'] = df_train['EntryStreetName'].apply(encode)

df_train['ExitTypeStreet'] = df_train['ExitStreetName'].apply(encode)

df_test['EntryTypeStreet'] = df_test['EntryStreetName'].apply(encode)

df_test['ExitTypeStreent'] = df_test['ExitStreetName'].apply(encode)

print(df_train['EntryTypeStreet'].sample(10).values)
df_train["same_street_exact"] = (df_train["EntryStreetName"] ==  df_train["ExitStreetName"]).astype(int)

df_test["same_street_exact"] = (df_test["EntryStreetName"] ==  df_test["ExitStreetName"]).astype(int)
'''Defineing the directions'''

directions = {

    'N': 0,

    'NE': 1/4,

    'E': 1/2,

    'SE': 3/4,

    'S': 1,

    'SW': 5/4,

    'W': 3/2,

    'NW': 7/4

}
df_train['EntryHeading'] = df_train['EntryHeading'].map(directions)

df_train['ExitHeading'] = df_train['ExitHeading'].map(directions)



df_test['EntryHeading'] = df_test['EntryHeading'].map(directions)

df_test['ExitHeading'] = df_test['ExitHeading'].map(directions)



df_train['diffHeading'] = df_train['EntryHeading']- df_train['ExitHeading']  

df_test['diffHeading'] = df_test['EntryHeading']- df_test['ExitHeading']



display(df_train[['ExitHeading','EntryHeading','diffHeading']].drop_duplicates().head(5))
#def cyclical_encode(data, col, max_val):

#    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)

#    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)

#    return data



#df_train = cyclical_encode(df_train, 'Hour', 24)

#df_test = cyclical_encode(df_test, 'Hour', 24) 
#'''One Hot Ecoding of month feature'''

#df_train = pd.concat([df_train, pd.get_dummies(df_train['Month'], prefix='Month', drop_first = False)], axis=1)

#df_test = pd.concat([df_test, pd.get_dummies(df_test['Month'], prefix='Month', drop_first = False)], axis=1)
'''One Hot Ecoding of City feature'''

df_train = pd.concat([df_train, pd.get_dummies(df_train['City'], drop_first = False)], axis=1)

df_test = pd.concat([df_test, pd.get_dummies(df_test['City'], drop_first = False)], axis=1)
"""Adding temperature (°F) of each city by month"""

# Reference: https://www.kaggle.com/dcaichara/feature-engineering-and-lightgbm

monthly_avg = {'Atlanta1': 43.0, 'Atlanta5': 68.5, 'Atlanta6': 76.0, 'Atlanta7': 78.0, 'Atlanta8': 78.0, 'Atlanta9': 72.5,

              'Atlanta10': 62.0, 'Atlanta11': 52.5, 'Atlanta12': 45.0, 'Boston1': 29.5, 'Boston5': 58.5, 'Boston6': 68.0,

              'Boston7': 74.0, 'Boston8': 73.0, 'Boston9': 65.5, 'Boston10': 54.5,'Boston11': 45.0, 'Boston12': 35.0,

              'Chicago1': 27.0, 'Chicago5': 59.5, 'Chicago6': 70.0, 'Chicago7': 76.0, 'Chicago8': 75.5, 'Chicago9': 68.0,

              'Chicago10': 56.0,  'Chicago11': 44.5, 'Chicago12': 32.0, 'Philadelphia1': 34.5, 'Philadelphia5': 66.0,

              'Philadelphia6': 75.5, 'Philadelphia7': 80.5, 'Philadelphia8': 78.5, 'Philadelphia9': 71.5, 'Philadelphia10': 59.5,

              'Philadelphia11': 49.0, 'Philadelphia12': 40.0}

# Concatenating the city and month into one variable

df_train['city_month'] = df_train["City"] + df_train["Month"].astype(str)

df_test['city_month'] = df_test["City"] + df_test["Month"].astype(str)



# Creating a new column by mapping the city_month variable to it's corresponding average monthly temperature

df_train["average_temp"] = df_train['city_month'].map(monthly_avg)

df_test["average_temp"] = df_test['city_month'].map(monthly_avg)
"""Adding rainfall (inches) of each city by month"""

monthly_rainfall = {'Atlanta1': 5.02, 'Atlanta5': 3.95, 'Atlanta6': 3.63, 'Atlanta7': 5.12, 'Atlanta8': 3.67, 'Atlanta9': 4.09,

              'Atlanta10': 3.11, 'Atlanta11': 4.10, 'Atlanta12': 3.82, 'Boston1': 3.92, 'Boston5': 3.24, 'Boston6': 3.22,

              'Boston7': 3.06, 'Boston8': 3.37, 'Boston9': 3.47, 'Boston10': 3.79,'Boston11': 3.98, 'Boston12': 3.73,

              'Chicago1': 1.75, 'Chicago5': 3.38, 'Chicago6': 3.63, 'Chicago7': 3.51, 'Chicago8': 4.62, 'Chicago9': 3.27,

              'Chicago10': 2.71,  'Chicago11': 3.01, 'Chicago12': 2.43, 'Philadelphia1': 3.52, 'Philadelphia5': 3.88,

              'Philadelphia6': 3.29, 'Philadelphia7': 4.39, 'Philadelphia8': 3.82, 'Philadelphia9':3.88 , 'Philadelphia10': 2.75,

              'Philadelphia11': 3.16, 'Philadelphia12': 3.31}

# Creating a new column by mapping the city_month variable to it's corresponding average monthly rainfall

df_train["average_rainfall"] = df_train['city_month'].map(monthly_rainfall)

df_test["average_rainfall"] = df_test['city_month'].map(monthly_rainfall)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

lat_long = ['Latitude', 'Longitude']

for col in lat_long:

    df_train[col] = (scaler.fit_transform(df_train[col].values.reshape(-1, 1)))

    df_test[col] = (scaler.fit_transform(df_test[col].values.reshape(-1, 1)))
"""Let's see the columns of data"""

df_train.columns.values
"""Let's drop the unwanted variables from test and train dataset"""

df_train.drop(['RowId', 'IntersectionId', 'EntryStreetName', 'ExitStreetName', 'Path', 'city_month', 'City'], axis=1, inplace=True)

df_test.drop(['RowId', 'IntersectionId', 'EntryStreetName', 'ExitStreetName', 'Path', 'city_month', 'City'], axis=1, inplace=True)
"""Let’s make a correlation matrix heatmap for the data set."""

plt.figure(figsize=(18,14))

sns.heatmap(df_train.corr(),vmin=-1, vmax=1, center=0,

            square=True, cmap = sns.diverging_palette(20, 220, n=200))

plt.show()
"""Let's look at our final train and test data for modelling."""

bold('**Updated train data for modelling:**')

display(df_train.head(3))

bold('**Updated test data for modelling:**')

display(df_test.head(3))
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
df_train = reduce_mem_usage(df_train)

df_test = reduce_mem_usage(df_test)
'''Seting X and Y'''

target_var = df_train.iloc[:, 7:22]

X_train = df_train.drop(target_var,axis = 1)



y1_train = df_train["TotalTimeStopped_p20"]

y2_train = df_train["TotalTimeStopped_p50"]

y3_train = df_train["TotalTimeStopped_p80"]

y4_train = df_train["DistanceToFirstStop_p20"]

y5_train = df_train["DistanceToFirstStop_p50"]

y6_train = df_train["DistanceToFirstStop_p80"]



X_test = df_test
"""Let's have a final look at our data"""

bold('**Data Dimension for Model Building:**')

print('Input matrix dimension:', X_train.shape)

print('Output vector dimension:',y1_train.shape)

print('Test data dimension:', X_test.shape)
description(X_train)
"""pecifying categorical features"""

cat_feat = ['Hour', 'Weekend','Month', 'same_street_exact', 'Intersection',

       'Atlanta', 'Boston', 'Chicago', 'Philadelphia', 'EntryTypeStreet', 'ExitTypeStreet']
all_preds ={0:[],1:[],2:[],3:[],4:[],5:[]}

all_target = [y1_train, y2_train, y3_train, y4_train, y5_train, y6_train]
# Reference: 

# https://medium.com/analytics-vidhya/hyperparameters-optimization-for-lightgbm-catboost-and-xgboost-regressors-using-bayesian-6e7c495947a9

# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#for-faster-speed



'''Importing Libraries'''

import lightgbm as lgb

from bayes_opt import BayesianOptimization



dtrain = lgb.Dataset(data=X_train, label=y1_train)



'''Define objective function'''

def hyp_lgbm(num_leaves, feature_fraction, bagging_fraction, max_depth, min_split_gain, min_child_weight, lambda_l1, lambda_l2):

      

        params = {'application':'regression','num_iterations': 400,

                  'learning_rate':0.01,

                  'metric':'rmse'} # Default parameters

        params["num_leaves"] = int(round(num_leaves))

        params['feature_fraction'] = max(min(feature_fraction, 1), 0)

        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

        params['max_depth'] = int(round(max_depth))

        params['min_split_gain'] = min_split_gain

        params['min_child_weight'] = min_child_weight

        params['lambda_l1'] = lambda_l1

        params['lambda_l2'] = lambda_l2

        

        cv_results = lgb.cv(params, dtrain, nfold=5, seed=44, categorical_feature=cat_feat, stratified=False,

                            verbose_eval =None)

#         print(cv_results)

        return -np.min(cv_results['rmse-mean'])
''' Define search space of hyperparameters'''

pds = {'num_leaves': (100, 230),

          'feature_fraction': (0.1, 0.5),

          'bagging_fraction': (0.8, 1),

          'lambda_l1': (0,3),

          'lambda_l2': (0,5),

          'max_depth': (8, 19),

          'min_split_gain': (0.001, 0.1),

          'min_child_weight': (1, 20)

        }
'''Define a surrogate model of the objective function and call it.'''

optimizer = BayesianOptimization(hyp_lgbm,pds,random_state=44)

                                  

# Optimize

optimizer.maximize(init_points=5, n_iter=12)
'''Best parameters after optimization'''

optimizer.max
p = optimizer.max['params']

param = {'num_leaves': int(round(p['num_leaves'])),

         'feature_fraction': p['feature_fraction'],

         'bagging_fraction': p['bagging_fraction'],

         'max_depth': int(round(p['max_depth'])),

         'lambda_l1': p['lambda_l1'],

         'lambda_l2':p['lambda_l2'],

         'min_split_gain': p['min_split_gain'],

         'min_child_weight': p['min_child_weight'],

         'learing_rate':0.05,

         'objective': 'regression',

         'boosting_type': 'gbdt',

         'verbose': 1,

         'seed': 44,

         'metric': 'rmse'

        }

param
'''Instantiate the models with optimized hyperparameters.'''

train = X_train 

test = X_test 

from sklearn.model_selection import train_test_split



for i in range(len(all_preds)):

    print('Training and predicting for target {}'.format(i+1))

    X_train,X_test,y_train,y_test=train_test_split(train,all_target[i], test_size=0.2, random_state=31)

    xg_train = lgb.Dataset(X_train,

                           label = y_train

                           )

    xg_valid = lgb.Dataset(X_test,

                           label = y_test

                           )

    clf = lgb.train(param, xg_train, 10000, valid_sets = [xg_valid],categorical_feature=cat_feat,

                         verbose_eval=100, early_stopping_rounds = 200)

    all_preds[i] = clf.predict(test, num_iteration=clf.best_iteration)
submission = pd.read_csv('../input/bigquery-geotab-intersection-congestion/sample_submission.csv')

#submission.head()
dt = pd.DataFrame(all_preds).stack()

dt = pd.DataFrame(dt)

submission['Target'] = dt[0].values
submission.head()
submission.to_csv('lgbm2_submission.csv', index=False)