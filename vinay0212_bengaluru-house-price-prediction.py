import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt #visualization

%matplotlib inline

import matplotlib

matplotlib.rcParams["figure.figsize"] = (20,15)
# additional imports:

import seaborn as sns

import re

import sys

from time import sleep

from tqdm.notebook import tqdm

import warnings;

warnings.filterwarnings('ignore');
df = pd.read_csv("../input/bengaluru-house-price-data/Bengaluru_House_Data.csv")
df.head()
df.shape
df.info()
df.head()
df.isnull().sum()
#Remove unnecessary columns 

df1 = df.drop(['area_type','society','balcony','availability'],axis='columns')

df1.head()
df1.shape
df1.isnull().sum()
df1 = df1.dropna()

df1.isnull().sum()
# lets check 'size' column

df1['size'].unique()
# FUNCTION to remove string from row values.

# Nan values will be replaced by 0

def remove_string(x):

    x = str(x)

    if x == 'nan':

        x = np.NaN

    else:

        x = int(x.split(" ")[0])

    return x
# We create new column for the cleaned values of size column:

df1['BHK'] = df1['size'].apply(lambda x: remove_string(x))
df1['BHK'].unique()
df1[df1.BHK > 20]
df1.isnull().sum()
df1.total_sqft.unique()
# Function to catch all non numeric and abnormal values:

def catch_abnormal_val(series):

    err_val = []

    for x in series:

        try:

            float(x)

        except:

            err_val.append(x)

    return err_val
catch_abnormal_val(df1['total_sqft'])
# Lets modify the range format values first:

# function that will identfy range format values and convert them to single float value:

def convert_rng_val(x):

    values = x.split('-')

    if len(values) == 2:

        return (float(values[0])+float(values[1]))/2 #return float mean value of range

    try:

        return float(x) #return remaining values in float.

    except:

        return x #return other abnormal value as it is.
print(convert_rng_val('1200'))

print(convert_rng_val('1200-2349'))

print(convert_rng_val('1200sqft. Meter'))
def sqmt_to_sqft(x):

    """convert sq.meters to sqft"""

    return x * 10.764



def sqyards_to_sqft(x):

    """convert sq.yards to sqft"""

    return x * 9



def gunta_to_sqft(x):

    """convert gunta to sqft"""

    return x * 1089



def acres_to_sqft(x):

    """convert acres to sqft"""

    return x * 43560



def perch_to_sqft(x):

    """convert perch to sqft"""

    return x * 272.25



def grounds_to_sqft(x):

    """convert grounds to sqft"""

    return x * 2400



def cents_to_sqft(x):

    """convert cents to sqft"""

    return x * 435.6
def clean_total_sqft(y):

    try:

        y = float(y)

    except:

        if "-" in y:

            y = round(convert_rng_val(y),1)

        elif "Sq. Meter" in y:

            y = round(sqmt_to_sqft(float(re.findall('\d+',y)[0])),1)

        elif "Sq. Yards" in y:

            y = sqyards_to_sqft(float(re.findall('\d+',y)[0]))

        elif "Guntha" in y:

            y = gunta_to_sqft(float(re.findall('\d+',y)[0]))

        elif "Acres" in y:

            y = acres_to_sqft(float(re.findall('\d+',y)[0]))

        elif "Perch" in y:

            y = perch_to_sqft(float(re.findall('\d+',y)[0]))

        elif "Grounds" in y:

            y = grounds_to_sqft(float(re.findall('\d+',y)[0])) 

        elif "Cents" in y:

            y = round(cents_to_sqft(float(re.findall('\d+',y)[0])),1)

        return y

    return y
clean_total_sqft("13Sq. Yards")
# Lets clean our column and create a cleaned version of it:

df1['total_sqft_cleaned'] = df1['total_sqft'].apply(lambda x : clean_total_sqft(x))

# lets check for abnormal values now :

catch_abnormal_val(df1['total_sqft_cleaned'])
# Remove unecessary columns:

df2 = df1.drop(['size','total_sqft'], axis=1)
df2.head()
df3 = df2.copy()
df3['location'].unique()
len(df3['location'].unique())
#Lets check number of classes in each categorical columns:

categorical_cols = df3.select_dtypes(include='object').columns

for col in categorical_cols:

    print(f'Number of classes in {col} : {df2[col].nunique()}')
# Creating new feature for detecting outliers:

df3['price_per_sqft'] = df3['price']*100000/df3['total_sqft_cleaned']

df3.head()
# Checking location statistics:

df3['location'] = df3['location'].apply(lambda x: x.strip())

location_stats = df3.groupby('location')['location'].agg('count').sort_values(ascending=False)

location_stats
# Locations with less than 10 count:

locations_stats_less_than_10 = location_stats[location_stats<=10]

locations_stats_less_than_10
df3['location'] = df3['location'].apply(lambda x : "other" if x in locations_stats_less_than_10 else x)
df3['location'].nunique()
# Per room sqft threshold be 300sqft: 

df3 = df3[~(df3.total_sqft_cleaned/df3.BHK < 300)]

df3.shape
df3['price_per_sqft'].describe()
# Function to remove outliers from price_per_sqft based on locations.

# As every location will have different price range.

def remove_price_outlier(df_in):

    df_out = pd.DataFrame()

    for key, subdf in df_in.groupby('location'):

        avg_price = np.mean(subdf.price_per_sqft)

        std_price = np.std(subdf.price_per_sqft)

        # data without outliers: 

        reduced_df = subdf[(subdf.price_per_sqft>(avg_price-std_price)) & (subdf.price_per_sqft<=(avg_price+std_price))]

        df_out =pd.concat([df_out, reduced_df], ignore_index=True)

    return df_out

df4 = remove_price_outlier(df3)

df4.shape
# Function to remove BHK outliers:

def remove_bhk_outliers(df_in):

    exclude_indices = np.array([])

    for location, location_subdf in df_in.groupby('location'):

        bhk_stats = {}

        for bhk, bhk_subdf in df_in.groupby('BHK'):

            bhk_stats[bhk] = {

                'mean':np.mean(bhk_subdf.price_per_sqft),

                'std':np.std(bhk_subdf.price_per_sqft),

                'count':bhk_subdf.shape[0]

            }

        for bhk, bhk_subdf in location_subdf.groupby('BHK'):

            stats = bhk_stats.get(bhk-1) #statistics of n-1 BHK

            if stats and stats['count'] > 5:

                exclude_indices = np.append(exclude_indices, bhk_subdf[bhk_subdf.price_per_sqft<(stats['mean'])].index.values)

    return df_in.drop(exclude_indices, axis='index')

        

df5 = remove_bhk_outliers(df4)

df5.shape
# Visualize to see number of data points for price_per_sqft

plt.hist(df5.price_per_sqft, rwidth=0.8)

plt.xlabel("Price Per Sqft.")

plt.ylabel("Count")
df5[df5.bath>10]
# Visualize to see data points based on number of bathrooms:

plt.hist(df5.bath, rwidth=0.8)

plt.xlabel("Number of Bathrooms")

plt.ylabel("Count")
df5[df5.bath > df5.BHK+2]
# Remove bathroom outliers:

df6 = df5[df5.bath<df5.BHK+2]

df6.shape
df6.head()
df7 = df6.drop(['price_per_sqft'], axis=1)

df7.head()
location_dummies = pd.get_dummies(df7.location)

location_dummies.head()
df8 = pd.concat([df7, location_dummies.drop('other', axis='columns')], axis='columns')

df8.head()
# Remove Location Column:

df9 = df8.drop(['location'], axis='columns')

df9.head()
df9.shape
# Independent variables:

X = df9.drop('price', axis='columns')

X.head()
# Dependent Variable:

y = df9['price']

y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
# Linear Regression: 

from  sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

print(f'Score: {lin_reg.score(X_test, y_test)}')
# K-fold validation for Linear Regression:

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

cv1 = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv1)
from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import ElasticNet
def find_best_model_grid_search(X, Y, tqdm=tqdm):

    algos = {

        'Linear_regression' : {

            'model' : LinearRegression(),

            'params': {

                'normalize':[True, False]

             }

          },  

         'Lasso' : {

             'model': Lasso(),

             'params': {

                  "max_iter": [1, 5, 10],

                 'alpha': [0.02, 0.024, 0.025, 0.026, 0.03, 0.05, 0.5, 1,2],

                 'selection':['random', 'cyclic'],

                  'normalize':[True, False]

             }

          },

         'Ridge' : {

             'model' : Ridge(),

             'params': {

                  "max_iter": [1, 5, 10],

                 'alpha': [0.05, 0.1, 0.5, 1, 5, 10, 200, 230, 250,265, 270, 275, 290, 300, 500],

                  'normalize':[True, False]

             }

         },

        'ElasticNet' : {

             'model' : ElasticNet(),

             'params' : {

                 "max_iter": [1, 5, 10],

                 'alpha': [0, 0.01, 0.02, 0.03, 0.05, 0.5, 1, 0.05, 0.1, 0.5, 1, 5, 10, 100],

                 'l1_ratio': np.arange(0.0, 1.0, 0.1),

                 'normalize':[True, False]

             } 

         },

          'Decision_tree': {

              'model': DecisionTreeRegressor(),

              'params': {

                  'criterion' : ['mse', 'friedman_mse'],

                  'splitter': ['best', 'random']

              }

          }

    }

    values = (algos.items())

    scores = []

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    print(f'Grid Search CV Initiated..' )    

    with tqdm(total=len(values), file=sys.stdout) as pbar:

        for algo_name, config in algos.items():

            pbar.set_description('Processed')

            gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)

            gs.fit(X,Y)

            scores.append({

                'model': algo_name,

                'best_score': gs.best_score_,

                'best_params': gs.best_params_

            })

            pbar.update(1)

            print(f'Grid search CV for {algo_name} done')

        print("Grid Search CV completed!")

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])
models = find_best_model_grid_search(X, y)

models
# Ridge best parameters:

models.loc[3]['best_params']
# Re-train using best parameter:

model = Ridge(alpha=0.1, max_iter=1)

model.fit(X_train, y_train)
# Prediction:

ypred = model.predict(X_test)
# Visualising the test vs predicted data:

plt.scatter(ypred, y_test)

plt.title('Actual Price vs Predicted Price (in Lacs)')

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')
# Calculate the absolute errors

errors = abs(ypred - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
X.columns
pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
# Prediction Function

def predict_price(location, sqft, bath, bhk, data=X):

    loc_index = np.where(data.columns==location)[0][0]

    x = np.zeros(len(data.columns)) #init a new array with zero values.

    x[0] = bath

    x[1] = bhk

    x[2] = sqft

    if loc_index >= 0:

        x[loc_index] = 1

    return model.predict([x])[0]
predict_price('1st Phase JP Nagar',1000,2,2)
# Indira Nagar is most expensive in Bengaluru. Lets predict

predict_price('Indira Nagar',1000,2,2)
# saving ml model as pickle file:

import pickle

with open('bengaluru_home_price_model.pickle','wb') as f:

    pickle.dump(model,f)
# saving column names as a json file:

import json

columns = {

    'data_columns' : [col.lower() for col in X.columns]

}

with open('columns.json', 'w') as f:

    f.write(json.dumps(columns))