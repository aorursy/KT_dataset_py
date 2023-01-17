# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd

import datetime as dt

from sklearn.model_selection import train_test_split

import xgboost as xgb

import os
train_df =  pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows = 6_000_000) #1M to test models

train_df.head()
#Clean dataset

def clean_df(df):

    return df[(df.fare_amount > 0) & 

            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &

            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &

            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &

            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) &

            (df.passenger_count > 0) & (df.passenger_count < 10)]



train_df = clean_df(train_df)

print(len(train_df))
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):

    



    R_earth = 6371



    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,

                                                             [pickup_lat, pickup_lon, 

                                                              dropoff_lat, dropoff_lon])

    #Compute distances along lat, lon dimensions

    dlat = dropoff_lat - pickup_lat

    dlon = dropoff_lon - pickup_lon

    

    #Compute haversine distance

    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2

    

    return 2 * R_earth * np.arcsin(np.sqrt(a))



def add_airport_dist(dataset):



    jfk_coord = (40.639722, -73.778889)

    ewr_coord = (40.6925, -74.168611)

    lga_coord = (40.77725, -73.872611)

    

    pickup_lat = dataset['pickup_latitude']

    dropoff_lat = dataset['dropoff_latitude']

    pickup_lon = dataset['pickup_longitude']

    dropoff_lon = dataset['dropoff_longitude']

    

    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 

    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 

    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])

    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 

    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 

    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon) 

    

    dataset['jfk_dist'] = pd.concat([pickup_jfk, dropoff_jfk], axis=1).min(axis=1)

    dataset['ewr_dist'] = pd.concat([pickup_ewr, dropoff_ewr], axis=1).min(axis=1)

    dataset['lga_dist'] = pd.concat([pickup_lga, dropoff_lga], axis=1).min(axis=1)

    

    return dataset

    

def add_datetime_info(dataset):

    #Convert to datetime format

    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")

    

    dataset['hour'] = dataset.pickup_datetime.dt.hour

    dataset['day'] = dataset.pickup_datetime.dt.day

    dataset['month'] = dataset.pickup_datetime.dt.month

    dataset['weekday'] = dataset.pickup_datetime.dt.weekday

    dataset['year'] = dataset.pickup_datetime.dt.year

    

    return dataset



train_df = add_datetime_info(train_df)

train_df = add_airport_dist(train_df)

train_df['distance'] = sphere_dist(train_df['pickup_latitude'], train_df['pickup_longitude'], 

                                   train_df['dropoff_latitude'] , train_df['dropoff_longitude'])



train_df.head()

train_df.drop(columns=['key', 'pickup_datetime'], inplace=True)

train_df.head()
y = train_df['fare_amount']

train = train_df.drop(columns=['fare_amount'])



x_train,x_test,y_train,y_test = train_test_split(train,y,random_state=0,test_size=0.01)
#Cross-validation

params = {

    # Parameters that we are going to tune.

    'max_depth': 8, #Result of tuning with CV

    'eta':.03, #Result of tuning with CV

    'subsample': 1, #Result of tuning with CV

    'colsample_bytree': 0.8, #Result of tuning with CV

    # Other parameters

    'objective':'reg:linear',

    'eval_metric':'rmse',

    'silent': 1

}



#Block of code used for hypertuning parameters. Adapt to each round of parameter tuning.

#Turn off CV in submission

CV=False

if CV:

    dtrain = xgb.DMatrix(train,label=y)

    gridsearch_params = [

        (eta)

        for eta in np.arange(.04, 0.12, .02)

    ]



    # Define initial best params and RMSE

    min_rmse = float("Inf")

    best_params = None

    for (eta) in gridsearch_params:

        print("CV with eta={} ".format(

                                 eta))



        # Update our parameters

        params['eta'] = eta



        # Run CV

        cv_results = xgb.cv(

            params,

            dtrain,

            num_boost_round=1000,

            nfold=3,

            metrics={'rmse'},

            early_stopping_rounds=10

        )



        # Update best RMSE

        mean_rmse = cv_results['test-rmse-mean'].min()

        boost_rounds = cv_results['test-rmse-mean'].argmin()

        #print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))

        if mean_rmse < min_rmse:

            min_rmse = mean_rmse

            best_params = (eta)



    print("Best params: {}, RMSE: {}".format(best_params, min_rmse))

else:

    #Print final params to use for the model

    params['silent'] = 0 #Turn on output

    print(params)

    

    

    

def XGBmodel(x_train,x_test,y_train,y_test,params):

    matrix_train = xgb.DMatrix(x_train,label=y_train)

    matrix_test = xgb.DMatrix(x_test,label=y_test)

    model=xgb.train(params=params,

                    dtrain=matrix_train,num_boost_round=5000, 

                    early_stopping_rounds=10,evals=[(matrix_test,'test')])

    return model



model = XGBmodel(x_train,x_test,y_train,y_test,params)





#Read and preprocess test set

test_df =  pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')

test_df = add_datetime_info(test_df)

test_df = add_airport_dist(test_df)

test_df['distance'] = sphere_dist(test_df['pickup_latitude'], test_df['pickup_longitude'], 

                                   test_df['dropoff_latitude'] , test_df['dropoff_longitude'])



test_key = test_df['key']

x_pred = test_df.drop(columns=['key', 'pickup_datetime'])



prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)


submission = pd.DataFrame({

        "key": test_key,

        "fare_amount": prediction.round(2)

})



submission.to_csv('taxi_fare_submission.csv',index=False)

submission.head()