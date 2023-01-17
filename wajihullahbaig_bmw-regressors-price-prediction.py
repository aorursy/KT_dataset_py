# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pprint import pprint

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/bmw_pricing_challenge.csv', delimiter=',')
# Check for any nulls

print(df.isnull().sum())

# Show data types

print(df.dtypes)
#Lets drop unwanted columns

df.drop(["maker_key","sold_at"], axis=1, inplace=True)

#Convert string/text to categorical values

car_models = df.model_key.copy()

model_labels = df['model_key'].astype('category').cat.categories.tolist()

model_labels_dict = {k: v for k,v in zip(model_labels,list(range(1,len(model_labels)+1)))}

fuel_labels = df['fuel'].astype('category').cat.categories.tolist()

fuel_labels_dict = {k: v for k,v in zip(fuel_labels,list(range(1,len(fuel_labels)+1)))}

paint_labels = df['paint_color'].astype('category').cat.categories.tolist()

paint_labels_dict =  {k: v for k,v in zip(paint_labels,list(range(1,len(paint_labels)+1)))}

type_labels = df['car_type'].astype('category').cat.categories.tolist()

type_labels_dict =  {k: v for k,v in zip(type_labels,list(range(1,len(type_labels)+1)))}





df.replace(model_labels_dict, inplace=True)

df.replace(fuel_labels_dict, inplace=True)

df.replace(paint_labels_dict, inplace=True)

df.replace(type_labels_dict, inplace=True)



df['model_key'] = df['model_key'].astype('category')



#Convert registration_date to integer

df['registration_date'] = df['registration_date'].str.replace("-","").astype(int)

print(df.dtypes)

# Data visualizations/Insights

import seaborn as sns

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,15))

ax = fig.gca()

c = car_models.value_counts()

c.sort_values(ascending=False).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=1)

plt.xlabel('Models',fontsize=10)

plt.ylabel('Counts',fontsize=10)

ax.tick_params(labelsize=10)

plt.title('BMW car models',fontsize=10)

plt.grid()

plt.ioff()

plt.show()



fig = plt.figure(figsize = (10,10))

ax = fig.gca()

sns.heatmap(df.corr(), annot=True, fmt=".2f")

plt.title("Correlation",fontsize=5)

plt.show()



fig = plt.figure(figsize = (10,15))

ax = fig.gca()

df.plot(ax=ax,kind='density',subplots=True,sharex=False)

plt.title("Density",fontsize=5)

plt.show()



# =============================================================================

# fig = plt.figure(figsize = (20,20))

# ax = fig.gca()

# sns.pairplot(data=df[0:100],hue="price") # pair plot a subset. Takes too long for the whole data

# plt.title("Pair plot",fontsize =10)

# plt.show()

# 

# =============================================================================
def feature_importance_plots(regressor,X_train):

        feature_importances = pd.DataFrame(regressor.feature_importances_,index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)

        feature_importances.plot(kind='bar')

        plt.show()



def do_prediction(df, stratify):

    price = df.price.copy()

    df.drop(['price'],inplace=True,axis = 1)

    

    if stratify and 'model_key' in df:

         X_train, X_test, y_train, y_test = train_test_split(df,price, test_size=0.25, stratify=df['model_key'], random_state=5811) 

    else:

        X_train, X_test, y_train, y_test = train_test_split(df, price, test_size=0.25, random_state=5811)

    

    

    gbr = GradientBoostingRegressor(loss ='ls',n_estimators=150, max_depth=7,max_leaf_nodes = 9,random_state=5811)

    # Look at the parameters

    print('Parameters currently in use:\n')

    pprint(gbr.get_params())

    # Fit the training data

    gbr.fit (X_train, y_train)

    # get the predicted values from the test set

    

    predicted_price = gbr.predict(X_test)

    

    print('GBR R squared: %.4f' % gbr.score(X_test, y_test))

    lin_mse = mean_squared_error(predicted_price, y_test)

    lin_rmse = np.sqrt(lin_mse)

    print('RMSE: %.4f' % lin_rmse)

    feature_importance_plots(gbr,X_train)

    

    

    forest_reg = RandomForestRegressor(n_estimators=150,min_samples_split=3,random_state=5811)

    # Look at the parameters

    print('Parameters currently in use:\n')

    pprint(forest_reg.get_params())

    forest_reg.fit(X_train, y_train)

    predicted_price = forest_reg.predict(X_test)

    

    print('RFR R squared: %.4f' % forest_reg.score(X_test, y_test))

    lin_mse = mean_squared_error(predicted_price, y_test)

    lin_rmse = np.sqrt(lin_mse)

    print('RMSE: %.4f' % lin_rmse)

    feature_importance_plots(forest_reg,X_train)
df_copy = df.copy()

do_prediction(df.copy(),False)
def replicate_low_count(df_copy):

    # What if between test/train splits we dont have the models of cars available?

    # For that we find the bear minimum count and replicate the data. 

    min_counted = c <= 1

    # Lets populate the data with replicas for the car models

    

    print("DataFrame size before append:",len(df_copy))

    for item in min_counted.iteritems():

        if item[1]:

            v = model_labels_dict[item[0]]

            v2 = df_copy[df_copy['model_key'] == v].copy()

            if len(v2) > 1:# Keep one copy- this happens when min_counted > 1

                mean_price = v2["price"].mean()

                v2 = v2.iloc[[0]].copy()

                v2["price"] = mean_price

            for copy_count in range(0,2):

                df_copy = df_copy.append(v2,ignore_index=True)

    

    print("DataFrame after append:",len(df_copy))

    return  df_copy
df_copy=replicate_low_count(df_copy)

do_prediction(df_copy,True)
# Now lets use the 8 important features as described in the description file of the dataset

df_copy = df.copy()

df.drop(["engine_power",'mileage','paint_color',"registration_date","model_key","car_type","fuel"],inplace=True,axis=1)



do_prediction(df,False)

df_copy = replicate_low_count(df_copy)

df_copy.drop(["engine_power",'mileage','paint_color',"registration_date","model_key","car_type","fuel"],inplace=True,axis=1)

do_prediction(df_copy,True)