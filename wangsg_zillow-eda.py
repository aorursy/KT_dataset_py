#!/usr/bin/python

import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns

import gc



from sklearn.preprocessing import LabelEncoder



pd.options.display.max_columns = 999
#

# Load data

#

def load_data():

    print("Read data from disk.")

    train_2016 = pd.read_csv('../input/train_2016_v2.csv')

    train_2017 = pd.read_csv('../input/train_2017.csv')

    train = pd.concat([train_2016, train_2017], ignore_index=True)

    properties = pd.read_csv('../input/properties_2017.csv')

    sample = pd.read_csv('../input/sample_submission.csv')



    print("Change properties dtypes from float64 to float32.")

    for c, dtype in zip(properties.columns, properties.dtypes):

        if dtype == np.float64:

            properties[c] = properties[c].astype(np.float32)



    return train, properties, sample



train, properties, sample = load_data()
properties.head(30)
properties.isnull().sum(axis=0)
feature = 'airconditioningtypeid'



#properties['airconditioningtypeid'] = properties['airconditioningtypeid'].fillna(0)

# airconditioningtypeid:Type of cooling system present in the home

print('{} is null:{}'.format(feature, properties['airconditioningtypeid'].isnull().sum(axis=0)))

plt.figure(figsize=(12,8))

sns.countplot(x="airconditioningtypeid", data=properties)

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('airconditioningtypeid Count', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of airconditioningtypeid", fontsize=15)

plt.show()



#properties['A-aircondition-center'] = properties['airconditioningtypeid'].apply(lambda x: 1 if x == 1.0 else 0)
feature = 'architecturalstyletypeid'



#properties['airconditioningtypeid'] = properties['airconditioningtypeid'].fillna(0)



# architecturalstyletypeid: Architectural style of the home

print('{} is null:{}'.format(feature, properties['architecturalstyletypeid'].isnull().sum(axis=0)))

plt.figure(figsize=(12,8))

sns.countplot(x="architecturalstyletypeid", data=properties)

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('architecturalstyletypeid Count', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of architecturalstyletypeid", fontsize=15)

plt.show()



# Label encoder or drop

#properties['A-architectura-contemporary'] = properties['architecturalstyletypeid'].apply(lambda x: 1 if x == 7.0 else 0)
feature = 'buildingclasstypeid'



#properties['buildingclasstypeid'] = properties['buildingclasstypeid'].fillna(-1)



# architecturalstyletypeid: Architectural style of the home

print('{} is null:{}'.format(feature, properties['buildingclasstypeid'].isnull().sum(axis=0)))

plt.figure(figsize=(12,8))

sns.countplot(x="buildingclasstypeid", data=properties)

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('buildingclasstypeid Count', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of buildingclasstypeid", fontsize=15)

plt.show()



properties['A-buildingclasstype'] = properties['buildingclasstypeid'].apply(lambda x: 1 if x==3.0 or x==4.0 else 0)
# propertyzoningdesc



properties['A-propertyzoning'] = properties['propertyzoningdesc'].apply(lambda x: str(x)[0:3])

#properties['A-propertyzoning'] = properties['propertyzoningdesc']

lbl = LabelEncoder()

lbl.fit(list(properties["A-propertyzoning"].values))

properties["A-propertyzoning"] = lbl.transform(list(properties["A-propertyzoning"].values))



plt.figure(figsize=(12,8))

sns.countplot(x="A-propertyzoning", data=properties)

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('propertyzoning Count', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of propertyzoning", fontsize=15)

plt.show()
feature = 'logerror'

#train = train[train.logerror > -0.4]

#train = train[train.logerror < 0.4]



feature_data = train['logerror']





plt.figure(figsize=(16,9))

plt.scatter(range(len(feature_data)), np.sort(feature_data))

plt.show()
feature
from sklearn.cluster import KMeans



# make geo_df

geo_df = properties[['parcelid','latitude', 'longitude']]

geo_df['longitude'] = geo_df['longitude'] / 1e6

geo_df['latitude'] = geo_df['latitude'] / 1e6





geo_df = geo_df[geo_df['latitude'].notnull()]

kmeans = KMeans(n_clusters=12).fit(geo_df[['latitude', 'longitude']])

pre = kmeans.labels_

geo_df['cluster_n'] = pre

#print(pre[:1000])



#centroids = clf.labels_

fig = plt.figure(figsize=(16, 12))

print("drow in figure")

x = np.array(geo_df['latitude'])

y = np.array(geo_df['longitude'])

#al = np.array(abs(geo_df['logerror']))



colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'aqua', 'brown', 'darkblue', 'darkgreen', 'gray']

color = np.absolute(geo_df['cluster_n'])

#print color

for i,c in enumerate(geo_df['cluster_n'][:500]):

    plt.scatter(x[i], y[i], c=colors[c], alpha=0.5)

plt.show()



#

# I have a error

# but, I dont' know why? My computer with python2.7

#
# country use code

properties["A-country-code"] = properties['propertycountylandusecode'].apply(lambda x: str(x)[0:2])

lbl = LabelEncoder()

lbl.fit(list(properties["A-country-code"].values))

properties["A-country-code"] = lbl.transform(list(properties["A-country-code"].values))



# city use code

properties["A-city-code"] = properties['propertycountylandusecode'].apply(lambda x: str(x)[0:3])

lbl = LabelEncoder()

lbl.fit(list(properties["A-city-code"].values))

properties["A-city-code"] = lbl.transform(list(properties["A-city-code"].values))



# town use code

properties["A-town-code"] = properties['propertycountylandusecode'].apply(lambda x: str(x)[3:4])

lbl = LabelEncoder()

lbl.fit(list(properties["A-town-code"].values))

properties["A-town-code"] = lbl.transform(list(properties["A-town-code"].values))



properties[["A-country-code", "A-city-code", "A-town-code"]].head()
# fips

feature = 'fips'



properties[feature] = properties[feature].fillna(-1)

lbl = LabelEncoder()

lbl.fit(list(properties[feature].values))

properties[feature] = lbl.transform(list(properties[feature].values))



plt.figure(figsize=(12,8))

sns.countplot(x=feature, data=properties)

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('fips Count', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of fips", fontsize=15)

plt.show()
#

# Feature Engineering

#

def feature_engineer(train, properties, sample):

    # Load data

    #train, properties, sample = train, properties, sample

    

    # airconditioningtypeid  

    properties['A-aircondition-center'] = properties['airconditioningtypeid'].apply(lambda x: 1 if x == 1.0 else 0)

    properties.drop(['airconditioningtypeid'], axis=1)

    

    # architecturalstyletypeid

    #properties['A-architectura-contemporary'] = properties['architecturalstyletypeid'].apply(lambda x: 1 if x == 7.0 else 0)

    properties.drop(['architecturalstyletypeid'], axis=1)

    

    # buildingclasstypeid

    #properties['A-buildingclasstype'] = properties['buildingclasstypeid'].apply(lambda x: 1 if x==3.0 or x==4.0 else 0)

    properties.drop(['buildingclasstypeid'], axis=1)

    

    print("Feature engineering...")



    print("Label encode object colums")

    id_feature = ['heatingorsystemtypeid', 'propertylandusetypeid', 'storytypeid',

                  'buildingqualitytypeid', 'typeconstructiontypeid']

    for c in properties.columns:

        properties[c] = properties[c].fillna(-1)

        if properties[c].dtype == 'object':

            lbl = LabelEncoder()

            lbl.fit(list(properties[c].values))

            properties[c] = lbl.transform(list(properties[c].values))

        if c in id_feature:

            lbl = LabelEncoder()

            lbl.fit(list(properties[c].values))

            properties[c] = lbl.transform(list(properties[c].values))

            dum_df = pd.get_dummies(properties[c])

            dum_df = dum_df.rename(columns=lambda x: c + str(x))

            properties = pd.concat([properties, dum_df], axis=1)

            properties = properties.drop([c], axis=1)

            # print np.get_dummies(properties[c])



    # life of property

    properties['N-life'] = 2018 - properties['yearbuilt']



    # error in calculation of the finished living area of home

    properties['N-LivingAreaError'] = properties['calculatedfinishedsquarefeet'] / properties['finishedsquarefeet12']



    # proportion of living area

    properties['N-LivingAreaProp'] = properties['calculatedfinishedsquarefeet'] / properties['lotsizesquarefeet']

    properties['N-LivingAreaProp2'] = properties['finishedsquarefeet12'] / properties['finishedsquarefeet15']



    # Amout of extra space

    properties['N-ExtraSpace'] = properties['lotsizesquarefeet'] - properties['calculatedfinishedsquarefeet']

    properties['N-ExtraSpace-2'] = properties['finishedsquarefeet15'] - properties['finishedsquarefeet12']



    # Total number of rooms

    properties['N-TotalRooms'] = properties['bathroomcnt'] + properties['bedroomcnt']



    # Average room size

    #properties['N-AvRoomSize'] = properties['calculatedfinishedsquarefeet'] / properties['roomcnt']



    # Number of Extra rooms

    properties['N-ExtraRooms'] = properties['roomcnt'] - properties['N-TotalRooms']



    # Ratio of the built structure value to land area

    properties['N-ValueProp'] = properties['structuretaxvaluedollarcnt'] / properties['landtaxvaluedollarcnt']



    # Does property have a garage, pool or hot tub and AC?

    #properties['N-GarPoolAC'] = ((properties['garagecarcnt'] > 0) & (properties['pooltypeid10'] > 0) & (properties['airconditioningtypeid'] != 5)) * 1



    properties["N-location"] = properties["latitude"] + properties["longitude"]

    properties["N-location-2"] = properties["latitude"] * properties["longitude"]

    properties["N-location-2round"] = properties["N-location-2"].round(-4)



    properties["N-latitude-round"] = properties["latitude"].round(-4)

    properties["N-longitude-round"] = properties["longitude"].round(-4)



    # Ratio of tax of property over parcel

    properties['N-ValueRatio'] = properties['taxvaluedollarcnt'] / properties['taxamount']



    # TotalTaxScore

    properties['N-TaxScore'] = properties['taxvaluedollarcnt'] * properties['taxamount']



    # polnomials of tax delinquency year

    properties["N-taxdelinquencyyear-2"] = properties["taxdelinquencyyear"] ** 2

    properties["N-taxdelinquencyyear-3"] = properties["taxdelinquencyyear"] ** 3



    # Length of time since unpaid taxes

    properties['N-live'] = 2018 - properties['taxdelinquencyyear']



    # Number of properties in the zip

    zip_count = properties['regionidzip'].value_counts().to_dict()

    properties['N-zip_count'] = properties['regionidzip'].map(zip_count)



    # Number of properties in the city

    city_count = properties['regionidcity'].value_counts().to_dict()

    properties['N-city_count'] = properties['regionidcity'].map(city_count)



    # Number of properties in the city

    region_count = properties['regionidcounty'].value_counts().to_dict()

    properties['N-county_count'] = properties['regionidcounty'].map(region_count)



    print("Set train and test dataframe.")

    train = train.merge(properties, on='parcelid', how='left')

    sample['parcelid'] = sample['ParcelId']

    test = sample.merge(properties, on='parcelid', how='left')

    test['transactiondate'] = '2017-01-01'



    #

    # add month feature

    #

    train["transactiondate"] = pd.to_datetime(train["transactiondate"])

    train['A-year'] = train['transactiondate'].dt.year

    train["A-month"] = train["transactiondate"].dt.month

    train["A-quarter"] = train["transactiondate"].dt.quarter

    test["transactiondate"] = pd.to_datetime(test["transactiondate"])

    test['A-year'] = test['transactiondate'].dt.year

    test["A-month"] = test["transactiondate"].dt.month

    test["A-quarter"] = test["transactiondate"].dt.quarter



    #

    # Drop outlier

    #

    train = train[train.logerror > -0.4]

    train = train[train.logerror < 0.419]



    x_train = train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'],

                         axis=1)

    y_train = train["logerror"].values

    x_test = test[x_train.columns]



    print("train and test shape:")

    print(x_train.shape, x_test.shape)



    return x_train, y_train, x_test



x_train, y_train, x_test = feature_engineer(train, properties, sample)
x_train.head(30)
import lightgbm as lgb

from sklearn.model_selection import train_test_split

# default lgb params

params = {}

params['max_bin'] = 10

params['learning_rate'] = 0.037 # shrinkage_rate

params['boosting_type'] = 'gbdt'

params['objective'] = 'regression'

params['metric'] = 'mae'          # or 'mae'

params['sub_feature'] = 0.35    # feature_fraction (small values => use very different submodels)

params['bagging_fraction'] = 0.85 # sub_row

params['bagging_freq'] = 40

params['num_leaves'] = 512        # num_leaf

params['min_data'] = 500         # min_data_in_leaf

params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

params['verbose'] = 0

params['feature_fraction_seed'] = 2

params['bagging_seed'] = 3



x_train_df, x_valid, y_train_df, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

print("x_train shape:{}\n"

      "x_valid shape:{}".format(x_train_df.shape, x_valid.shape))



lgb_train = lgb.Dataset(x_train_df, label=y_train_df)

lgb_valid = lgb.Dataset(x_valid, label=y_valid)



gc.collect()



watchlist = [lgb_valid]

#watchlist = [(lgb_train, 'train'), (lgb_valid, 'valid')]

clf = lgb.train(params, lgb_train, 2000, valid_sets=watchlist, early_stopping_rounds=20)

# 0.0518894
fig, ax = plt.subplots(figsize=(12,20))

lgb.plot_importance(clf, max_num_features=50, height=0.8, ax=ax)