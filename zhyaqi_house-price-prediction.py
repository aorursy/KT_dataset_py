# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

import datetime as dt

import matplotlib.pyplot as plt

from scipy import stats

import mpl_toolkits

import seaborn as sns

from scipy.stats import norm, skew

import random 

import warnings

warnings.filterwarnings('ignore')



from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.colors import Normalize  

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/citizens/Citizens"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

path = '../input/citizens/Citizens'



#Read in the data

data = open(path + '/datathon_propattributes.obj', 'rb')

data = pickle.load(data)



pd.set_option('display.max_rows', 200)
#data_original = data.copy()

print ("Size of data : {}" .format(data.shape))
data.columns
#columns are deleted because 'transaction_date' has corresponding 'transaction_dt' column

# and 'IsTraining' does not contain any useful information

columns_to_drop3 = ['transaction_date','IsTraining'] 





for col in columns_to_drop3:

    data.drop(col,axis = 1,inplace = True)



# columns with 'avm' are only available for training samples and not available for testing samples. 

# They are also dropped for consistance.

col_avm = []

for col in data.columns:

    if 'avm' in col:

        col_avm.append(col)

        

for col in col_avm:

    data.drop(col,axis = 1,inplace = True)
data['transaction_year'] = data['transaction_dt'].apply(lambda x: x.year)
mindatetime = min(data['transaction_dt'])

data['transaction_dt'] = (data['transaction_dt'] - mindatetime)

data['transaction_dt'] = data['transaction_dt'].apply(lambda x: x.days)
# def check_skewness(col):

#     sns.distplot(data[col] , fit=norm);

#     fig = plt.figure()

#     res = stats.probplot(data[col], plot=plt)

#     # Get the fitted parameters used by the function

#     (mu, sigma) = norm.fit(data[col])

#     print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    

# check_skewness('sale_amt')
data['prop_state'].unique()
# split dataset into three subsets. Because some features are not contained for all the states.

dataRI = data[data['prop_state'] == 'RI']

dataPA = data[data['prop_state'] == 'PA']

dataMA = data[data['prop_state'] == 'MA']

len(dataRI),len(dataPA),len(dataMA)
var = 'transaction_year'

data_year = pd.concat([dataMA['sale_amt'], dataMA[var]], axis=1)

f, ax = plt.subplots(figsize=(12, 6))

fig = sns.boxplot(x=var, y="sale_amt", data=data_year)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
var = 'transaction_year'

data_year = pd.concat([dataRI['sale_amt'], dataRI[var]], axis=1)

f, ax = plt.subplots(figsize=(12, 6))

fig = sns.boxplot(x=var, y="sale_amt", data=data_year)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
var = 'transaction_year'

data_year = pd.concat([dataPA['sale_amt'], dataPA[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="sale_amt", data=data_year)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
dataMAdesc = pd.DataFrame((dataMA.isnull().sum() / len(dataMA)) * 100, columns = ['missing_ratio'])

dataMAdesc['data_type'] = dataMA.dtypes

dataMAdesc['number_unique'] = np.nan

dataMAdesc['zeros_ratio'] = np.nan

for col in dataMA.columns:

    if (dataMAdesc['data_type'].loc[col] == 'object'):

        dataMAdesc.loc[[col],['number_unique']] = len(dataMA[col].value_counts())

    else:

        dataMAdesc.loc[[col],['zeros_ratio']] = sum(dataMA[col] == 0)/len(dataMA)*100

dataMAdesc['number_unique_ratio'] = dataMAdesc['number_unique']/len(data)*100

dataMAdesc
# large missing ratio

columns_to_drop = ['prop_house_number_2', 'prop_house_number_suffix', 'prop_direction_left','prop_direction_right',

                   'irregular_lot_flg','tax_cd_area','prop_unit_number','prop_unit_type','mobile_home_ind','timeshare_ind',

                   'garage_type','construction_quality']

# only zeros in the columns

columns_to_drop2 = ['market_total_value','market_improvement_value','market_land_value','total_garage_parking_square_feet']

# too many unique values and not helpful with predictions

columns_to_drop4 = ['apn','prop_house_number','prop_street_name','prop_suffix','prop_zip_plus_4']
for col in columns_to_drop:

    dataMA.drop(col,axis = 1,inplace = True)

for col in columns_to_drop2:

    dataMA.drop(col,axis = 1,inplace = True)

for col in columns_to_drop4:

    dataMA.drop(col,axis = 1,inplace = True)
dataMA.shape
# 'census_tract', 'zoning' are all location related. I would like to see the correlation between them.

dataMAnew = pd.DataFrame()

dataMAnew['census_tract'] = pd.factorize(dataMA['census_tract'])[0]

dataMAnew['zoning'] = pd.factorize(dataMA['zoning'])[0]

dataMAnew['sale_amt'] = pd.factorize(dataMA['sale_amt'])[0]

corrmat = dataMAnew.corr()

plt.figure()

g = sns.heatmap(corrmat,cmap="RdYlGn",annot=True)
plt.clf()

f, ax = plt.subplots()

plt.scatter(dataMA['acres'],dataMA['land_square_footage'])

ax.set_xlabel('acres')

ax.set_ylabel('land_square_footage') 

plt.xlim([0, 1300])

plt.ylim([0, 5*10000000])

plt.show()
# There is linear relation between 'acres' and 'land_square_footage'. Drop 'acres'

dataMA.drop('acres',axis = 1,inplace = True)

dataMA.drop('prop_state',axis = 1,inplace = True)
plt.clf()

f, ax = plt.subplots()

plt.scatter(dataMA['assessed_total_value'],dataMA['assessed_land_value'])

ax.set_xlabel('assessed_total_value')

ax.set_ylabel('assessed_land_value')

plt.show()
plt.clf()

f, ax = plt.subplots()

plt.scatter(dataMA['assessed_total_value'],dataMA['assessed_improvement_value'])

ax.set_xlabel('assessed_total_value')

ax.set_xlabel('assessed_improvement_value')

plt.show()
plt.clf()

f, ax = plt.subplots()

plt.hist2d(np.log1p(dataMA['assessed_total_value']/10000),np.log1p(dataMA['sale_amt']/10000),bins = 20)

ax.set_xlabel('assessed_total_value')

ax.set_ylabel('sale_amt')

plt.xlim([0, 6])

plt.ylim([0, 6])

plt.show()
numeric_feats = dataMA.dtypes[dataMA.dtypes != "object"].index

for col in numeric_feats:

    #print(col)

    if col != 'geocode_longitude':

        dataMA[col] = np.log1p(dataMA[col])

    else:

        dataMA[col] = np.log1p(dataMA[col] - min(dataMA[col]))
def plot_images(imgs, labels, cols=3):

    # Set figure to 13 inches x 8 inches



    rows = len(imgs) // cols + 1

    figure = plt.figure(figsize=(16, 4*rows))

    for i in range(len(imgs)):

        subplot = figure.add_subplot(rows, cols, i + 1)

        #subplot.axis('Off')

        if labels:

            subplot.set_title(labels[i], fontsize=14)

        subplot.hist(imgs[i])



#plot_images(plt.hist(np.log1p(data[numeric_feats[1]])), [])

imgs = []       

labels = []

for col in numeric_feats:

    imgs.append(dataMA[col])

    labels.append(col)

plot_images(imgs, labels, cols=4)
corrmat = dataMA[['total_rooms','bedrooms','total_baths_calculated']].corr()

plt.figure()

g = sns.heatmap(corrmat,cmap="RdYlGn",annot=True)
# for some of the columns, it does not make sense to have zero values. I will try to delete those rows.

columns = ['land_square_footage', 'assessed_total_value', 'assessed_land_value', 'assessed_improvement_value',

          'building_square_feet', 'total_rooms']

data_new = dataMA.copy()



for col in columns:

    data_new[col] = data_new[col].where(data_new[col] > 0, np.nan)

    data_new.dropna(subset=[col], inplace = True, axis = 0)

    print(col, len(data_new)/len(dataMA))



data_new.drop(data_new[(data_new['year_built'] == 0) & (data_new['effective_year_built'] == 0)].index, inplace = True)
# Fill the nan in the numeric columns with mean of the column.

numeric_feats = data_new.dtypes[dataMA.dtypes != "object"].index

for col in numeric_feats:

    if data_new[col].isnull().sum() > 0:

        print(col)

        data_new[col].fillna((data_new[col].mean()))
# The data is splitted by date. Entries before 2018-10-01 is training data and after 2018-10-01 is test data.

checkdate = pd.to_datetime('2018-10-01')

datetimediff = np.log1p((checkdate - mindatetime).days)



checkdateend = pd.to_datetime('2018-12-31')

datetimeenddiff = np.log1p((checkdateend - mindatetime).days)

testset = data_new[(data_new['transaction_dt'] >= datetimediff)&(data_new['transaction_dt'] < datetimeenddiff)]

trainset = data_new[data_new['transaction_dt'] < datetimediff]



train_X = trainset.copy()

train_X.drop('sale_amt',axis = 1, inplace = True)

train_Y = trainset['sale_amt']



test_X = testset.copy()

test_X.drop('sale_amt',axis = 1, inplace = True)

test_Y = testset['sale_amt']
train_X.shape,test_X.shape
N1 = len(train_X)

indices = np.arange(len(trainset))

selind = random.sample(list(indices), N1)

train_X_use = train_X.iloc[selind]

train_Y_use = train_Y.iloc[selind]
N2 = len(test_X)

indices = np.arange(len(testset))

selind = random.sample(list(indices), N2)

test_X_use = test_X.iloc[selind]

test_Y_use = test_Y.iloc[selind]
# factorize the categorical columns. For columns with too many unique values, get_dummies would result in dataset with too 

# many columns. I just used factorize to transform these columns. For other columns, I used get_dummies to do the 

# transformation.



feats = ['prop_city','prop_zip_code','census_tract','zoning']

cat_feats = list(train_X_use.dtypes[train_X_use.dtypes == "object"].index)

train_test = pd.concat([train_X_use, test_X_use], ignore_index=True)

for col in feats:

    train_test[col] = pd.factorize(train_test[col], sort = True)[0]

    cat_feats.remove(col)

train_test = pd.get_dummies(train_test, dummy_na = 'True', prefix = cat_feats)

#train_test = RobustScaler().fit_transform(train_test)

# RobustScaler is not quite helpful for random forest model.
train_X_use = train_test[:N1]

test_X_use = train_test[N1:]

train_X_use.shape
from lightgbm import LGBMRegressor



lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=40,#4, 10, 20, 30, 50, 80

                                       learning_rate=0.01, #0.01, 0.002

                                       n_estimators=5000,

                                       max_bin=1000, # 200

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.8, #0.2

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )

lightgbm.fit(train_X_use, train_Y_use)



y_predict = lightgbm.predict(test_X_use)

train_pre = lightgbm.predict(train_X_use)
# Results are shown for the percentage that the prediction is within 10% error of the true trading prices. 



thresh = 0.1

pererror = []

counts = 0

labels = list(train_Y_use)

for i in range(len(train_pre)):

    pererror.append((np.expm1(train_pre[i])-np.expm1(labels[i]))/np.expm1(labels[i]))

    if np.abs(pererror[i]) < thresh:

        counts += 1

print('train', counts/len(train_pre))



pererror = []

counts = 0

labels = list(test_Y_use)

for i in range(len(y_predict)):

    pererror.append((np.expm1(y_predict[i])-np.expm1(labels[i]))/np.expm1(labels[i]))

    if np.abs(pererror[i]) < thresh:

        counts += 1

print('test', counts/len(y_predict))
importantce_list = list(zip(train_test.columns, lightgbm.feature_importances_))

def takeSecond(elem):

    return elem[1]

importantce_list.sort(key=takeSecond, reverse=True)

importantce_list[:20]
labels = list(train_Y_use)

#range = (0, 1000000)

f, ax = plt.subplots(figsize=(12, 10))

f.suptitle('LightGBM model, dataMA', fontsize=18)

bins = np.linspace(0,1500000,40)

plt.subplot(2, 2, 1)

plt.hist(np.expm1(labels),bins= bins)

plt.legend(['True sale_price'],fontsize=14)

plt.subplot(2, 2, 2)

plt.hist(np.expm1(train_pre), bins = bins, color = 'orange')

plt.legend(['Predicted sale_price'],fontsize=14)

plt.subplot(212)

plt.hist(np.expm1(labels),bins= bins)

plt.hist(np.expm1(train_pre), bins = bins, color = 'orange')

plt.legend(['True sale_price', 'Predicted sale_price'],fontsize=14)

plt.show()
dataRIdesc = pd.DataFrame((dataRI.isnull().sum() / len(dataRI)) * 100, columns = ['missing_ratio'])

dataRIdesc['data_type'] = dataRI.dtypes

dataRIdesc['number_unique'] = np.nan

dataRIdesc['zeros_ratio'] = np.nan

for col in dataRI.columns:

    if (dataRIdesc['data_type'].loc[col] == 'object'):

        dataRIdesc.loc[[col],['number_unique']] = len(dataRI[col].value_counts())

    else:

        dataRIdesc.loc[[col],['zeros_ratio']] = sum(dataRI[col] == 0)/len(dataRI)

dataRIdesc['number_unique_ratio'] = dataRIdesc['number_unique']/len(data)

dataRIdesc
# large missing ratio

columns_to_drop = ['prop_house_number_2', 'prop_house_number_suffix', 'prop_direction_left','prop_direction_right',

                   'irregular_lot_flg','tax_cd_area','prop_unit_number','prop_unit_type','mobile_home_ind','timeshare_ind',

                   'garage_type','construction_quality']

# only zeros in the columns

columns_to_drop2 = ['market_total_value','market_improvement_value','market_land_value','total_garage_parking_square_feet','delinquent_tax_year']

# too many unique values and not helpful with predictions

columns_to_drop4 = ['apn','prop_house_number','prop_street_name','prop_suffix','prop_zip_plus_4']
for col in columns_to_drop:

    dataRI.drop(col,axis = 1,inplace = True)

for col in columns_to_drop2:

    dataRI.drop(col,axis = 1,inplace = True)

for col in columns_to_drop4:

    dataRI.drop(col,axis = 1,inplace = True)
print ("Size of dataRI : {}" .format(dataRI.shape))
dataRI.drop('acres', axis = 1, inplace=True)

dataRI.drop('prop_state', axis = 1, inplace=True)
numeric_feats = dataRI.dtypes[dataRI.dtypes != "object"].index

for col in numeric_feats:

    #print(col)

    if col != 'geocode_longitude':

        dataRI[col] = np.log1p(dataRI[col])

    else:

        dataRI[col] = np.log1p(dataRI[col] - min(dataRI[col]))
def plot_images(imgs, labels, cols=3):

    # Set figure to 13 inches x 8 inches



    rows = len(imgs) // cols + 1

    figure = plt.figure(figsize=(16, 4*rows))

    for i in range(len(imgs)):

        subplot = figure.add_subplot(rows, cols, i + 1)

        #subplot.axis('Off')

        if labels:

            subplot.set_title(labels[i], fontsize=14)

        #print(subplot)

        subplot.hist(imgs[i])



#plot_images(plt.hist(np.log1p(data[numeric_feats[1]])), [])

imgs = []       

labels = []

for col in numeric_feats:

    imgs.append(dataRI[col])

    labels.append(col)

plot_images(imgs, labels, cols=4)
# for some of the columns, it does not make sense to have zero values. I will try to delete those rows.

columns = ['land_square_footage', 'assessed_total_value', 'assessed_land_value', 'assessed_improvement_value',

          'building_square_feet', 'total_rooms']

data_new = dataRI.copy()



# for col in columns:

#     data_new[col] = data_new[col].where(data_new[col] > 0, np.nan)

#     data_new.dropna(subset=[col], inplace = True, axis = 0)

#     print(len(data_new)/len(dataRI))



# data_new.drop(data_new[(data_new['year_built'] == 0) & (data_new['effective_year_built'] == 0)].index, inplace = True)
numeric_feats = data_new.dtypes[dataMA.dtypes != "object"].index

for col in numeric_feats:

    if data_new[col].isnull().sum() > 0:

        print(col)

        data_new[col].fillna((data_new[col].mean()),inplace = True)
checkdate = pd.to_datetime('2018-10-01')

datetimediff = np.log1p((checkdate - mindatetime).days)



checkdateend = pd.to_datetime('2018-12-31')

datetimeenddiff = np.log1p((checkdateend - mindatetime).days)

testset = data_new[(data_new['transaction_dt'] >= datetimediff)&(data_new['transaction_dt'] < datetimeenddiff)]

trainset = data_new[data_new['transaction_dt'] < datetimediff]



train_X = trainset.copy()

train_X.drop('sale_amt',axis = 1, inplace = True)

train_Y = trainset['sale_amt']



test_X = testset.copy()

test_X.drop('sale_amt',axis = 1, inplace = True)

test_Y = testset['sale_amt']
train_X.shape, test_X.shape
N1 = len(train_X)

indices = np.arange(len(trainset))

selind = random.sample(list(indices), N1)

train_X_use = train_X.iloc[selind]

train_Y_use = train_Y.iloc[selind]



N2 = len(test_X)

indices = np.arange(len(testset))

selind = random.sample(list(indices), N2)

test_X_use = test_X.iloc[selind]

test_Y_use = test_Y.iloc[selind]
feats = ['prop_city','prop_zip_code','census_tract','zoning']

cat_feats = list(train_X_use.dtypes[train_X_use.dtypes == "object"].index)

train_test = pd.concat([train_X_use, test_X_use], ignore_index=True)

for col in feats:

    train_test[col] = pd.factorize(train_test[col], sort = True)[0]

    cat_feats.remove(col)

train_test = pd.get_dummies(train_test, dummy_na = 'True', prefix = cat_feats)
train_X_use = train_test[:N1]

test_X_use = train_test[N1:]

train_X_use.shape
from lightgbm import LGBMRegressor



lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=40,#4, 10, 20, 30, 50, 80

                                       learning_rate=0.01, #0.01, 0.002

                                       n_estimators=5000,

                                       max_bin=1000, # 200

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.8, #0.2

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )

lightgbm.fit(train_X_use, train_Y_use)



y_predict = lightgbm.predict(test_X_use)

train_pre = lightgbm.predict(train_X_use)
# Results are shown for the percentage that the prediction is within 10% error of the true trading prices. 

thresh = 0.1

pererror = []

counts = 0

labels = list(train_Y_use)

for i in range(len(train_pre)):

    pererror.append((np.expm1(train_pre[i])-np.expm1(labels[i]))/np.expm1(labels[i]))

    if np.abs(pererror[i]) < thresh:

        counts += 1

print('train', counts/len(train_pre))



pererror = []

counts = 0

labels = list(test_Y_use)

for i in range(len(y_predict)):

    pererror.append((np.expm1(y_predict[i])-np.expm1(labels[i]))/np.expm1(labels[i]))

    if np.abs(pererror[i]) < thresh:

        counts += 1

print('test', counts/len(y_predict))
labels = list(train_Y_use)

f, ax = plt.subplots(figsize=(12, 10))

f.suptitle('LightGBM model, dataRI', fontsize=18)

bins = np.linspace(0,1500000,40)

plt.subplot(2, 2, 1)

plt.hist(np.expm1(labels),bins= bins)

plt.legend(['True sale_price'],fontsize=14)

plt.subplot(2, 2, 2)

plt.hist(np.expm1(train_pre), bins = bins, color = 'orange')

plt.legend(['Predicted sale_price'],fontsize=14)

plt.subplot(212)

plt.hist(np.expm1(labels),bins= bins)

plt.hist(np.expm1(train_pre), bins = bins, color = 'orange')

plt.legend(['True sale_price','Predicted sale_price'],fontsize=14)

plt.show()
from sklearn import linear_model



clf = linear_model.HuberRegressor()

clf.fit(train_X_use, train_Y_use)

r2 = clf.score(test_X_use, test_Y_use)

y_predict = clf.predict(test_X_use)

train_pre = clf.predict(train_X_use)
pererror = []

counts = 0

labels = list(train_Y_use)

for i in range(len(train_pre)):

    pererror.append((np.expm1(train_pre[i])-np.expm1(labels[i]))/np.expm1(labels[i]))

    if np.abs(pererror[i]) < 0.1:

        counts += 1

print('train', counts/len(train_pre))



pererror = []

counts = 0

labels = list(test_Y_use)

for i in range(len(y_predict)):

    pererror.append((np.expm1(y_predict[i])-np.expm1(labels[i]))/np.expm1(labels[i]))

    if np.abs(pererror[i]) < 0.1:

        counts += 1

print('test', counts/len(y_predict))
#K-NN

# from sklearn.neighbors import KNeighborsRegressor

# from sklearn.neighbors import RadiusNeighborsRegressor



# model = KNeighborsRegressor(n_neighbors=200, p = 2)

# model.fit(train_X_use, np.array(train_Y_use))

# y_predict = model.predict(test_X_use)

# train_pre = model.predict(train_X_use)
# result for K-NN is:



# train 0.18923403205016867

# test 0.1600549882168107
# from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor



# GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

#                                    max_depth=10, max_features='sqrt',

#                                    min_samples_leaf=15, min_samples_split=10, 

#                                    loss='huber', random_state =5)



# # GBoost = RandomForestRegressor(n_estimators=1000,max_depth = 8)

# GBoost.fit(train_X_use, train_Y_use)

# r2 = GBoost.score(test_X_use, test_Y_use)

# print("[INFO] Random Forest raw pixel accuracy: {:.2f}%".format(r2 * 100))

# y_predict = GBoost.predict(test_X_use)

# train_pre = GBoost.predict(train_X_use)
