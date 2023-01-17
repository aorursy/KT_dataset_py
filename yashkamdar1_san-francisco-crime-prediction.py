# load all the needed packages

import time

import numpy as np

import pandas as pd

import seaborn as sns

import geopandas as gpd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.metrics import log_loss

from sklearn.naive_bayes import BernoulliNB

from sklearn.preprocessing import LabelEncoder

from shapely.geometry import Point, Polygon, shape

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
sns.set(style = 'darkgrid')

sns.set_palette('PuBuGn_d')
# load dataset then parsing the dates column into datetime

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# show the train data

train.head(3)
# show the tolal number of recoreds

train.shape
# check for NAN values

train.isnull().sum()
# check for doublications

train.duplicated().any()
# show the number of doublications

train.duplicated().sum()
# drop duplicate rows and keep only the unique values

train = train.drop_duplicates()

train.shape
# explore crime categories

train['Category'].unique()
# plot the total number of incidents for each category

x = sns.catplot('Category', data = train, kind = 'count', aspect = 3, height = 4.5)

x.set_xticklabels(rotation = 85)
# encode crime categories

le = preprocessing.LabelEncoder()

category = le.fit_transform(train['Category'])
# encode weekdays, districts and hours

district = pd.get_dummies(train['PdDistrict'])

days = pd.get_dummies(train['DayOfWeek'])

train['Dates'] = pd.to_datetime(train['Dates'], format = '%Y/%m/%d %H:%M:%S')

hour = train['Dates'].dt.hour

hour = pd.get_dummies(hour)
# pass encoded values to a new dataframe

enc_train = pd.concat([hour, days, district], axis = 1)

enc_train['Category'] = category
# add gps coordinates

enc_train['X'] = train['X']

enc_train['Y'] = train['Y']
# repeat data handling for test data by encoding weekdays, districts and hours

district = pd.get_dummies(test['PdDistrict'])

days = pd.get_dummies(test['DayOfWeek'])

test['Dates'] = pd.to_datetime(test['Dates'], format = '%Y/%m/%d %H:%M:%S')

hour = test['Dates'].dt.hour

hour = pd.get_dummies(hour)
# create a new dataframe for encoded test values

enc_test = pd.concat([hour, days, district], axis = 1)
# add gps coordinates

enc_test['X'] = test['X']

enc_test['Y'] = test['Y']
training, validation = train_test_split(enc_train, train_size = 0.60)
features = ['Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN', 'X', 'Y']

# add the time

features.extend(x for x in range(0,24))
start = time.time()

model = BernoulliNB()

model.fit(training[features], training['Category'])

predicted = np.array(model.predict_proba(validation[features]))

end = time.time()

secs = (end - start)

loss = log_loss(validation['Category'], predicted)

print("Total seconds: {} and loss {}".format(secs, loss))
# now let's see what log_loss score we get if we apply LogisticRegression

start = time.time()

model = LogisticRegression(C = 0.01)

model.fit(training[features], training['Category'])

predicted = np.array(model.predict_proba(validation[features]))

end = time.time()

secs = (end - start)

loss = log_loss(validation['Category'], predicted)

print("Total seconds: {} and loss {}".format(secs, loss))
model = BernoulliNB()

model.fit(enc_train[features], enc_train['Category'])

predicted = model.predict_proba(enc_test[features])
# extract results

result = pd.DataFrame(predicted, columns = le.classes_)

result.to_csv('results.csv', index = True, index_label = 'Id')