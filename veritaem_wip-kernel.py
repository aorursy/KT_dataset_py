!pip install category_encoders
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, PolynomialFeatures

from sklearn.metrics import precision_score, r2_score, mean_squared_error, classification_report, confusion_matrix, mean_absolute_error, recall_score, accuracy_score

from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectKBest, f_regression, f_classif

import seaborn as sns

import category_encoders as ce

import xgboost as xgb

import os



print(os.listdir("../input"))


#importing the training features, training labels, and test

df1 = pd.read_csv('../input/kaggle-comp-proj-2/train_features.csv')

df2 = pd.read_csv('../input/kaggle-comp-proj-2/train_labels.csv')

testdf = pd.read_csv('../input/kaggle-comp-proj-2/test_features.csv')



#joining the labels and features just so i dont get confused with my numbers

df = df1.join(df2, rsuffix = 'status_group')



#pulling out the ids from the test df in order to reattach them to the predictions later

id = pd.DataFrame(testdf['id'])



df.head()
print(df.shape)

print(df.isna().sum())

nan_values_list = ['Not Known', 'Unknown', 'None', 'Not known', 'not known',

                  '-', 'unknown', 'Unknown Installer', '##', 'none', '0']



df = df.replace(nan_values_list, np.nan)

testdf = testdf.replace(nan_values_list, np.nan)



df = df.replace(np.nan, 'unknown')

testdf = testdf.replace(np.nan, 'unknown')
'''because certain columns seem unimportant, redundant, 

or missing most of their values, 

I will drop them in this cell, to be modified as needed later'''

df2 = df.drop(['scheme_name', 'public_meeting', 'payment_type', 'region', 'idstatus_group', 'recorded_by', 'latitude', 'longitude'],axis = 1)

testdf = testdf.drop(['scheme_name', 'public_meeting', 'payment_type', 'region', 'recorded_by', 'latitude', 'longitude'],axis = 1)
#this will get us the error from just guessing the most often classifications

yencode = df['status_group'].replace({'functional':2, 'functional needs repair':1, 'non functional':0})

yencode = pd.DataFrame(yencode)

u = yencode.status_group.mean()

baseline = [u] * len(yencode.status_group)  

print(mean_absolute_error(yencode.status_group, baseline))

maj_classification = yencode.status_group.mode()

y_pred = np.full(shape=yencode['status_group'].shape, fill_value=maj_classification)

print(recall_score(yencode['status_group'], y_pred, average = 'micro'))
#latitude and longitude arent too telling by themselves, but together they give a unique position

df2['latlong'] = abs(df['latitude'].round(2)) + abs(df['longitude'].round(2))

testdf['latlong'] = abs(df['latitude'].round(2)) + abs(df['longitude'].round(2))



#I see two values that would really ebnefit from a bin, and those are 

#population and construction year, lets do it!

year_bins = [-1, 1980, 1990, 2000, 2010, 2020]

year_labels = [1, 2, 3, 4, 5]

df2['year_made'] = pd.cut(df2['construction_year'], bins = year_bins, labels = year_labels)

testdf['year_made'] = pd.cut(testdf['construction_year'], bins = year_bins, labels = year_labels)



pop_bins = [-1, 10, 20, 100, 250, 1000, 5000, 10000, 100000]

pop_labels = [1, 2, 3, 4, 5, 6, 7, 8]

df2['pop'] = pd.cut(df2['population'], bins = pop_bins, labels = pop_labels)

testdf['pop'] = pd.cut(testdf['population'], bins = pop_bins, labels = pop_labels)
#alright, whats it looking like?  Numbers, awayyyyyy~

print(df2.nunique())

df2.describe(include = 'all')
df2['wpt_name'].value_counts()

# i wanted to reduce the number of unique variables in each column, and it looks like a jackpot!

#a bunch of these look similar to each other, im sure theres some regex for it!...

#...and heres my version! 





df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*clinic.*$)', 'health')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*hospital.*$)', 'health')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*zahanati.*$)', 'health')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*health.*$)', 'health')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*secondary.*$)', 'school')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*school.*$)', 'school')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*shule.*$)', 'school')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*sekondari.*$)', 'school')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*msingi.*$)', 'school')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*primary.*$)', 'school')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*secondary.*$)', 'school')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*school.*$)', 'school')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*shule.*$)', 'school')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*sekondari.*$)', 'school')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*msingi.*$)', 'school')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*primary.*$)', 'school')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*clinic.*$)', 'health')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*hospital.*$)', 'health')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*zahanati.*$)', 'health')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*health.*$)', 'health')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*ccm.*$)', 'official')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*office.*$)', 'official')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*kijiji.*$)', 'official')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*ofis.*$)', 'official')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*idara.*$)', 'offical')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*maziwa.*$)', 'farm')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*farm.*$)', 'farm')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*maji.*$)', 'pump')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*water.*$)', 'pump')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*pump house.*$)', 'pump')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*pump.*$)', 'pump')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*bombani.*$)', 'pump')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*center.*$)', 'center')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*madukani.*$)', 'center')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*sokoni.*$)', 'center')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*market.*$)', 'center')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*kwa.*$)', 'name')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*ccm.*$)', 'official')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*office.*$)', 'official')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*kijiji.*$)', 'official')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*ofis.*$)', 'official')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*idara.*$)', 'offical')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*maziwa.*$)', 'farm')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*farm.*$)', 'farm')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*maji.*$)', 'pump')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*water.*$)', 'pump')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*pump house.*$)', 'pump')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*pump.*$)', 'pump')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*bombani.*$)', 'pump')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*center.*$)', 'center')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*madukani.*$)', 'center')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*sokoni.*$)', 'center')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*market.*$)', 'center')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*kwa.*$)', 'name')

df2['wpt_name'] = df2.wpt_name.str.replace(r'((?i)^.*none.*$)', 'other')

testdf['wpt_name'] = testdf.wpt_name.str.replace(r'((?i)^.*none.*$)', 'other')



#and then removing anything with less than 100 entries

value_counts = df2['wpt_name'].value_counts()

to_remove = value_counts[value_counts <= 100].index

df2['wpt_name'].replace(to_remove, 'other', inplace=True)

value_counts = testdf['wpt_name'].value_counts()

to_remove = value_counts[value_counts <= 100].index

testdf['wpt_name'].replace(to_remove, 'other', inplace=True)



df2['wpt_name'].nunique()

#a little better
#so what are we working with here

testdf.nunique()
#ok lga seems important, lets go ahead and get that one manageable next



series = df2['lga'].copy()

series[series.str.contains('Rural')] = 'rural'

series[series.str.contains('Urban')] = 'urban'

other_flag = series.str.contains('rural') | series.str.contains('urban')

other_flag = other_flag == False

series[other_flag] = 'other'



df2['lga'] = series



series = testdf['lga'].copy()

series[series.str.contains('Rural')] = 'rural'

series[series.str.contains('Urban')] = 'urban'

other_flag = series.str.contains('rural') | series.str.contains('urban')

other_flag = other_flag == False

series[other_flag] = 'other'



testdf['lga'] = series
#population number is a string apparently??...

df2['population'] = df2['population'].astype(int)

testdf['population'] = testdf['population'].astype(int)

#not anymore!  Nice try, python, but you cant beat me with simple string!
#i want to make two new columns, one which shows the date 

#recorded as an integer, for manipulating purposes..

#the other is a column that takes the recorded date and

#the year and makes an approx. years in operation column

df2['2011'] = df2['date_recorded'].str.contains('2011', na=False, regex=True)

df2['2012'] = df2['date_recorded'].str.contains('2012', na=False, regex=True)

df2['2013'] = df2['date_recorded'].str.contains('2013', na=False, regex=True)



testdf['2011'] = testdf['date_recorded'].str.contains('2011', na=False, regex=True)

testdf['2012'] = testdf['date_recorded'].str.contains('2012', na=False, regex=True)

testdf['2013'] = testdf['date_recorded'].str.contains('2013', na=False, regex=True)



df2['2011'] = df2['2011'].replace(True, 2011)

df2['2012'] = df2['2012'].replace(True, 2012)

df2['2013'] = df2['2013'].replace(True, 2013)



testdf['2011'] = testdf['2011'].replace(True, 2011)

testdf['2012'] = testdf['2012'].replace(True, 2012)

testdf['2013'] = testdf['2013'].replace(True, 2013)



larger = lambda s1, s2: s1 if s1.sum() > s2.sum() else s2



df2['yrs_intermediate'] = df2['2011'].combine(df2['2012'], larger)

df2['yrs_intermediate'] = df2['yrs_intermediate'].combine(df2['2013'], larger)

df2['yrs in operation'] = df2['yrs_intermediate'] - df2['construction_year']



testdf['yrs_intermediate'] = testdf['2011'].combine(testdf['2012'], larger)

testdf['yrs_intermediate'] = testdf['yrs_intermediate'].combine(testdf['2013'], larger)

testdf['yrs in operation'] = testdf['yrs_intermediate'] - testdf['construction_year']



df2.head()
# But first!  Targets and features separated, plz and ty!

X = df2.drop(['status_group', 'funder', 'installer', 'subvillage', 'construction_year', 'num_private',

             'extraction_type_group', 'quantity_group', 'source_class', 'source_type', 'subvillage',

             'permit', 'date_recorded', '2011', '2012', '2013', 'id', 'waterpoint_type_group',

             'amount_tsh', 'management_group', 'district_code', 'quality_group',

             'extraction_type_class'], axis = 1)

y = df2['status_group']



Xtdf = testdf.drop(['funder', 'installer', 'subvillage', 'construction_year', 'num_private',

                   'extraction_type_group', 'quantity_group', 'source_class', 'source_type',

                   'subvillage', 'permit', 'date_recorded', '2011', '2012', '2013', 'id', 'waterpoint_type_group',

                   'amount_tsh', 'management_group', 'district_code', 'quality_group',

                   'extraction_type_class'], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y)
#After trying a bunch of pipelines, i settled on

from sklearn.ensemble import RandomForestClassifier

pipelinef = make_pipeline(

    ce.OneHotEncoder(use_cat_names=True),

    RobustScaler(),

    RandomForestClassifier(n_estimators = 1000)

)

#possibly the others would work with more feature engineering, but i got lower scores with them

#and i spent half the week looking up regex :,D
pipelinef.fit(X_train, y_train)

y_pred = pipelinef.predict(X_test)

accuracy_score(y_test, y_pred)

#yay thats not horrible~
#convoluted but it works!

#turning my pipeline prediction into a DF then joining it up with the index to make a dec file

subby = pipelinef.predict(Xtdf)

print(subby.shape)

subby = pd.DataFrame(subby)

subs = id.join(subby, rsuffix = '0')

subs = subs.rename(index=str, columns={0:'status_group'})

print(subs.shape)

subs.head()
subs.to_csv('forest_for_the_trees.csv', index=False)

#alright this was my best performing models and features out of many more, now get this out of my face!!