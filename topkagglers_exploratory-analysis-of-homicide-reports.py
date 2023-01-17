import numpy as np
import scipy as scp
import pandas as pd
import seaborn as sns
import re
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report

import time
import pickle
import warnings
import math
import graphviz
%matplotlib inline
warnings.filterwarnings('ignore')
homicideDF = pd.read_csv(r"../input/homicide-reports/database.csv")
homicideDF.drop(labels=['Record ID', 'Agency Code'], axis=1, inplace=True)
homicideDF = homicideDF[homicideDF['Perpetrator Age'] != " "]
homicideDF['Perpetrator Age'] = homicideDF['Perpetrator Age'].astype(np.int64)
homicideDF.shape
with open('../input/state-lat-long/cityinfo.pickle', 'rb') as f:
    cityLatLongDF = pickle.load(f)
homicideDF.head()
homicideDF.columns
#def do_geocode(p_city_name):
#    try:
#        return geolocator.geocode(p_city_name)
#    except GeocoderTimedOut:
#        print("time out for city: ", p_city_name)
#        time.sleep(45)
#        return do_geocode(p_city_name)


#city_latlong = []
#geolocator = Nominatim(timeout=3)

#for index, row in crimeCntByCity.iterrows():
#    loc = do_geocode(row['city'])
#    if loc is not None:
#        city_latlong.append((row['city'], loc.longitude, loc.latitude))
#    else:
#        city_latlong.append((row['city'], None, None))
#len(city_latlong)
crimeCntByCity = homicideDF['City'].value_counts().reset_index()
crimeCntByCity.columns = ['city', 'cnt']

map = Basemap(width=10000000,height=6000000,projection='lcc',
            resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
plt.figure(figsize=(25,25))
map.bluemarble()

for city, goe_long, geo_lat in cityLatLongDF.to_dict('split')['data']:
    x, y = map(goe_long, geo_lat)
    crime_cnt = crimeCntByCity[crimeCntByCity['city'] == city]['cnt'].values[0]
    map.plot(x,y,marker='o',color='Red',markersize= round(np.log(crime_cnt) + 1) ** 1.3)
    plt.annotate(city, xy = (x,y), xytext=(-20,20))
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(12, 10)})

''' picking top 10'''
tempDF = homicideDF.groupby(['State']).size().reset_index(name='cnt').sort_values(by='cnt', ascending=False).iloc[:10]
top_crime_cities = list(tempDF['State'])
sns.countplot(x='State', data=homicideDF[homicideDF['State'].isin(top_crime_cities)])
homicideDF.groupby(['Year']).size().plot(kind='line', color='green')
#sns.pairplot(data=homicideDF[['Victim Age', 'Perpetrator Age']], kind='scatter')
plt.scatter(x='Victim Age', y='Perpetrator Age', data=homicideDF, marker='*', color='red')
plt.xlabel("Victim Age")
plt.ylabel("Perpetrator Age")
homicideDF[['Victim Age', 'Perpetrator Age']].describe()
''' Rule of thumb to select number of bins'''
#bin_width = 2 * iqr * (n ** -1/3)
# num_bins = (max - min)/bin_width

bin_width = 2 * 20 * (homicideDF.shape[0] ** (-0.333))
l_max = np.max(homicideDF[homicideDF['Victim Age'] < 998]['Victim Age'])
l_min = np.min(homicideDF[homicideDF['Victim Age'] < 998]['Victim Age'])
num_bins = int((l_max - l_min)/bin_width)
print("optimal bins to create: ", num_bins)

homicideDF[homicideDF['Victim Age'] < 998]['Victim Age'].plot.hist(alpha = 0.75, bins=100)
bin_width = 2 * 20 * (homicideDF.shape[0] ** (-0.333))
l_max = np.max(homicideDF['Perpetrator Age'])
l_min = np.min(homicideDF['Perpetrator Age'])
num_bins = int((l_max - l_min)/bin_width)
print("optimal bins to create: ", num_bins)

homicideDF['Perpetrator Age'].plot.hist(alpha = 0.75, bins=100)
homicideDF[homicideDF['Perpetrator Age'] > 0]['Perpetrator Age'].plot.hist(alpha = 0.75, bins=100)
plt.scatter(x='Victim Age', y='Perpetrator Age', \
            data=homicideDF[(homicideDF['Perpetrator Age'] > 0) & (homicideDF['Victim Age'] < 998)], \
            marker='*', color='red')
plt.xlabel("Victim Age")
plt.ylabel("Perpetrator Age")
tempDF = homicideDF[(homicideDF['Perpetrator Age'] > 0) & (homicideDF['Victim Age'] < 998)]
np.corrcoef(x=tempDF['Perpetrator Age'], \
           y=tempDF['Victim Age'])
tempDF = homicideDF[(homicideDF['Perpetrator Age'] > 0) & (homicideDF['Victim Age'] < 998)]
tempDF[['Perpetrator Age', 'Victim Age']].plot.hist(stacked=True, bins=50)
homicideDF[(homicideDF['Victim Sex'] != 'Unknown') & \
          (homicideDF['Victim Age'] < 998)].boxplot(column=['Victim Age'], by=['Victim Sex'], vert=False)
grouped = homicideDF[homicideDF['Victim Age'] < 998].groupby('Victim Sex')
grouped['Victim Age'].agg([np.count_nonzero, np.mean, np.std])
homicideDF[(homicideDF['Victim Sex'] != 'Unknown') & \
           (homicideDF['Perpetrator Age'] > 0)].boxplot(column=['Perpetrator Age'], by=['Victim Sex'], vert=False)
grouped = homicideDF[homicideDF['Perpetrator Age'] > 0].groupby('Victim Sex')
grouped['Perpetrator Age'].agg([np.count_nonzero, np.mean, np.std])
homicideDF['Perpetrator Sex'].value_counts()
homicideDF['Victim Sex'].value_counts()
grouped = homicideDF[['Perpetrator Sex', 'Perpetrator Age']].groupby('Perpetrator Sex')
grouped.hist(color='red', alpha=0.6, bins=50)
def computePercentages(p_df):
    temp_arr = (p_df.as_matrix()/np.sum(p_df.as_matrix(), axis = 1).reshape(len(p_df.index.values), 1)) * 100.0
    return pd.DataFrame(temp_arr, index=p_df.index.values, columns=p_df.columns)
tempDF = pd.pivot_table(data=homicideDF, \
                       index=['Victim Sex'], \
                       columns =['Victim Race'], \
                       values = ['Victim Age'],
                       aggfunc='count', \
                       margins=False)
tempDF
computePercentages(tempDF)
tempDF.plot.bar(stacked=True, figsize=(12, 10), title='Victim Sex vs Victim Race')
scp.stats.chi2_contingency(observed=tempDF.as_matrix()[:-1,:])
tempDF = pd.pivot_table(data=homicideDF[homicideDF['Perpetrator Sex'] != 'Unknown'], \
                       index=['Perpetrator Sex'], \
                       columns =['Perpetrator Race'], \
                       values = ['Victim Age'], \
                       aggfunc='count', \
                       margins=False)
tempDF
computePercentages(tempDF)
tempDF.plot.bar(stacked=True, figsize=(12, 10), title='Perpetrator Sex vs Perpetrator Race')
scp.stats.chi2_contingency(observed=tempDF.as_matrix())
l_cond = (homicideDF['Perpetrator Sex'] != 'Unknown') & \
            (homicideDF['Victim Sex'] != 'Unknown') & \
            (homicideDF['Relationship'] != 'Unknown')
homicideDF[l_cond][['Perpetrator Sex', 'Victim Sex', 'Relationship']].head(10)
homicideDF['Relationship'].value_counts()
homicideDF['Weapon'].value_counts()
def boxplot_sorted(df, by, column):
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    meds = df2.median().sort_values()
    df2[meds.index].boxplot(rot=45, figsize=(16, 11))
boxplot_sorted(homicideDF[homicideDF['Perpetrator Age'] > 0], by='Relationship', column='Perpetrator Age')
boxplot_sorted(homicideDF[homicideDF['Perpetrator Age'] > 0], by='Weapon', column='Perpetrator Age')
tempDF = pd.pivot_table(data=homicideDF[~homicideDF['Perpetrator Race'].isin(['Unknown'])], \
                       index=['Perpetrator Race'], \
                       columns =['Weapon'], \
                       values = ['Victim Age'], \
                       aggfunc='count', \
                       margins=False)
tempDF
computePercentages(tempDF)
tempDF = pd.pivot_table(data=homicideDF[~homicideDF['Weapon'].isin(['Unknown'])], \
                       index=['State'], \
                       columns =['Weapon'], \
                       values = ['City'], \
                       aggfunc='count', \
                       margins=False, 
                       fill_value=0)
tempDF
computePercentages(tempDF).head()
print("Shape of the input data: ", homicideDF.shape)
homicideDF.columns
l_cond = ((homicideDF['Victim Age'] < 990) & \
           (homicideDF['Victim Sex'].isin(['Male', 'Female'])) & \
           (homicideDF['Perpetrator Sex'].isin(['Male', 'Female'])) & \
           (homicideDF['Victim Race'] != 'Unknown') & \
           (homicideDF['Weapon'] != 'Unknown') & \
           (homicideDF['Perpetrator Race'] != 'Unknown') & \
           (homicideDF['Relationship'] != 'Unknown'))
modelData = homicideDF[l_cond]
modelData.shape
modelData['Perpetrator Race'].value_counts()
d1 = pd.get_dummies(modelData['State'])
d2 = pd.get_dummies(modelData['Victim Sex'])
d3 = pd.get_dummies(modelData['Victim Race'])
d4 = pd.get_dummies(modelData['Weapon'])
d5 = pd.get_dummies(modelData['Relationship'])
d6 = pd.get_dummies(modelData['Perpetrator Sex'])

''' Define target column and encode'''
target_col = 'Race'
label_encoder = LabelEncoder()
label_encoder.fit_transform(modelData['Perpetrator Race'])
targetDF = pd.DataFrame(label_encoder.fit_transform(modelData['Perpetrator Race']), columns=[target_col])
#targetDF.shape
transformedDF = pd.concat([d1, d2, d3, d4, d5, modelData['Victim Age']], axis=1)
transformedDF.shape
''' Input variables are State, Victim Age/Sex/Race, weapon'''
input_cols = list(set(transformedDF.columns) - set([target_col]))

X_train, X_test, y_train, y_test  = train_test_split(transformedDF[input_cols], targetDF[target_col], \
                                                     train_size = 0.75, test_size = 0.25, \
                                                     stratify=targetDF[target_col], \
                                                     random_state = 1234)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X=np.array(X_train), y=y_train)
predicted_labels = clf.predict(X=X_test)
pd.crosstab(y_test, predicted_labels, rownames=['True'], colnames=['Predicted'], margins=True)
print("Accuracy score: on test data", round(accuracy_score(y_test, predicted_labels) * 100, 2))
print(classification_report(y_test, predicted_labels))
print(classification_report(y_train,clf.predict(X=X_train)))
