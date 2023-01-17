import numpy as np

import pandas as pd
# visualization



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from mpl_toolkits.basemap import Basemap

import folium

from folium import plugins
from sklearn.cross_validation import train_test_split
from sklearn import linear_model

import statsmodels.api as sm
df = pd.read_csv('../input/crime.csv', encoding = "ISO-8859-1")
def print_five_rows(name_column):

    return df[name_column][0:5]
def describe_column(name_column):

    return df[name_column].describe()
def create_list_number_crime(name_column, list_unique):

    # list_unique = df[name_column].unique()

    

    i = 0

    

    list_number = list()

    

    while i < len(list_unique):

        list_number.append(len(df.loc[df[name_column] == list_unique[i]]))

        i += 1

    

    return list_unique, list_number
def pie_plot(list_number, list_unique):

    plt.figure(figsize=(20,10))

    plt.pie(list_unique, 

        labels=list_number,

        autopct='%1.1f%%', 

        shadow=True, 

        startangle=140)

 

    plt.axis('equal')

    plt.show()

    return 0
def bar_chart(list_number, list_unique):

    objects = list_unique

    y_pos = np.arange(len(objects))

    performance = list_number

 

    plt.figure(figsize=(20,10))    

    plt.bar(y_pos, performance, align='center', alpha=0.5)

    plt.xticks(y_pos, objects)

    plt.ylabel('Number') 

    plt.show()

    

    return 0
def drop_NaN_two_var(x, y):



    df1 = df[[x, y]].dropna()

    print(df1.shape)



    x_value = df1[x]

    y_value = df1[y]



    del df1

        

    print(x + ': ' + str(x_value.shape))

    print(y + ': ' + str(y_value.shape))

        

    return x_value, y_value
def function_OLS_Regression(x, y):

    

    model = sm.OLS(y, x)

    res = model.fit()

    return res.summary()
df.shape
df.columns
df.isnull().sum()
print_five_rows('INCIDENT_NUMBER')
describe_column('INCIDENT_NUMBER')
df = df.drop('INCIDENT_NUMBER', 1)
print_five_rows('OFFENSE_CODE')
len(df['OFFENSE_CODE'].unique())
print_five_rows('OFFENSE_CODE_GROUP')
describe_column('OFFENSE_CODE_GROUP')
print_five_rows('OFFENSE_DESCRIPTION')
describe_column('OFFENSE_DESCRIPTION')
print_five_rows('DISTRICT')
describe_column('DISTRICT')
df['DISTRICT'].unique()
print_five_rows('REPORTING_AREA')
describe_column('REPORTING_AREA')
print_five_rows('SHOOTING')
df['SHOOTING'].unique()
print_five_rows('OCCURRED_ON_DATE')
df['OCCURRED_ON_DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'])
describe_column('OCCURRED_ON_DATE')
print_five_rows('YEAR')
df['MONTH'].unique()
df['DAY_OF_WEEK'].unique()
df['HOUR'].unique()
df['UCR_PART'].unique()
print_five_rows('STREET')
describe_column('STREET')
df[['Lat', 'Long']].head()
describe_column('Lat')
describe_column('Long')
df['Location'].head()
plt.figure(figsize=(16,8))

df['DISTRICT'].value_counts().plot.bar()

plt.show()
# 2015

plt.figure(figsize=(8,4))

df['DISTRICT'].loc[df['YEAR']==2015].value_counts().plot.bar()

plt.show()



# 2016

plt.figure(figsize=(8,4))

df['DISTRICT'].loc[df['YEAR']==2016].value_counts().plot.bar()

plt.show()



# 2017

plt.figure(figsize=(8,4))

df['DISTRICT'].loc[df['YEAR']==2017].value_counts().plot.bar()

plt.show()



# 2018

plt.figure(figsize=(8,4))

df['DISTRICT'].loc[df['YEAR']==2018].value_counts().plot.bar()

plt.show()
i = 1



while i < 13:

    print('== ' + str(i) + ' ==')

    print(df['DISTRICT'].loc[df['MONTH']==i].value_counts())

    i +=1
list_unique_year, list_number_year = create_list_number_crime('YEAR',df['YEAR'].unique())
pie_plot(list_unique_year, list_number_year)
bar_chart(list_number_year,list_unique_year)
list_unique_month, list_number_month = create_list_number_crime('MONTH',list(range(1,13)))
# pie_plot(list_unique_month,list_number_month)
bar_chart(list_number_month,list_unique_month)
day_of_week = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
list_unique_day, list_number_day = create_list_number_crime('DAY_OF_WEEK',day_of_week)
#pie_plot(list_unique_day,list_number_day)
bar_chart(list_number_day,list_unique_day)
list_unique_hour, list_number_hour = create_list_number_crime('HOUR',list(range(0,24)))
# pie_plot(list_unique_hour, list_number_hour)
bar_chart(list_number_hour,list_unique_hour)
df['SHOOTING'].fillna(0, inplace = True)



df['SHOOTING'] = df['SHOOTING'].map({

    0: 0,

    'Y':1

})
shoot_true = len(df.loc[df['SHOOTING'] == 1])

shoot_false = len(df.loc[df['SHOOTING'] == 0])
print('With shooting(num): ' + str(shoot_true))

print('With shooting(%):   ' + str(round(shoot_true*100/len(df),2))+'%')

print()

print('Without shooting(num): ' + str(shoot_false))

print('Without shooting(%):   ' + str(round(shoot_false*100/len(df),2))+'%')
df_shoot = df.loc[df['SHOOTING'] == 1]

df_shoot.shape
shoot_y_2015 = len(df_shoot.loc[df_shoot['YEAR'] == 2015])

shoot_y_2016 = len(df_shoot.loc[df_shoot['YEAR'] == 2016])

shoot_y_2017 = len(df_shoot.loc[df_shoot['YEAR'] == 2017])

shoot_y_2018 = len(df_shoot.loc[df_shoot['YEAR'] == 2018])



unique_shoot_year = '2015', '2016', '2017', '2018'

number_shoot_year = [shoot_y_2015, shoot_y_2016, shoot_y_2017, shoot_y_2018]
# pie_plot(unique_shoot_year,number_shoot_year)
bar_chart(number_shoot_year,unique_shoot_year)
i = 1

list_month = list()



while i <= 12:

    list_month.append(len(df_shoot.loc[df_shoot['MONTH'] == i]))

    i+=1
# pie_plot(list(range(1,13)), list_month)
bar_chart(list_month,list(range(1,13)))
day_of_week = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')



i = 0

day_number = list()



while i < 7:

    day_number.append(len(df_shoot.loc[df_shoot['DAY_OF_WEEK'] == day_of_week[i]]))

    

    i +=1
# pie_plot(day_of_week, day_number)
bar_chart(day_number,day_of_week)
i = 0

hour_number = list()



while i < 24:

    hour_number.append(len(df_shoot.loc[df_shoot['HOUR'] == i]))

    i +=1
# pie_plot(list(range(0,24)), hour_number)
bar_chart(hour_number,list(range(0,24)))
plt.figure(figsize=(20,10))

df_shoot['DISTRICT'].value_counts().plot.bar()

plt.show()
location_shoot = df_shoot[['Lat','Long']]

location_shoot = location_shoot.dropna()



location_shoot = location_shoot.loc[(location_shoot['Lat']>40) & (location_shoot['Long'] < -60)]  



x_shoot = location_shoot['Long']

y_shoot = location_shoot['Lat']



# Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”

sns.jointplot(x_shoot, y_shoot, kind='scatter')

sns.jointplot(x_shoot, y_shoot, kind='hex')

sns.jointplot(x_shoot, y_shoot, kind='kde')
plt.figure(figsize=(20,10))

df['UCR_PART'].value_counts().plot.bar()

plt.show()
df[['Lat','Long']].describe()
location = df[['Lat','Long']]

location = location.dropna()



location = location.loc[(location['Lat']>40) & (location['Long'] < -60)]  
x = location['Long']

y = location['Lat']





colors = np.random.rand(len(x))



plt.figure(figsize=(20,20))

plt.scatter(x, y,c=colors, alpha=0.5)

plt.show()
m = folium.Map([42.348624, -71.062492], zoom_start=11)

m
x = location['Long']

y = location['Lat']





# Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”

sns.jointplot(x, y, kind='scatter')

sns.jointplot(x, y, kind='hex')

sns.jointplot(x, y, kind='kde')
#plt.figure(figsize=(20,20))



#map = Basemap(

#    projection='merc', 

#    lat_0 = 42.2, 

#    lon_0 = -70.9,

#    resolution = 'h', 

#    area_thresh = 0.1,

#    llcrnrlon=-70.8, 

#    llcrnrlat=42.2,

#    urcrnrlon=-71.5, 

#    urcrnrlat=42.5

#)

 

#map.drawcoastlines()

#map.drawcountries()

#map.fillcontinents(color = 'coral')

#map.drawmapboundary()



#lons = list(long[0:1000])

#lats = list(lat[0:1000])

#x,y = map(lons, lats)

#map.plot(x, y, 'bo', markersize=3)



#plt.show()
df.isnull().sum()
df['Day'] = 0
df['Night'] = 0
# Day or night for 1st month

df['Day'].loc[(df['MONTH'] == 1) & (df['HOUR'] >= 6) & (df['HOUR'] <= 18)] = 1



# Day or night for 2st month

df['Day'].loc[(df['MONTH'] == 2) & (df['HOUR'] >= 6) & (df['HOUR'] <= 19)] = 1



# for 3st month

df['Day'].loc[(df['MONTH'] == 3) & (df['HOUR'] >= 6) & (df['HOUR'] <= 20)] = 1



# Day or night for 4st month

df['Day'].loc[(df['MONTH'] == 4) & (df['HOUR'] >= 5) & (df['HOUR'] <= 20)] = 1



# Day or night for 5st month

df['Day'].loc[(df['MONTH'] == 5) & (df['HOUR'] >= 5) & (df['HOUR'] <= 21)] = 1



# Day or night for 6st month

df['Day'].loc[(df['MONTH'] == 6) & (df['HOUR'] >= 4) & (df['HOUR'] <= 21)] = 1



# Day or night for 7st month

df['Day'].loc[(df['MONTH'] == 7) & (df['HOUR'] >= 5) & (df['HOUR'] <= 21)] = 1



# Day or night for 8st month

df['Day'].loc[(df['MONTH'] == 8) & (df['HOUR'] >= 5) & (df['HOUR'] <= 21)] = 1



# Day or night for 9st month

df['Day'].loc[(df['MONTH'] == 9) & (df['HOUR'] >= 6) & (df['HOUR'] <= 20)] = 1



# Day or night for 10st month

df['Day'].loc[(df['MONTH'] == 10) & (df['HOUR'] >= 6) & (df['HOUR'] <= 19)] = 1



# Day or night for 11st month

df['Day'].loc[(df['MONTH'] == 11) & (df['HOUR'] >= 6) & (df['HOUR'] <= 17)] = 1



# Day or night for 12st month

df['Day'].loc[(df['MONTH'] == 12) & (df['HOUR'] >= 7) & (df['HOUR'] <= 17)] = 1
df['Night'].loc[df['Day']==0]=1
plt.figure(figsize=(16,8))

df['Night'].value_counts().plot.bar()

plt.show()
df['OFFENSE_CODE_GROUP'].value_counts().head(15)
list_offense_code_group = ('Motor Vehicle Accident Response',

                           'Larceny',

                           'Medical Assistance',

                           'Investigate Person',

                           'Other',

                           'Drug Violation',

                           'Simple Assault',

                           'Vandalism',

                           'Verbal Disputes',

                           'Towed',

                           'Investigate Property',

                           'Larceny From Motor Vehicle')
df_model = pd.DataFrame()
i = 0



while i < len(list_offense_code_group):



    df_model = df_model.append(df.loc[df['OFFENSE_CODE_GROUP'] == list_offense_code_group[i]])

    

    i+=1
list_column = ['DISTRICT','REPORTING_AREA','MONTH','DAY_OF_WEEK',

               'HOUR','Lat','Long', 'OFFENSE_CODE_GROUP','Day','Night']
df_model = df_model[list_column]
# DISTRICT



df_model['DISTRICT'] = df_model['DISTRICT'].map({

    'B3':1, 

    'E18':2, 

    'B2':3, 

    'E5':4, 

    'C6':5, 

    'D14':6, 

    'E13':7, 

    'C11':8, 

    'D4':9, 

    'A7':10, 

    'A1':11, 

    'A15':12

})



df_model['DISTRICT'].unique()
# REPORTING_AREA



df_model['REPORTING_AREA'] = pd.to_numeric(df_model['REPORTING_AREA'], errors='coerce')
# MONTH



df_model['MONTH'].unique()
# DAY_OF_WEEK



df_model['DAY_OF_WEEK'] = df_model['DAY_OF_WEEK'].map({

    'Tuesday':2, 

    'Saturday':6, 

    'Monday':1, 

    'Sunday':7, 

    'Thursday':4, 

    'Wednesday':3,

    'Friday':5

})



df_model['DAY_OF_WEEK'].unique()
# HOUR



df_model['HOUR'].unique()
# Lat, Long



df_model[['Lat', 'Long']].head()
df_model.fillna(0, inplace = True)
x = df_model[['DISTRICT','REPORTING_AREA','MONTH','DAY_OF_WEEK','HOUR','Lat','Long','Day','Night']]
y = df_model['OFFENSE_CODE_GROUP']
y.unique()
y = y.map({

    'Motor Vehicle Accident Response':1, 

    'Larceny':2, 

    'Medical Assistance':3,

    'Investigate Person':4, 

    'Other':5, 

    'Drug Violation':6, 

    'Simple Assault':7,

    'Vandalism':8, 

    'Verbal Disputes':9, 

    'Towed':10, 

    'Investigate Property':11,

    'Larceny From Motor Vehicle':12

})
# Split dataframe into random train and test subsets



X_train, X_test, Y_train, Y_test = train_test_split(

    x,

    y, 

    test_size = 0.1,

    random_state=42

)



print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.tree import ExtraTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.semi_supervised import LabelSpreading

from sklearn.svm import LinearSVC

from sklearn.neighbors.nearest_centroid import NearestCentroid

from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
def fun_results(result):

    print('mean: ' + str(result.mean()))

    print('max: ' + str(result.max()))

    print('min: ' + str(result.min()))

    return result
# DecisionTreeClassifier



def fun_DecisionTreeClassifier(X_train, Y_train):

    dec_tree = DecisionTreeClassifier()

    dec_tree = dec_tree.fit(X_train, Y_train)



    dec_tree_pred = dec_tree.predict(X_test)



    dec_tree_score = f1_score(Y_test, dec_tree_pred, average=None)

    return fun_results(dec_tree_score)



fun_DecisionTreeClassifier(X_train, Y_train)
# BernoulliNB



def fun_BernoulliNB(X_train, Y_train):

    bernoulli = BernoulliNB()

    bernoulli = bernoulli.fit(X_train, Y_train)



    bernoulli_pred = bernoulli.predict(X_test)



    bernoulli_score = f1_score(Y_test, bernoulli_pred, average=None)

    return fun_results(bernoulli_score)



fun_BernoulliNB(X_train, Y_train)
# ExtraTreeClassifier



def fun_ExtraTreeClassifier(X_train, Y_train):

    ext_tree = ExtraTreeClassifier()

    ext_tree = ext_tree.fit(X_train, Y_train)



    ext_tree_pred = ext_tree.predict(X_test)



    ext_tree_score = f1_score(Y_test, ext_tree_pred, average=None)

    return fun_results(ext_tree_score)



fun_ExtraTreeClassifier(X_train, Y_train)
# KNeighborsClassifier



def fun_KNeighborsClassifier(X_train, Y_train):

    neigh = KNeighborsClassifier()

    neigh.fit(X_train, Y_train) 



    neigh_pred = neigh.predict(X_test)



    neigh_score = f1_score(Y_test, neigh_pred, average=None)

    return fun_results(neigh_score)



fun_KNeighborsClassifier(X_train, Y_train)
# GaussianNB



def fun_GaussianNB(X_train, Y_train):

    gauss = GaussianNB()

    gauss = gauss.fit(X_train, Y_train)



    gauss_pred = gauss.predict(X_test)



    gauss_score = f1_score(Y_test, gauss_pred, average=None)

    return fun_results(gauss_score)



fun_GaussianNB(X_train, Y_train)
# RandomForestClassifier



def fun_RandomForestClassifier(X_train, Y_train):

    rfc = RandomForestClassifier()

    rfc = rfc.fit(X_train, Y_train)



    rfc_pred = rfc.predict(X_test)



    rfc_score = f1_score(Y_test, rfc_pred, average=None)

    return fun_results(rfc_score)



fun_RandomForestClassifier(X_train, Y_train)
# LGBMClassifier



def fun_LGBMClassifier(X_train, Y_train):

    clf = LGBMClassifier()

    clf.fit(X_train, Y_train)



    clf_pred = clf.predict(X_test)



    clf_score = f1_score(Y_test, clf_pred, average=None)

    return fun_results(clf_score)



fun_LGBMClassifier(X_train, Y_train)
df_model_2 = df[['OFFENSE_CODE', 'DISTRICT','MONTH','DAY_OF_WEEK','HOUR','Day','Night']]
df_model_2['OFFENSE_CODE'] = pd.to_numeric(df_model_2['OFFENSE_CODE'], errors='coerce')
# DISTRICT



df_model_2['DISTRICT'] = df_model_2['DISTRICT'].map({

    'B3':1, 

    'E18':2, 

    'B2':3, 

    'E5':4, 

    'C6':5, 

    'D14':6, 

    'E13':7, 

    'C11':8, 

    'D4':9, 

    'A7':10, 

    'A1':11, 

    'A15':12

})



df_model_2['DISTRICT'].unique()
# DAY_OF_WEEK



df_model_2['DAY_OF_WEEK'] = df_model_2['DAY_OF_WEEK'].map({

    'Tuesday':2, 

    'Saturday':6, 

    'Monday':1, 

    'Sunday':7, 

    'Thursday':4, 

    'Wednesday':3,

    'Friday':5

})



df_model_2['DAY_OF_WEEK'].unique()
df_model_2.isnull().sum()
df_model_2 = df_model_2.dropna()
df_model_2['DISTRICT'].unique()
df_model_2.shape
x = df_model_2[['OFFENSE_CODE','MONTH','DAY_OF_WEEK','HOUR','Day','Night']]

y = df_model_2['DISTRICT']
# Split dataframe into random train and test subsets



X_train, X_test, Y_train, Y_test = train_test_split(

    x,

    y, 

    test_size = 0.1,

    random_state=42

)



print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)
fun_DecisionTreeClassifier(X_train, Y_train)

fun_BernoulliNB(X_train, Y_train)
fun_ExtraTreeClassifier(X_train, Y_train)
fun_KNeighborsClassifier(X_train, Y_train)
fun_GaussianNB(X_train, Y_train)
fun_RandomForestClassifier(X_train, Y_train)
fun_LGBMClassifier(X_train, Y_train)
df_model3 = df[['DISTRICT','REPORTING_AREA', 'MONTH','DAY_OF_WEEK','HOUR','UCR_PART','Lat','Long']]
df_model3['DISTRICT'] = df_model3['DISTRICT'].map({

    'B3':1, 

    'E18':2, 

    'B2':3, 

    'E5':4, 

    'C6':5, 

    'D14':6, 

    'E13':7, 

    'C11':8, 

    'D4':9, 

    'A7':10, 

    'A1':11, 

    'A15':12

})
# REPORTING_AREA



df_model3['REPORTING_AREA'] = pd.to_numeric(df_model3['REPORTING_AREA'], errors='coerce')
# DAY_OF_WEEK



df_model3['DAY_OF_WEEK'] = df_model3['DAY_OF_WEEK'].map({

    'Tuesday':2, 

    'Saturday':6, 

    'Monday':1, 

    'Sunday':7, 

    'Thursday':4, 

    'Wednesday':3,

    'Friday':5

})
df_model3['UCR_PART'].unique()
df_model3['UCR_PART'] = df_model3['UCR_PART'].map({

    'Part Three':3, 

    'Part One':1, 

    'Part Two':2, 

#    'Other':4

})
df_model3 = df_model3.dropna()

print(df_model3.shape)

df_model3.isnull().sum()
x = df_model3[['DISTRICT','REPORTING_AREA', 'MONTH','DAY_OF_WEEK','HOUR','Lat','Long']]

y = df_model3['UCR_PART']
# Split dataframe into random train and test subsets



X_train, X_test, Y_train, Y_test = train_test_split(

    x,

    y, 

    test_size = 0.1,

    random_state=42

)



print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)
fun_DecisionTreeClassifier(X_train, Y_train)
fun_BernoulliNB(X_train, Y_train)
fun_ExtraTreeClassifier(X_train, Y_train)
fun_KNeighborsClassifier(X_train, Y_train)
fun_GaussianNB(X_train, Y_train)
fun_RandomForestClassifier(X_train, Y_train)
fun_LGBMClassifier(X_train, Y_train)
location.isnull().sum()
location.shape
x = location['Long']

y = location['Lat']



colors = np.random.rand(len(location))



plt.figure(figsize=(20,20))

plt.scatter(x, y,c=colors, alpha=0.5)

plt.show()
from sklearn.cluster import KMeans
X = location

X = X[~np.isnan(X)]
#K means Clustering #K means  

def doKmeans(X, nclust):

    model = KMeans(nclust)

    model.fit(X)

    clust_labels = model.predict(X)

    cent = model.cluster_centers_

    return (clust_labels, cent)



clust_labels, cent = doKmeans(X, 2)

kmeans = pd.DataFrame(clust_labels)

X.insert((X.shape[1]),'kmeans',kmeans)
#Plot the clusters obtained using k means#Plot the 

fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(111)

scatter = ax.scatter(X['Long'],X['Lat'],

                     c=kmeans[0],s=50)

ax.set_title('K-Means Clustering')

ax.set_xlabel('Long')

ax.set_ylabel('Lat')

plt.colorbar(scatter)
X = location

X = X[~np.isnan(X)]
clust_labels, cent = doKmeans(X, 3)

kmeans = pd.DataFrame(clust_labels)

X.insert((X.shape[1]),'kmeans',kmeans)
#Plot the clusters obtained using k means#Plot the 

fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(111)

scatter = ax.scatter(X['Long'],X['Lat'],

                     c=kmeans[0],s=50)

ax.set_title('K-Means Clustering')

ax.set_xlabel('Long')

ax.set_ylabel('Lat')

plt.colorbar(scatter)
X = location

X = X[~np.isnan(X)]
clust_labels, cent = doKmeans(X, 5)

kmeans = pd.DataFrame(clust_labels)

X.insert((X.shape[1]),'kmeans',kmeans)
#Plot the clusters obtained using k means#Plot the 

fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(111)

scatter = ax.scatter(X['Long'],X['Lat'],

                     c=kmeans[0],s=50)

ax.set_title('K-Means Clustering')

ax.set_xlabel('Long')

ax.set_ylabel('Lat')

plt.colorbar(scatter)
X = location

X = X[~np.isnan(X)]
clust_labels, cent = doKmeans(X, 10)

kmeans = pd.DataFrame(clust_labels)

X.insert((X.shape[1]),'kmeans',kmeans)
#Plot the clusters obtained using k means#Plot the 

fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(111)

scatter = ax.scatter(X['Long'],X['Lat'],

                     c=kmeans[0],s=50)

ax.set_title('K-Means Clustering')

ax.set_xlabel('Long')

ax.set_ylabel('Lat')

plt.colorbar(scatter)
df_clus = df[['OFFENSE_CODE','Long','Lat']]
df_clus = df_clus.loc[(df_clus['Lat'] > 40) & (df_clus['Long'] < -60)]
#df_clus['REPORTING_AREA'] = pd.to_numeric(df_model['REPORTING_AREA'], errors='coerce')
df_clus = df_clus.dropna()
df_clus.describe()
X = df_clus

X = X[~np.isnan(X)]
clust_labels, cent = doKmeans(X, 2)

kmeans = pd.DataFrame(clust_labels)

X.insert((X.shape[1]),'kmeans',kmeans)
#Plot the clusters obtained using k means#Plot the 

fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(111)

scatter = ax.scatter(X['Long'],X['Lat'],

                     c=kmeans[0],s=5)

ax.set_title('K-Means Clustering')

ax.set_xlabel('Long')

ax.set_ylabel('Lat')

plt.colorbar(scatter)
df_clus = df[['MONTH','Long','Lat']]
df_clus = df_clus.loc[(df_clus['Lat'] > 40) & (df_clus['Long'] < -60)]
df_clus = df_clus.dropna()
X = df_clus

X = X[~np.isnan(X)]
clust_labels, cent = doKmeans(X, 2)

kmeans = pd.DataFrame(clust_labels)

X.insert((X.shape[1]),'kmeans',kmeans)
#Plot the clusters obtained using k means#Plot the 

fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(111)

scatter = ax.scatter(X['Long'],X['Lat'],

                     c=kmeans[0],s=5)

ax.set_title('K-Means Clustering')

ax.set_xlabel('Long')

ax.set_ylabel('Lat')

plt.colorbar(scatter)