import unicodecsv
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from PIL import Image
from scipy.stats import ttest_ind


baltimore = pd.read_csv('../input/BPD_Part_1_Victim_Based_Crime_Data.csv')
baltimore.head()
baltimore.isnull().sum()
baltimore['Weapon'].value_counts(dropna=False)
baltimore.fillna(method='ffill', inplace=True)
baltimore['Weapon'].value_counts(dropna=True)
baltimore['Premise'].value_counts(dropna=False)
baltimore['Neighborhood'].value_counts(dropna=False)
baltimore.fillna(method='ffill', inplace=True)
baltimore['District'].value_counts(dropna=False)
baltimore.fillna(method='ffill', inplace=True)
baltimore['Post'].value_counts(dropna=False)
baltimore.fillna(method='ffill', inplace=True)
baltimore['Inside/Outside'].value_counts(dropna=False)
def change_category(InsideOutside):
    if InsideOutside=='Inside':
        return ('I')
    elif InsideOutside=='Outside':
        return ('O')
    elif InsideOutside=='O':
        return ('O')
    elif InsideOutside=='I':
        return('I')
baltimore['InsideOutside'] = baltimore['Inside/Outside'].apply(change_category)
baltimore['InsideOutside'].value_counts(dropna=False)
baltimore.head()
baltimore['Location'].unique()
baltimore['Description'].unique()
baltimore['Weapon'].unique()
baltimore['Post'].unique()
def create_list_number_crime(name_column, list_unique):
    # list_unique = df[name_column].unique()
    
    
    i = 0
    
    list_number = list()
    
    while i < len(list_unique):
        list_number.append(len(baltimore.loc[baltimore[name_column] == list_unique[i]]))
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
for n in range(0, 276528):
    x = baltimore.loc[n, 'CrimeDate']
    baltimore.loc[n,'CrimeYear'] = int(x[6:])
crime_year = baltimore.CrimeYear.value_counts()
crime_yearindex = crime_year.sort_index(axis=0, ascending=True)
print(crime_yearindex)
bar_chart(crime_yearindex,crime_year.index)
pie_plot(crime_yearindex.index, crime_yearindex)

# Create a new column that has the hour at which the crime occurred

for n in range(0, 276528):
    x = baltimore.loc[n, 'CrimeTime']
    baltimore.loc[n,'CrimeHour'] = int(x[:2])
# There is only one occurrence of one crime at hour 24. For 24-hour format, midnight can be described as either 24:00
# or 00:00, so we will change the observation from 24 to 0

print(baltimore[baltimore['CrimeHour'] == 24])
baltimore.at[239894, 'CrimeHour'] = 0
print(baltimore.loc[239894])


# Incorporate the change of the observation

crime_hour = baltimore.CrimeHour.value_counts()
crime_hourindex = crime_hour.sort_index(axis=0, ascending=True)
# Create line plot that shows the crime occurrence by hour

fig = plt.figure(figsize=(20, 20))
f, ax = plt.subplots(1)
xdata = crime_hourindex.index
ydata = crime_hourindex
ax.plot(xdata, ydata)
ax.set_ylim(ymin=0, ymax=17000)
ax.set_xlim(xmin=0, xmax=24)
plt.xlabel('Hour')
plt.ylabel('Number of Crimes')
plt.title('Number of Crimes by Hour')
plt.show(f)
pie_plot(crime_hourindex.index, crime_hourindex)

bar_chart(crime_hourindex,crime_hourindex.index)
# Create a pivot table to identify if hours had more occurence of specific crime categories

baltimore.pivot_table(index='Description',
               columns='CrimeHour',
               values='CrimeTime',
               aggfunc= 'count')
#Create a dataframe that has the number of crime occurences by district

districtcount = baltimore.District.value_counts()
baltimore.District.value_counts()
#Create bar graph of number of crimes by district

my_colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
districtcount.plot(kind='bar',
                color=my_colors,
                title='Number of Crimes Committed by District')
#Create a dataframe that has the occurrence of crimes by category

crimecount = baltimore.Description.value_counts()
baltimore.Description.value_counts()
#Create bar graph of number of crimes by category

my_colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
crimecount.plot(kind='bar',
                color=my_colors,
                title='Crimes Committed by Category')
# visualization
#map out the data and visualize where the specific crimes occur by using the Longitude and Latitude datapoints, 
#this can help visualize crime cluster locations
from mpl_toolkits.basemap import Basemap
import folium
from folium import plugins
m = folium.Map([39.2903848, -76.6121893], zoom_start=11)
m
baltimore[['Longitude','Latitude']].describe()
location = baltimore[['Longitude','Latitude']]
location = location.dropna()

location = location.loc[(location['Latitude']>30) & (location['Longitude'] < -60)]

x = baltimore['Longitude']
y = baltimore['Latitude']



colors = np.random.rand(len(x))

plt.figure(figsize=(20,20))
plt.scatter(x, y,c=colors, alpha=0.5)
plt.show()
x = baltimore['Longitude']
y = baltimore['Latitude']


# Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”
sns.jointplot(x, y, kind='scatter')
sns.jointplot(x, y, kind='hex')
sns.jointplot(x, y, kind='kde')
list_column = ['description','Weapon', 'Post','District','Longitude','Latitude']
list_column
df_model = pd.DataFrame()
list_offense_code_group = ('ROBBERY - RESIDENCE',
                           'AUTO THEFT',
                           'SHOOTING',
                           'AGG. ASSAULT',
                           'COMMON ASSAULT',
                           'BURGLARY',
                           'HOMICIDE',
                           'ROBBERY - STREET',
                           'ROBBERY - COMMERCIAL',
                           'LARCENY',
                           'LARCENY FROM AUTO',
                           'ARSON',
                          'ROBBERY - CARJACKING',
                          'ASSAULT BY THREAT',
                          'RAPE')
i = 0

while i < len(list_offense_code_group):

    df_model = df_model.append(baltimore.loc[baltimore['Description'] == list_offense_code_group[i]])
    
    i+=1
def change_category_num(description):
    if description=='ROBBERY - RESIDENCE':
        return 1
    elif description=='AUTO THEFT':
        return 2
    elif description=='SHOOTING':
        return 3
    elif description=='AGG. ASSAULT':
        return 4
    elif description=='COMMON ASSAULT':
        return 5
    elif description=='BURGLARY':
        return 6
    elif description=='HOMICIDE':
        return 7
    elif description=='ROBBERY - STREET':
        return 8
    elif description=='ROBBERY - COMMERCIAL':
        return 9
    elif description=='LARCENY':
        return 10
    elif description=='LARCENY FROM AUTO':
        return 11
    elif description=='ARSON':
        return 12
    elif description=='ROBBERY - CARJACKING':
        return 13
    elif description=='ASSAULT BY THREAT':
        return 14
    elif description=='RAPE':
        return 15

baltimore['description'] = baltimore['Description'].apply(change_category_num)
baltimore['description'].unique()
y.unique()
# DISTRICT

df_model['District'] = df_model['District'].map({
    'SOUTHERN':1, 
    'CENTRAL':2, 
    'NORTHERN':3, 
    'SOUTHEASTERN':4, 
    'NORTHWESTERN':5, 
    'EASTERN':6, 
    'SOUTHWESTERN':7, 
    'NORTHEASTERN':8, 
    'WESTERN':9, 
    'nan':10, 
    
})

df_model['District'].unique()
# Weapon

df_model['Weapon'] = df_model['Weapon'].map({
    'KNIFE':1, 
    'nan':2, 
    'FIREARM':3, 
    'OTHER':4, 
    'HANDS':5, 
   
})

df_model['Weapon'].unique()
# Lat, Long

df_model[['Latitude', 'Longitude']].head()
# POST

df_model['Post'].unique()
df_model.fillna(0, inplace = True)
x = df_model[['Weapon', 'Post','District','Longitude','Latitude']]
baltimore['description'].unique()
df_model = df_model[list_column]
y=df_model['description']
y.unique()
# Split dataframe into random train and test subsets
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import train_test_split

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
def fun_results(result):
    print('mean:' + str(result.mean()))
    print('max: ' + str(result.max()))
    print('min: ' + str(result.min()))
    return result
# DecisionTreeClassifier

def fun_DecisionTreeClassifier(X_train, Y_train):
    dec_tree = DecisionTreeClassifier()
    dec_tree = dec_tree.fit(X_train, Y_train)

    dec_tree_pred = dec_tree.predict(X_test)
     
    dec_tree_score = f1_score(Y_test, dec_tree_pred, average=None)
    print(dec_tree_pred)
    return fun_results(dec_tree_score)

fun_DecisionTreeClassifier(X_train, Y_train)
# BernoulliNB

def fun_BernoulliNB(X_train, Y_train):
    bernoulli = BernoulliNB()
    bernoulli = bernoulli.fit(X_train, Y_train)

    bernoulli_pred = bernoulli.predict(X_test)

    bernoulli_score = f1_score(Y_test, bernoulli_pred, average=None)
    print(bernoulli_pred)
    return fun_results(bernoulli_score)

fun_BernoulliNB(X_train, Y_train)

