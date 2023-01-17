import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # used for plot interactive graph.

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



GPStore = pd.read_csv("../input/googleplaystore.csv")

GPStore     # reads the Google Play store dataset
GPStore['Rating'].describe()  # Get the ratings 
# Get the size 

GPStore['Size'].head()
# GEt the installs

GPStore['Installs'].head()
# Number of genres

GPStore['Genres'].value_counts()
GPStore['Main Genre'] = GPStore['Genres'].apply(lambda x: x.split(';')[0])

GPStore['Sub Genre'] = GPStore['Genres'].apply(lambda x: x.split(';')[1] if len(x.split(';')) > 1 else 'no sub genre')



plt.figure(figsize=(10,5))

GPStore['Main Genre'].value_counts().plot(kind='bar')
# Compare whats free and not free

GPStore['Type'].value_counts().plot(kind='bar')
# Count of each category

GPStore.Category.value_counts().plot(kind='barh',figsize= (12,8))
# Count of each category

catCount = sns.countplot(x="Category",data=GPStore, palette = "Set1")

catCount.set_xticklabels(catCount.get_xticklabels(), rotation=90, ha="right")

catCount 

plt.title('Count of each app category',size = 20)
# total count of ratings

plt.figure(figsize=(8, 15))

sns.countplot(y='Rating',data=GPStore )

plt.show()
# Contnt rating by each categorey

sns.barplot(x='Content Rating', y='Rating', data=GPStore)
# Error that needs to be fixed!

plt.figure(figsize = (10,10))

sns.regplot(x="Installs", y="Rating", color = 'teal',data=GPStore)

plt.title('Rating VS Installs',size = 20)
# Find missing data to see if any has NaN values.

# Which finds most missing values for "rating". 

plt.figure(figsize=(7,5))

sns.heatmap(GPStore.isnull(), cmap='viridis')

GPStore.isnull().any()
GPStore.isnull().sum() # Clearer stat to handle missing data
# This doesn't look right at all 



corr = GPStore.corr()

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot=True)
GPStore.columns
GPStore.head(3)
from sklearn import tree

import sys

import re



# The best way to fill missing values might be using the median instead of mean.

GPStore['Rating'] = GPStore['Rating'].fillna(GPStore['Rating'].median())



# Before filling null values we have to clean all non numerical values & unicode charachters 

replaces = [u'\u00AE', u'\u2013', u'\u00C3', u'\u00E3', u'\u00B3', '[', ']', "'"  ]

for i in replaces:

    GPStore['Current Ver'] = GPStore['Current Ver'].astype(str).apply(lambda x : x.replace(i, ''))



regex = [r'[-+|/:/;(_)@]', r'\s+', r'[A-Za-z]+']

for j in regex:

    GPStore['Current Ver'] = GPStore['Current Ver'].astype(str).apply(lambda x : re.sub(j, '0', x))



GPStore['Current Ver'] = GPStore['Current Ver'].astype(str).apply(lambda x : x.replace('.', ',',1).replace('.', '').replace(',', '.',1)).astype(float)

GPStore['Current Ver'] = GPStore['Current Ver'].fillna(GPStore['Current Ver'].median())
GPStore['Category'].unique() # Counts the number of uniuque values in category columns
# Checks the record  of unreasonable value which is 1.9

i = GPStore[GPStore['Category'] == '1.9'].index

GPStore.loc[i]



# Drop this bad column

GPStore = GPStore.drop(i)



# Removes NaN values

GPSTore = GPStore[pd.notnull(GPStore['Last Updated'])]

GPStore = GPStore[pd.notnull(GPStore['Content Rating'])]
# Trying to encode App values 

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

GPStore['App'] = le.fit_transform(GPStore['App']) # Encoder converts the values into numeric values

# Category features encoding

category_list = GPStore['Category'].unique().tolist() 

category_list = ['cat_' + word for word in category_list]

GPStore = pd.concat([GPStore, pd.get_dummies(GPStore['Category'], prefix='cat')], axis=1)
# Genres encoding fot the features

le = preprocessing.LabelEncoder()

GPStore['Genres'] = le.fit_transform(GPStore['Genres'])
# Encode the "Content Rating" 

le = preprocessing.LabelEncoder()

GPStore['Content Rating'] = le.fit_transform(GPStore['Content Rating'])
# Price 

GPStore['Price'] = GPStore['Price'].apply(lambda x : x.strip('$'))
# Installs 

GPStore['Installs'] = GPStore['Installs'].apply(lambda x : x.strip('+').replace(',', ''))
# Type encoding

GPStore['Type'] = pd.get_dummies(GPStore['Type'])
# Encoding th Last Updated

import time

import datetime

GPStore['Last Updated'] = GPStore['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))
# Convert kbytes to Mbytes

#selecting all k values from the "Size" column and replace those values by 

# their corresponding M values, and since k indices belong to a list of 

# non-consecutive numbers, a new dataframe (converter) will be created with 

# these k indices to perform the conversion, then the final values will be 

# assigned back to the "Size" column.



k_indices = GPStore['Size'].loc[GPStore['Size'].str.contains('k')].index.tolist()

converter = pd.DataFrame(GPStore.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x / 1024).apply(lambda x: round(x, 3)).astype(str))

GPStore.loc[k_indices,'Size'] = converter
# Size cleaning

GPStore['Size'] = GPStore['Size'].apply(lambda x: x.strip('M'))

GPStore[GPStore['Size'] == 'Varies with device'] = 0

GPStore['Size'] = GPStore['Size'].astype(float)
# Predict app ratings based on the other matrices

# Split data into training and testing sets

attributes = ['App', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver']

attributes.extend(category_list)

X = GPStore[attributes]

y = GPStore['Rating']



# Should splits the dataset into 85% train data and 25% test data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)
# Looks at the 15 closest neighbors

from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=15)



# Finds the mean accuracy of KNN regression using X_test and y_test

model.fit(X_train, y_train)
# Calculates the mean accuracy of the KNN model

acc = model.score(X_test,y_test)  # accuracy

'Accuracy: ' + str(np.round(acc*100, 2)) + '%'
# Random Forest Model

from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_jobs=-1)



# Try different numbers of n_estimators - this will take a minute or so

estimators = np.arange(10, 200, 10)

scores = []

for n in estimators:

    model.set_params(n_estimators=n)

    model.fit(X_train, y_train)

    scores.append(model.score(X_test, y_test))

plt.figure(figsize=(7, 5))

plt.title("The Effect of the estimate")

plt.xlabel("# of estimator")

plt.ylabel("score")

plt.plot(estimators, scores)

results = list(zip(estimators,scores))

results
from sklearn import metrics



predictions = model.predict(X_test)

'Mean Error:', metrics.mean_absolute_error(y_test, predictions)
# The mean squared error percentage

'Mean Squared Error:', metrics.mean_squared_error(y_test, predictions)
# The Root mean

'Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions))