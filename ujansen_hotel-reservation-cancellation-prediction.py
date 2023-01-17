# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Importing the dataset

dataset = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

dataset
# Exploring the dataset

print(dataset.info(), '\n\n')

print('Information about null values in dataset\n')

print(dataset.isnull().sum())
# Cleaning the dataset

replacements = {'children': 0.0, 'country': 'Unknown', 'agent': 0.0,

                'company': 0.0}

dataset.fillna(replacements, inplace = True)
dataset['meal'].value_counts()
# Since meal has Undefined values, it was changed to SC

dataset['meal'].replace(to_replace = 'Undefined', value = 'SC', inplace = True)
# Checking if there are rows with 0 guests

no_guests = list(dataset.loc[dataset['adults']

                   + dataset['children']

                   + dataset['babies'] == 0].index)

no_guests
# Since rows with 0 guests make no sense, those rows were dropped

dataset.drop(dataset.index[no_guests], inplace = True)
# Separating the resort and city hotel information

resort = dataset.loc[(dataset['hotel'] == 'Resort Hotel')]

city = dataset.loc[(dataset['hotel'] == 'City Hotel')]
# Finding the cancellation for each hotel per month

resort_per_month = resort.groupby('arrival_date_month')['hotel'].count()

resort_cancel_per_month = resort.groupby('arrival_date_month')['is_canceled'].sum()



city_per_month = city.groupby('arrival_date_month')['hotel'].count()

city_cancel_per_month = city.groupby('arrival_date_month')['is_canceled'].sum()



resort_cancel_data = pd.DataFrame({'Hotel': 'Resort Hotel',

                                'Month': list(resort_per_month.index),

                                'Bookings': list(resort_per_month.values),

                                'Cancelations': list(resort_cancel_per_month.values)})



city_cancel_data = pd.DataFrame({'Hotel': 'City Hotel',

                                'Month': list(city_per_month.index),

                                'Bookings': list(city_per_month.values),

                                'Cancelations': list(city_cancel_per_month.values)})



full_cancel_data = pd.concat([resort_cancel_data, city_cancel_data], ignore_index = True)

full_cancel_data['cancel_percent'] = full_cancel_data['Cancelations'] / full_cancel_data['Bookings'] * 100



ordered_months = ['January', 'February', 'March', 'April', 'May', 'June', 

          'July', 'August', 'September', 'October', 'November', 'December']

full_cancel_data['Month'] = pd.Categorical(full_cancel_data['Month'], categories = ordered_months, ordered = True)
# Plot figure

plt.figure(figsize = (12, 8))

sns.barplot(x = 'Month', y = 'cancel_percent' , hue = 'Hotel',

            hue_order = ['City Hotel', 'Resort Hotel'], data = full_cancel_data)

plt.title('Cancelations per month', fontsize = 16)

plt.xlabel('Month', fontsize = 16)

plt.xticks(rotation = 45)

plt.ylabel('Cancelations [%]', fontsize = 16)

plt.legend(loc = 'upper right')

plt.show()
# Finding the percentage of cancellations for each hotel

resort_cancellations = resort['is_canceled'].sum()

resort_percentage_cancel = round(resort_cancellations / resort.shape[0] * 100, 2)

city_cancellations = city['is_canceled'].sum()

city_percentage_cancel = round(city_cancellations / city.shape[0] * 100, 2)



print('% of cancellations for Resort Hotel is {0}%'.format(resort_percentage_cancel))

print('% of cancellations for City Hotel is {0}%'.format(city_percentage_cancel))
# The above data begs the question whether the pricing of each hotel plays a part in cancellation or not

# Checking to see if each room has adults or not since it is plausible that adults would pay most of the room fare



# Checking to see if each room has adults

index_city = list(city.loc[city['adults'] == 0].index)

index_resort = list(resort.loc[resort['adults'] == 0].index)



print(index_city)

print('\n')

print(index_resort)
# Since there are quite a few rows with no adults, it is safe to assume that adults and children pay the room fare



# Finding the average price per person in each hotel

resort_avg_price = round((resort['adr'] / (resort['adults'] + resort['children'])).mean(), 2)

city_avg_price = round((city['adr'] / (city['adults'] + city['children'])).mean(), 2)



print('The average price per person at Resort Hotel is € {0}'.format(resort_avg_price))

print('The average price per person at City Hotel is € {0}'.format(city_avg_price))
# Finding the correlation

correlation = dataset.corr()['is_canceled']

print(correlation)
# Dropping irrelevant columns

dataset.drop(['arrival_date_year', 'assigned_room_type', 'booking_changes', 'reservation_status', 

              'country', 'reservation_status_date', 'days_in_waiting_list'], axis = 1, inplace = True)
dataset.info()
# Making the numerical features and categorical features for one-hot encoding and simple imputing

numerical_features = list(dataset.select_dtypes(exclude = [object]))

categorical_features = list(dataset.select_dtypes(include = [object]))

numerical_features.remove('is_canceled')
# Changing the dataset into dependent and independent variables

X = dataset.drop(['is_canceled'], axis = 1)[numerical_features + categorical_features]

y = dataset['is_canceled']
# Pre-processing features

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

numeric_transformer = SimpleImputer(strategy = 'constant')

categoric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'Unknown')),

    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))])

preprocessor = ColumnTransformer(transformers = [('numeric', numeric_transformer, numerical_features),

                                               ('categorical', categoric_transformer, categorical_features)])



X = preprocessor.fit_transform(X)
# Splitting the dataset into training and testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Create and fit the classifier on the training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 300, n_jobs = -1, random_state = 0)

classifier.fit(X_train, y_train)
# Make predictions using the classifier

y_pred = classifier.predict(X_test)
# Evaluate the performance of the classifier

from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

report = classification_report(y_test, y_pred)



print('The accuracy of the classifier is {0}%'.format(accuracy))

print('\nThe calculated RMSE is {0}'.format(rmse))

print('\nThe classification report is as follows:\n')

print(report)
# Plotting the confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (10, 8))

sns.heatmap(cm, annot = True, fmt = '.0f', linewidths = .5, square = True)

plt.xlabel('Predicted labels')

plt.ylabel('Actual labels')

plt.title('Accuracy: {0}'.format(round(accuracy, 2)))

plt.show()