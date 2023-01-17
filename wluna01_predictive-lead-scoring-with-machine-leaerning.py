# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np

import missingno as mno

from sklearn import preprocessing

from sklearn.model_selection import train_test_split





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print("Setup Complete")

# Any results you write to the current directory are saved as output.
#Load data

sales_data_filepath = "../input/individual-company-sales-data/sales_data.csv"

sales_data = pd.read_csv(sales_data_filepath, index_col=0, encoding="latin-1")

sales_data.sample(10)
#Visualize null data in array

mno.matrix(sales_data)
#delete marriage column, since it has so many missing data values

sales_data = sales_data.drop('marriage', axis=1)

sales_data.head()
#what are the unique values in education

sales_data['education'].unique()
#column deletions

sales_data = sales_data.drop(['customer_psy','child', 'house_owner', 'fam_income'], axis=1)
#create a new column for the flag

sales_data['flag'] = sales_data.index

#replace the old values in the index with ascending integers

sales_data.index = np.arange(len(sales_data))

#rename index using the rename_axis method

sales_data = sales_data.rename_axis('customerID')

sales_data.head()
#sales_data.iloc[[2],[0]] = "M"

#print(sales_data.iloc[[2],[0]])
#sales_data['flag'] = sales_data.index

#print(sales_data['flag'])
#clean strings while iterating through each row

'''

for index, row in sales_data.iterrows():

    if (row['education'] == '4. Grad'):

        row['education'] = '4'

    elif (row['education'] == '3. Bach'):

        row['education'] = '3'

    elif (row['education'] == '2. Some College'):

        row['education'] = '2'

       #print(row)

'''

#The above code appears to work when I print out the rows, but when I look at sales_data.head() again

# nothing seems to have changed. Credit to Rish Patel for showing me replace.



#On education

sales_data.replace('0. <HS', 'dropout', inplace=True)

sales_data.replace('1. HS', 'hs', inplace=True)

sales_data.replace('2. Some College', 'associates', inplace=True)

sales_data.replace('3. Bach', 'bachelors', inplace=True)

sales_data.replace('4. Grad', 'masters', inplace=True)



#On age

sales_data.replace('1_Unk', '1', inplace=True)

sales_data.replace('2_<=25', '2', inplace=True)

sales_data.replace('3_<=35', '3', inplace=True)

sales_data.replace('4_<=45', '4', inplace=True)

sales_data.replace('5_<=55', '5', inplace=True)

sales_data.replace('6_<=65', '6', inplace=True)

sales_data.replace('7_>65', '7', inplace=True)



#On mortgage

sales_data.replace('1Low', 'low', inplace=True)

sales_data.replace('2Med', 'medium', inplace=True)

sales_data.replace('3High', 'high', inplace=True)
#There are two ways to address the null values in the education column. You can either fill them or drop the entire row

# Uncomment the one you prefer (I'm opting to drop rows)



#1.Replace null values

#sales_data['education'].fillna('none', inplace=True)

#2.drop all columns that have a null value

sales_data = sales_data.dropna()



#Two different ways to check if data contains empty values

print('Columns with null values:\n', sales_data.isnull().sum())

print("-"*10)
#Split data into testing and training sets

#This divides the data by absolute number of entries. However we want to divide by percent.

'''

train_data = sales_data[:200]

test_data = sales_data[200:]

'''

#train_test_split function is imported from sklearn. 

#test_size set so that 80% of data is used for training

train_data, test_data = train_test_split(sales_data,test_size=0.2)
def encode_features(df_train, df_test):

    features = ['gender', 'education','house_val','age','online','occupation','mortgage','region','car_prob']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test



train_data, test_data = encode_features(train_data, test_data)
#I want to see if all the data was successfully converted to numerical:

test_data.info()
#Separate data into features(x) and target(y)

x_all = train_data.drop(['flag'], axis=1)

y_all = train_data['flag']
#We're using train_test_split for the second time, but this time with additional parameters

num_test = 0.3

X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=100)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 

rfc = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(rfc, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

rfc = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

rfc.fit(X_train, y_train)
rfc_prediction = rfc.predict(X_test)

rfc_score=accuracy_score(y_test, rfc_prediction)

print(rfc_score)