# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn import model_selection

from sklearn import linear_model 

from sklearn import neural_network 

from sklearn import ensemble



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_set = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')

test_set = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip')

stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')

features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
test_set.info()
train_set.info()
stores.info()
features.info()
#Selecting the features which have missing values

features_nan = features.iloc[:, (features.isna().sum() > 0).values].columns



#Calculating the percentage of the missing values

percentage_nan = (features.isna().sum()/features.shape[0])*100





#plotting those values

fig, ax = plt.subplots(figsize=(10, 5))

ax.axhline(y=50, color="red", linestyle="--")

ax.bar(features_nan, percentage_nan[features_nan].values)

ax.set_ylabel('Percentage of Missing Values', fontsize=13)

ax.set_xlabel('Features which have Missing Values', fontsize=13)

ax.set_title('Features set Missing Values Analysis', fontsize=16)

#ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
training_data = train_set.merge(stores).merge(features).sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)

test_data = test_set.merge(stores).merge(features).sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)



del stores, features, train_set, test_set



Y = training_data['Weekly_Sales']
training_data.info()
test_data.info()
#Selecting the features which have missing values

training_data_nan = training_data.iloc[:, (training_data.isna().sum() > 0).values].columns



#Calculating the percentage of the missing values

percentage_nan = (training_data.isna().sum()/training_data.shape[0])*100





#plotting those values

fig, ax = plt.subplots(figsize=(10, 5))

ax.axhline(y=50, color="red", linestyle="--")

ax.bar(training_data_nan, percentage_nan[training_data_nan].values)

ax.set_ylabel('Percentage of Missing Values', fontsize=13)

ax.set_xlabel('Features which have Missing Values', fontsize=13)

ax.set_title('New Training Set Missing Values Analysis', fontsize=16)

#ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
#Selecting the features which have missing values

test_data_nan = test_data.iloc[:, (test_data.isna().sum() > 0).values].columns



#Calculating the percentage of the missing values

percentage_nan = (test_data.isna().sum()/test_data.shape[0])*100





#plotting those values

fig, ax = plt.subplots(figsize=(10, 5))

ax.axhline(y=50, color="red", linestyle="--")

ax.bar(test_data_nan, percentage_nan[test_data_nan].values)

ax.set_ylabel('Percentage of Missing Values', fontsize=13)

ax.set_xlabel('Features which have Missing Values', fontsize=13)

ax.set_title('New Test Set Missing Values Analysis', fontsize=16)

#ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
all_features = training_data.columns

categorical_features = ['Store', 'Dept', 'Date', 'IsHoliday', 'Type']

target_value = 'Weekly_Sales'

numeric_features = all_features.drop(categorical_features)

numeric_features = numeric_features.drop(target_value)
training_data[numeric_features].hist(figsize=(12,8))

plt.tight_layout()

plt.show()
#Selecting the CPI values

CPI = training_data['CPI'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(CPI, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('CPI', fontsize=13)
#Selecting the Fuel Price values

Fuel_Price = training_data['Fuel_Price'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(Fuel_Price, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('Fuel Price', fontsize=13)
#Selecting the MArkDown1 values

Mkd1 = training_data['MarkDown1'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(Mkd1, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('MarkDown1', fontsize=13)
#Selecting the MarkDown2 values

Mkd2 = training_data['MarkDown2'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(Mkd2, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('MarkDown2', fontsize=13)
#Selecting the MarkDown3 values

Mkd3 = training_data['MarkDown3'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(Mkd3, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('MarkDown3', fontsize=13)
#Selecting the MarkDown4 values

Mkd4 = training_data['MarkDown4'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(Mkd4, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('MarkDown4', fontsize=13)
#Selecting the CPI values

Mkd5 = training_data['MarkDown5'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(Mkd5, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('MarkDown5', fontsize=13)
#Selecting the Size values

Size = training_data['Size'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(Size, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('Size', fontsize=13)
#Selecting the Temperature values

Temperature = training_data['Temperature'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(Temperature, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('Temperature', fontsize=13)
#Selecting the Unemployment values

Unemployment = training_data['Unemployment'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(Unemployment, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('Unemployment', fontsize=13)
fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(training_data[np.append(numeric_features, target_value)].corr(method='pearson'), annot=True, 

            fmt='.2f', cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)

ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
sns.set(font_scale=1)

plt.figure(figsize=(20, 5))

sns.countplot(training_data['Store'], color='gray')
sns.set(font_scale=1)

plt.figure(figsize=(20,5))

sns.countplot(training_data['Dept'], color='gray')
sns.set(font_scale=1)

plt.figure(figsize=(20, 5))

chart = sns.countplot(training_data['Date'], color='gray')

chart.set(xticklabels=[])
sns.set(font_scale=1)

plt.figure(figsize=(20, 5))

sns.countplot(training_data['IsHoliday'], color='gray')
sns.set(font_scale=1)

plt.figure(figsize=(20, 5))

sns.countplot(training_data['Type'], color='gray')
#Selecting the Stores values

Stores = training_data['Store'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(Stores, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('Store', fontsize=13)
#Selecting the Dept values

Dept = training_data['Dept'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(Dept, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('Dept', fontsize=13)
#Selecting the Date values

Date = training_data['Date'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(Date, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('Date', fontsize=13)

ax.set_xticklabels([])
great_sales = training_data[training_data['Weekly_Sales'] > 300000]

great_sales[['Date', 'Dept', 'IsHoliday']]
#Selecting the IsHoliday values

IsHoliday = training_data['IsHoliday'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(IsHoliday, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('IsHoliday', fontsize=13)
#Selecting the Tyoe values

Type = training_data['Type'].values

Weekly_Sales = training_data['Weekly_Sales'].values



#plotting the relation

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(Type, Weekly_Sales)

ax.set_ylabel('Weekly Sales', fontsize=13)

ax.set_xlabel('Type', fontsize=13)
#Training_Data

training_data['Week'] = pd.to_datetime(training_data['Date']).dt.week

training_data['Year'] = pd.to_datetime(training_data['Date']).dt.year

training_data = training_data.drop(columns='Date')



#Test_Data

test_data['Week'] = pd.to_datetime(test_data['Date']).dt.week

test_data['Year'] = pd.to_datetime(test_data['Date']).dt.year



#We are going to need for the submission

Date_list = [str(x) for x in test_data['Date']]



test_data = test_data.drop(columns='Date')



#The Categorical_feature has changed

categorical_features = ['Store', 'Dept', 'Year', 'Week', 'IsHoliday', 'Type']
Types = np.unique(training_data['Type'])

TypeOrdinal = preprocessing.LabelEncoder()

TypeOrdinal.fit(Types)

training_data['Type'] = TypeOrdinal.transform(training_data['Type'])

test_data['Type'] = TypeOrdinal.transform(test_data['Type'])



Holidays = np.unique(training_data['IsHoliday'])

IsHolidayOrdinal = preprocessing.LabelEncoder()

IsHolidayOrdinal.fit(Holidays)

training_data['IsHoliday'] = IsHolidayOrdinal.transform(training_data['IsHoliday'])

test_data['IsHoliday'] = IsHolidayOrdinal.transform(test_data['IsHoliday'])
training_data = training_data.fillna(training_data.median())

test_data = test_data.fillna(test_data.median())
#Separating Between features and target value

Y = training_data[target_value]

X = training_data.drop(columns=target_value)



#Removing the Weekly Sales

training_data_features = X.columns

features_without_holiday = training_data_features.drop('IsHoliday') #We are not going to scale the holliday



scaler = preprocessing.StandardScaler().fit(training_data[features_without_holiday])

X_test = test_data.copy()



X[features_without_holiday] = scaler.transform(X[features_without_holiday])

X_test[features_without_holiday] = scaler.transform(test_data[features_without_holiday])



X.head()



#training_data_scaled = training_data.copy()

#training_data_scaled = training_data_scaled.drop(columns='Weekly_Sales')

#test_data_scaled = test_data.copy()
fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(training_data.corr(method='pearson'), annot=True, 

            fmt='.2f', cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)

ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
features_drop = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Fuel_Price']



X = X.drop(columns=features_drop)

X_test = X_test.drop(columns=features_drop)



X.head()
X_test
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, Y, 

                                                                                test_size = 0.2, random_state = 42)
def WMAE(holiday, y_hat, y):

    W = np.ones(y_hat.shape)

    

    W[holiday == 1] = 5

    

    metric = (1/np.sum(W))*np.sum(W*np.abs(y-y_hat))

    

    return metric
#List of alphas to be tested

alphas = np.logspace(-6, 6, 13)

WMAE_list_val = []

WMAE_list_tra = []



for i in alphas:

    RidgeModel = linear_model.Ridge(alpha=i)

    RidgeModel.fit(X_train, y_train)

    

    y_hat_val = RidgeModel.predict(X_validation)

    y_hat_tra = RidgeModel.predict(X_train)

    

    wError_val = WMAE(X_validation['IsHoliday'], y_hat_val, y_validation)

    wError_tra = WMAE(X_train['IsHoliday'], y_hat_tra, y_train)

    

    WMAE_list_val.append(wError_val)

    WMAE_list_tra.append(wError_tra)
#Ploting the Training and Validation Curves

plt.plot(alphas, WMAE_list_tra, label='Training Set')

plt.plot(alphas, WMAE_list_val, label='Validation Set')

plt.xlabel('Alpha')

plt.ylabel('Weighted Mean Absolute Error')

plt.title('Ridge Regression Error Analysis')

plt.legend()

plt.show()



#Selecting the alpha

WMAE_min = np.amin(WMAE_list_val)

alpha_min = alphas[WMAE_list_val == WMAE_min]



print("Alpha: " + str(alpha_min))

print("Minimum WMAE: " + str(WMAE_min))
#Params

n_estimators = [10, 50, 80, 100, 150, 200]

WMAE_list_val = []

WMAE_list_tra = []



for i in n_estimators:

    print('Number of Estimators: ', i)

    RF = ensemble.RandomForestRegressor(n_estimators = i, n_jobs = -1, random_state = 42)

    RF.fit(X_train, y_train)

    

    y_hat_val = RF.predict(X_validation)

    y_hat_tra = RF.predict(X_train)

    

    wError_val = WMAE(X_validation['IsHoliday'], y_hat_val, y_validation)

    wError_tra = WMAE(X_train['IsHoliday'], y_hat_tra, y_train)

    

    WMAE_list_val.append(wError_val)

    WMAE_list_tra.append(wError_tra)
#Ploting the Training and Validation Curves

plt.plot(n_estimators, WMAE_list_tra, label='Training Set')

plt.plot(n_estimators, WMAE_list_val, label='Validation Set')

plt.xlabel('Number of Estimators')

plt.ylabel('Weighted Mean Absolute Error')

plt.title('Random Forest Regression Error Analysis')

plt.legend()

plt.show()



#Selecting the alpha

WMAE_min = np.amin(WMAE_list_val)

n_estimators_min = n_estimators[np.argmax(WMAE_list_val == WMAE_min)]



print("Number of Estimators: " + str(n_estimators_min))

print("Minimum WMAE: " + str(WMAE_min))
#Params

max_depth = [10, 30, 50, 70, 90]

WMAE_list_val = []

WMAE_list_tra = []



for i in max_depth:

    print('Max Depth: ', i)

    RF = ensemble.RandomForestRegressor(n_estimators = n_estimators_min, max_depth = i, 

                                        n_jobs = -1, random_state = 42)

    RF.fit(X_train, y_train)

    

    y_hat_val = RF.predict(X_validation)

    y_hat_tra = RF.predict(X_train)

    

    wError_val = WMAE(X_validation['IsHoliday'], y_hat_val, y_validation)

    wError_tra = WMAE(X_train['IsHoliday'], y_hat_tra, y_train)

    

    WMAE_list_val.append(wError_val)

    WMAE_list_tra.append(wError_tra)
#Ploting the Training and Validation Curves

plt.plot(max_depth, WMAE_list_tra, label='Training Set')

plt.plot(max_depth, WMAE_list_val, label='Validation Set')

plt.xlabel('Max Depth')

plt.ylabel('Weighted Mean Absolute Error')

plt.title('Random Forest Regression Error Analysis')

plt.legend()

plt.show()



#Selecting the alpha

WMAE_min = np.amin(WMAE_list_val)

max_depth_min = max_depth[np.argmax(WMAE_list_val == WMAE_min)]



print("Max Depth: " + str(max_depth_min))

print("Minimum WMAE: " + str(WMAE_min))
#List of architectures to be tested

architectures = [(30, 30, 10, 10), (30, 30, 10, 10, 5, 5), (30, 30, 10, 5, 5, 5)]



WMAE_list_val = []

WMAE_list_tra = []



for i in architectures: 

    print('Architecture: ', i)

    MLP = neural_network.MLPRegressor(hidden_layer_sizes = i, max_iter=10000, random_state=42)

    MLP.fit(X_train, y_train)



    y_hat_val = MLP.predict(X_validation)

    y_hat_tra = MLP.predict(X_train)

    

    wError_val = WMAE(X_validation['IsHoliday'], y_hat_val, y_validation)

    wError_tra = WMAE(X_train['IsHoliday'], y_hat_tra, y_train)

    

    WMAE_list_val.append(wError_val)

    WMAE_list_tra.append(wError_tra)
#Ploting the Training and Validation Curves

plt.plot(np.arange(0, len(architectures)), WMAE_list_tra, label='Training Set')

plt.plot(np.arange(0, len(architectures)), WMAE_list_val, label='Validation Set')

plt.xlabel('Architectures')

plt.ylabel('Weighted Mean Absolute Error')

plt.title('MLP Regression Error Analysis')

plt.legend()

plt.show()



#Selecting the alpha

WMAE_min = np.amin(WMAE_list_val)

architecture_min = architectures[np.argmax(WMAE_list_val == WMAE_min)]



print("Architecture: " + str(architecture_min))

print("Minimum WMAE: " + str(WMAE_min))
#Params

new_features_remove = ['CPI', 'Unemployment', 'Temperature']

WMAE_list_val = []

WMAE_list_tra = []



for i in new_features_remove:

    print('Removing: ', i)

    

    X_train = X_train.drop(columns=i)

    X_validation = X_validation.drop(columns=i)

    

    RF = ensemble.RandomForestRegressor(n_estimators = n_estimators_min, max_depth = max_depth_min, 

                                        n_jobs = -1, random_state = 42)

    RF.fit(X_train, y_train)

    

    y_hat_val = RF.predict(X_validation)

    y_hat_tra = RF.predict(X_train)

    

    wError_val = WMAE(X_validation['IsHoliday'], y_hat_val, y_validation)

    wError_tra = WMAE(X_train['IsHoliday'], y_hat_tra, y_train)

    

    WMAE_list_val.append(wError_val)

    WMAE_list_tra.append(wError_tra)
#Ploting the Training and Validation Curves

plt.plot(new_features_remove, WMAE_list_tra, label='Training Set')

plt.plot(new_features_remove, WMAE_list_val, label='Validation Set')

plt.xlabel('Features Removed')

plt.ylabel('Weighted Mean Absolute Error')

plt.title('Random Forest Error Analysis')

plt.legend()

plt.show()



#Selecting the alpha

WMAE_min = np.amin(WMAE_list_val)

#new_features_remove_min = new_features_remove[np.argmax(WMAE_list_val == WMAE_min)]



#print("After " + str(architecture_min) + ' we got minimum error')

print("Minimum WMAE: " + str(WMAE_min))
RF = ensemble.RandomForestRegressor(n_estimators = n_estimators_min, max_depth=max_depth_min, n_jobs = -1)

RF.fit(X_train, y_train)



X_test = X_test.drop(columns=new_features_remove)

y_hat_test = RF.predict(X_test)
Store_list = [str(x) for x in test_data['Store']]

Dept_list = [str(x) for x in test_data['Dept']]



id_list = []

for i in range(len(Store_list)):

    id_list.append(Store_list[i] + '_' + Dept_list[i] + '_' + Date_list[i])



Output = pd.DataFrame({'id':id_list, 'Weekly_Sales':y_hat_test})

Output.to_csv('submission.csv', index=False)



Output