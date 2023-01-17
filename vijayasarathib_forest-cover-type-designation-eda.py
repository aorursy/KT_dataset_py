# Import libraries

import pandas as pd # linear algebra

import pandas_profiling as pp

import numpy as np # Data processsing

import os # os functions

import matplotlib.pyplot as plt # Plotting library

import seaborn as sns # Data visualization 

from sklearn.model_selection import train_test_split,KFold,GridSearchCV,RandomizedSearchCV



from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier, plot_importance

import xgboost as xgb

import re

import sklearn

from sklearn.preprocessing import MinMaxScaler



import warnings

warnings.filterwarnings('ignore')

from sklearn.experimental import enable_hist_gradient_boosting

# Going to use these  base models for the stacking

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, 

                              ExtraTreesClassifier,  HistGradientBoostingClassifier)

from sklearn.svm import SVC

from sklearn.metrics import make_scorer, accuracy_score,classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier







sns.set() #Set aesthetic parameters in one step.

pd.options.display.max_columns=100 # display setting
# Load training and test data 

forest_classification_train =pd.read_csv('/kaggle/input/learn-together/train.csv',index_col = 'Id')

forest_classification_test = pd.read_csv('/kaggle/input/learn-together/test.csv',index_col = 'Id')
forest_classification_train.shape
forest_classification_test.shape
forest_classification_train.head()
forest_classification_test.head()
forest_classification_train.describe().T
# Missing values in the training data

print(f"Missing Values in train: {forest_classification_train.isna().any().any()}")



# Missing values in the test data

print(f"Missing Values in test: {forest_classification_test.isna().any().any()}")
# Lets check the column Types

print(f"Train Column Types: {set(forest_classification_train.dtypes)}")

print(f"Test Column Types: {set(forest_classification_test.dtypes)}")
# Next lets check the number of unique values by column.

for column in forest_classification_train.columns:

    print(column, forest_classification_train[column].nunique())
print("Soil_Type7: ", forest_classification_test["Soil_Type7"].nunique())

print("Soil_Type15: ", forest_classification_test["Soil_Type15"].nunique())
print("- - - Test - - -")

print(forest_classification_test["Soil_Type7"].value_counts())

print(forest_classification_test["Soil_Type15"].value_counts())
# drop soil type 7 and 15.



forest_classification_train = forest_classification_train.drop(["Soil_Type7", "Soil_Type15"], axis = 1)

forest_classification_test = forest_classification_test.drop(["Soil_Type7", "Soil_Type15"], axis = 1)
forest_classification_train.columns
# Histogram for the variable elevation

forest_classification_train["Elevation"].plot(kind="hist",bins=30)
# Cover type vs Elevation

forest_classification_train.plot(kind="scatter", x="Cover_Type", y="Elevation")
forest_classification_train["Aspect"].plot(kind="hist", bins = 30)
forest_classification_train.plot(kind="scatter", x="Cover_Type", y="Aspect")

forest_classification_train["Slope"].plot(kind="hist", bins = 30)
sns.scatterplot(x=forest_classification_train["Slope"], y=forest_classification_train["Cover_Type"])
forest_classification_train["Horizontal_Distance_To_Hydrology"].plot(kind="hist", bins = 30)
forest_classification_train.plot(kind="scatter", x="Cover_Type", y="Horizontal_Distance_To_Hydrology")
forest_classification_train["Vertical_Distance_To_Hydrology"].plot(kind='hist', bins = 30)
forest_classification_train.plot(kind="scatter", x="Cover_Type", y="Vertical_Distance_To_Hydrology")
forest_classification_train["Horizontal_Distance_To_Roadways"].plot(kind='hist', bins = 30)

forest_classification_train.plot(kind="scatter", x="Cover_Type", y="Horizontal_Distance_To_Roadways")
forest_classification_train["Hillshade_9am"].plot(kind="hist", bins = 30)

forest_classification_train.plot(kind="scatter", x="Cover_Type", y="Hillshade_9am")
forest_classification_train["Hillshade_Noon"].plot(kind="hist", bins = 30)

forest_classification_train.plot(kind="scatter", x="Cover_Type", y="Hillshade_Noon")
forest_classification_train["Hillshade_3pm"].plot(kind="hist", bins = 30)
forest_classification_train.plot(kind="scatter", x="Cover_Type", y="Hillshade_3pm")
forest_classification_train["Horizontal_Distance_To_Fire_Points"].plot(kind="hist", bins = 30)

forest_classification_train.plot(kind="scatter", x="Cover_Type", y="Horizontal_Distance_To_Fire_Points")

sns.countplot(x="Wilderness_Area1", data=forest_classification_train)

sns.countplot(x="Wilderness_Area2", data=forest_classification_train)
sns.countplot(x="Wilderness_Area3", data=forest_classification_train)
sns.countplot(x="Wilderness_Area4", data=forest_classification_train)
res_soil_dict = {}

for col in forest_classification_train.columns[14:-1]:

    res_soil_dict[col] = forest_classification_train[col].value_counts().loc[1] 

sorted_d = sorted(res_soil_dict.items(), key=lambda kv: kv[1])
sorted_d
forest_classification_train["Cover_Type"].plot(kind="hist", bins = 30)
forest_classification_train["Cover_Type"].shape
#Change the variable with just two values into categorical variables.

forest_classification_train.iloc[:,10:-1] = forest_classification_train.iloc[:,10:-1].astype("category")

forest_classification_test.iloc[:,10:] = forest_classification_test.iloc[:,10:].astype("category")



# Create the correlation plot

f,ax = plt.subplots(figsize=(8,6))

sns.heatmap(forest_classification_train.corr(),annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()



train = forest_classification_train

test = forest_classification_test

train.iloc[:,10:-1] = train.iloc[:,10:-1].astype("category")

test.iloc[:,10:] = test.iloc[:,10:].astype("category")
# some of the feature engineering steps based on "Pruthvi H.R, Nisha K.K, Chandana T.L, Navami K and Biju R M (2015) ‘Feature engineering on forest cover type data with ensemble of decision trees’, 2015 IEEE International Advance Computing Conference (IACC). Banglore, India: IEEE, pp. 1093–1098. Available at: 10.1109/IADCC.2015.7154873 (Accessed: 2 September 2019)."



train['EV_DTH'] = (train.Elevation - train.Vertical_Distance_To_Hydrology)

test['EV_DTH'] = (test.Elevation - test.Vertical_Distance_To_Hydrology)



train['EH_DTH'] = (train.Elevation -  (train.Horizontal_Distance_To_Hydrology *0.2))

test['EH_DTH'] = (test.Elevation -  (test.Horizontal_Distance_To_Hydrology *0.2))



train['Dis_To_Hy'] = (((train.Horizontal_Distance_To_Hydrology **2) + (train.Vertical_Distance_To_Hydrology **2))**0.5)

test['Dis_To_Hy'] = (((test.Horizontal_Distance_To_Hydrology **2) + (test.Vertical_Distance_To_Hydrology **2))**0.5)



train['HyF_1'] = (train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Fire_Points)

test['HyF_1'] = (test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Fire_Points)



train['HyF_2'] = (train.Horizontal_Distance_To_Hydrology - train.Horizontal_Distance_To_Fire_Points)

test['HyF_2'] = (test.Horizontal_Distance_To_Hydrology - test.Horizontal_Distance_To_Fire_Points)



train['HyR_1'] = (train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways)

test['HyR_1'] = (test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways)



train['HyR_2'] = (train.Horizontal_Distance_To_Hydrology - train.Horizontal_Distance_To_Roadways)

test['HyR_2'] = (test.Horizontal_Distance_To_Hydrology - test.Horizontal_Distance_To_Roadways)





train['FiR_1'] = (train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Roadways)

test['FiR_1'] = (test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Roadways)



train['FiR_1'] = (train.Horizontal_Distance_To_Fire_Points - train.Horizontal_Distance_To_Roadways)

test['FiR_1'] = (test.Horizontal_Distance_To_Fire_Points - test.Horizontal_Distance_To_Roadways)



train['Avg_shade'] = ((train.Hillshade_9am + train.Hillshade_Noon + train.Hillshade_3pm) /3)

test['Avg_shade'] = ((test.Hillshade_9am + test.Hillshade_Noon + test.Hillshade_3pm) /3)



train['Morn_noon_int'] = ((train.Hillshade_9am + train.Hillshade_Noon) / 2)

test['Morn_noon_int'] = ((test.Hillshade_9am + test.Hillshade_Noon) / 2)



train['noon_eve_int'] = ((train.Hillshade_3pm + train.Hillshade_Noon) / 2)

test['noon_eve_int'] = ((test.Hillshade_3pm + test.Hillshade_Noon) / 2)



train['Slope2'] = np.sqrt(train.Horizontal_Distance_To_Hydrology**2 + train.Vertical_Distance_To_Hydrology**2)

test['Slope2'] = np.sqrt(test.Horizontal_Distance_To_Hydrology**2 + test.Vertical_Distance_To_Hydrology**2)
train.isna().any().any()
train.columns
cols=list(train.columns)

numeric_columns=[]

for i in range(10):

    numeric_columns.append(cols[i])

    

for i in range(12):

    numeric_columns.append(cols[-(i+1)])

print(numeric_columns)
train_minmax=train[numeric_columns]

test_minmax=test[numeric_columns]

mm_scaler = MinMaxScaler()

# my_train_minmax = mm_scaler.fit_transform(train_df[other_columns])

mm_scaler.fit(train_minmax)

train_trans=mm_scaler.transform(train_minmax)

test_trans=mm_scaler.transform(test_minmax)



temp_train=pd.DataFrame(train_trans)

temp_test=pd.DataFrame(test_trans)

train.isna().any().any()
for i in range(22):

    temp_train.rename(columns={i:numeric_columns[i]},inplace=True)

    temp_test.rename(columns={i:numeric_columns[i]},inplace=True)

temp_train.head()
temp_train.index +=1

temp_test.index +=1
train[numeric_columns]=temp_train[numeric_columns]

test[numeric_columns]=temp_test[numeric_columns]

train.head()
temp_train
train
# train and validation data split



X_train, X_val, y_train, y_val = train_test_split(forest_classification_train.drop(['Cover_Type'], axis=1), forest_classification_train['Cover_Type'], test_size=0.2,shuffle = True, random_state = 2)
X_train.shape
X_val.shape
y_train.shape
y_val.shape
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 8)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

# rf_model = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

# rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

# rf_random.fit(X_train, y_train)
# rf_random.best_params_

train.isna()

#test.isna().any().any()
# model = RandomForestClassifier(n_estimators=100,random_state = 42)

RF_random=RandomForestClassifier(n_estimators=885,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=110,bootstrap=False)

# Train the model on training data

RF_random.fit(train.drop(['Cover_Type'], axis=1), train['Cover_Type'])

rf = RandomForestClassifier()

model = Pipeline(steps=[('model',rf),])
n_features=test.shape[1]

int(np.sqrt(n_features))
#param_grid = {'model__n_estimators': np.logspace(2,3.5,8).astype(int),

#              'model__max_features': [0.1,0.3,0.5,0.7,0.9],

#              'model__max_depth': np.logspace(0,3,10).astype(int),

#              'model__min_samples_split': [2, 5, 10],

#              'model__min_samples_leaf': [1, 2, 4],

#              'model__bootstrap':[True, False]}

              



param_grid = {'model__n_estimators': [100,  1000],

              'model__max_features': [int(np.sqrt(n_features))]}



grid = RandomizedSearchCV(estimator=model, 

                          param_distributions=param_grid, 

                          n_iter=1, # This was set to 100 in my offline version

                          cv=3, 

                          verbose=3, 

                          n_jobs=1,

                          scoring = {'NLL':'neg_log_loss', 'Accuracy':'accuracy'}, 

                          refit='NLL')



#grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=0)

grid.fit(train.drop(['Cover_Type'], axis=1), train['Cover_Type'])
grid.best_params_
grid_pred = grid.predict(X_val)
accuracy_score(y_val, grid_pred)
xgb= XGBClassifier( n_estimators=1000,  #todo : search for good parameters

                    learning_rate= 0.5,  #todo : search for good parameters

                    objective= 'binary:logistic', #this outputs probability,not one/zero. should we use binary:hinge? is it better for the learning phase?

                    random_state= 1,

                    n_jobs=-1)
X_train.iloc[:,10:] = X_train.iloc[:,10:].astype("int64")

X_val.iloc[:,10:] = X_val.iloc[:,10:].astype("int64")



param = {'n_estimators': [100, 500, 1000, 2000],

         'learning_rate': [0.001, 0.01, 0.1, 0.5, 1, 2, 5]}

grider = GridSearchCV(xgb, param, n_jobs=-1, cv=3, scoring='accuracy', verbose=True)

#grider.fit(X_train, y_train)


xgb.fit(X_train, y_train)

xgb_val = xgb.predict(X_val)

accuracy_score(y_val, xgb_val)
gbc= GradientBoostingClassifier( n_estimators = 2000, max_depth = 35, max_features = 20, random_state = 1,

                    learning_rate= 0.1)
gbc.fit(X_train, y_train)

gbc_val = gbc.predict(X_val)

accuracy_score(y_val, gbc_val)
forest_classification_train.iloc[:,10:] = forest_classification_train.iloc[:,10:].astype("int64")

forest_classification_test.iloc[:,10:] = forest_classification_test.iloc[:,10:].astype("int64")

# Final model.

clf = XGBClassifier(n_estimators=500,colsample_bytree=0.9,max_depth=9,random_state=1,eta=0.2)

clf.fit(forest_classification_train.drop(['Cover_Type'], axis=1),forest_classification_train['Cover_Type'])
test_pred = clf.predict(forest_classification_train.drop(['Cover_Type'], axis=1),forest_classification_train.drop['Cover_Type'])
# temp

RF_random_pred = RF_random.predict(test)
# Save test predictions to file

output = pd.DataFrame({'ID': forest_classification_test.index.values,

                       'Cover_Type': RF_random_pred})

output.to_csv('submission.csv', index=False)