import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
import os

import gc

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
PATH = '../input/allstate-claims-severity/'
# Getting the file names

files = os.listdir(PATH)

files
# Reading the csv files as Pandas DataFrame and store it separate variables 

variables = [x.replace('.csv', '') for x in files]



for v, f in zip(variables, files):

    exec("%s=pd.read_csv(PATH+f)" % v)
# Dropping Id column from train and test sets

tr, ts = train.drop(['id'], 1), test.drop(['id'], 1)
# Overview of Train set

tr.head()
# Overview of Test set

ts.head()
# Checking shape

print(f'Shape of Train: {tr.shape}')

print(f'Shape of Test: {ts.shape}')



# Inference: 

# There are 131 columns in train and 130 in test
# Checking for missing values

print(f'No.of missing values in Train: {tr.isnull().sum().sum()}')

print(f'No.of missing values in Test: {ts.isnull().sum().sum()}')



# Inference: 

# There are no missing values in the dataset.
# Checking for different dtypes

pd.DataFrame({'Train': tr.dtypes.value_counts(), 'Test': ts.dtypes.value_counts()})



# Inference: 

# There are no problems with the datatypes between train and test sets.
# Understanding the categorical variables

categories = tr.describe(include='object')

categories



# Inference: 

# We understand the no.of distinct values in each categorical columns and the most frequent one
# Understanding the continuous variables

fig, axes = plt.subplots(4, 4, figsize=(24, 15))



for col, ax in zip(tr.select_dtypes(['float64']).columns, axes.flat):

    sns.kdeplot(tr[col], shade=True, ax=ax)

    ax.set_title(col)

plt.show()



# Inference: 

# We are having multi-modal distributions in some columns
# Checking for skewness in the data

plt.figure(figsize=(20,5))

sns.barplot(tr.select_dtypes(['float64']).columns, tr.skew())



# Inference:

# No major skewness identified other than loss column
# Understanding the correlation.

plt.figure(figsize=(20, 10))

sns.heatmap(tr.corr(), annot=True)



# Inference: 

# There are certain independent variables correlated to other independent variables. These only adds bias to the model so we can remove them.

# No feature is highly correlated with the target
tr['loss'] = np.log1p(tr['loss'])

sns.kdeplot(tr['loss'], shade=True)



# Note:

# We use np.log1p returns all positive log distribution rather than np.log which returns negative values
SEED = 10
# # Shuffling the train set

# tr = tr.sample(random_state=SEED, frac=1)



# # Note: 

# # This shuffles the data. 

# # Shuffling is necessary so that after splitting the data into train, validation and test sets our model gets diverse amount of data to be trained on.
# Filtering categories having low percent of data distribution

# Length of the dataframe / (No.of unique values * Frequency)



dist_percent = np.divide(len(tr), np.multiply(categories.loc['unique'].values, categories.loc['freq'].values))

percentages = pd.DataFrame({'column': categories.columns, 'percent': dist_percent})

cat_cols = list(percentages[percentages['percent'] > 0.1]['column'].values)



print(f"Filtered categories: {len(cat_cols)}")



# Note:

# I am experimenting this technique and let us see if this works
# Filtering continuous variables with correlation greater than 0.75

corr = tr.drop('loss', 1).corr().abs()

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

to_drop_num = [column for column in upper.columns if any(upper[column] > 0.6)]

num_cols = list(set(tr.drop('loss', 1).select_dtypes(['float64'])) - set(to_drop_num))
# Select only the filtered categories and continuous variables

final_tr = tr[cat_cols + num_cols + ['loss']]

final_ts = ts[cat_cols + num_cols]

final_tr.shape, final_ts.shape
# Prearing features and target before encoding

features = final_tr.drop('loss', 1)

target = final_tr['loss']
# One-hot encode train data

features = pd.get_dummies(features, columns=features.select_dtypes('object').columns)

features.shape
# One-hot encode test data

payload = pd.get_dummies(final_ts, columns=final_ts.select_dtypes('object').columns)

payload.shape
# Handling missing columns in train and test sets

features, payload = features.align(payload, join='outer', axis=1, fill_value=0)

features.shape, payload.shape
cols = features.columns
# # Scaling

# from sklearn.preprocessing import StandardScaler



# sc = StandardScaler()

# features = sc.fit_transform(features)

# payload = sc.transform(payload)
SIZE = 0.2
from sklearn.model_selection import train_test_split



x, x_test, y, y_test = train_test_split(features, target, random_state=SEED, test_size=SIZE)

x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=SEED, test_size=SIZE)
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from sklearn.model_selection import KFold



from sklearn.metrics import mean_absolute_error, accuracy_score
eval_set  = [(x_val,y_val)]

model = CatBoostRegressor(random_state=SEED, 

                          n_estimators=6000,

                          max_depth=3, 

                          learning_rate=0.1, 

                          loss_function='MAE', 

                          eval_metric='MAE', 

                          reg_lambda=3, 

                          verbose=100)

%time model.fit(x_train, y_train, eval_set=eval_set, early_stopping_rounds=30)
y_train_pred = model.predict(x_train)

y_val_pred = model.predict(x_val)

y_test_pred = model.predict(x_test)
print(f"MAE Train: {mean_absolute_error(np.expm1(y_train_pred), np.expm1(y_train))}")

print(f"MAE Val: {mean_absolute_error(np.expm1(y_val_pred), np.expm1(y_val))}")

print(f"MAE Test: {mean_absolute_error(np.expm1(y_test_pred), np.expm1(y_test))}")
# importances = pd.DataFrame({'Feature': cols, 'Importance': np.round(model.feature_importances_, 4)}).sort_values('Importance', ascending=False)

# importances[importances['Importance'] > 0]
# Scoring

predictions = np.round(np.expm1(model.predict(payload)), 2)



submission = pd.DataFrame({'id': test['id'], 'loss': pd.Series(predictions)})

submission.to_csv('submission.csv', index=False)



# Note:

# We need to take the inverse for our predictions as it is the result of np.log1p(). We can do that using np.expm1() function
submission.head(10)