#Pretty Display of Variables

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# import required packages



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline





import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,confusion_matrix,roc_curve,auc

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures,LabelEncoder,OneHotEncoder

from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier





import lightgbm as lgb

import xgboost as xgb


# load data

#data = pd.read_csv("dell_data.csv")

data = pd.read_csv("../input/inventory/inventory.csv")



# View first 5 rows

data.head()



# View last 5 rows

data.tail()



# Shape of the data

data.shape
# dtypes of the data

data.dtypes
# Finding null values

data.isnull().sum()
# columns of the data

data.columns
# Converting date columns to pandas datetime

data['DMND_WEEK_STRT_DATE'] = pd.to_datetime(data['DMND_WEEK_STRT_DATE'])

data['VRSN_WEEK_STRT_DATE'] = pd.to_datetime(data['VRSN_WEEK_STRT_DATE'])
# Assaign difference of the DMND_WEEK_STRT_DATE & VRSN_WEEK_STRT_DATE to DMND_VRSN_Diff

data['DMND_VRSN_Diff'] = (data['DMND_WEEK_STRT_DATE'] - data['VRSN_WEEK_STRT_DATE']).dt.days.astype(int)
# Converting DW_PKG_UPD_DTS to min(int)

data["UPD_TIME"] = data.DW_PKG_UPD_DTS.str.slice(0, 2).astype(int)



# view first rows

data["UPD_TIME"].head()
# Dropping processed columns

data = data.drop(columns=["DMND_WEEK_STRT_DATE","VRSN_WEEK_STRT_DATE", "DW_PKG_UPD_DTS"], axis=1)



# Shape of the data

data.shape
#The number of unique values??

for col in data.columns[0:]:

    print(col, "------", data[col].nunique())
# Descriptive Summary of the data



def description(data):

    summary = pd.DataFrame(data.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = data.isnull().sum().values    

    summary['Uniques'] = data.nunique().values

    summary['First Value'] = data.iloc[0].values

    summary['Second Value'] = data.iloc[1].values

    summary['Third Value'] = data.iloc[2].values

    return summary





description(data)
# object columns

str_cols= data.loc[:, data.dtypes=='object'].columns.tolist()



str_cols
print(f'Shape before dummy transformation: {data.shape}')



dummies = pd.get_dummies(data, columns=str_cols,\

                          prefix=str_cols, drop_first=True)

print(f'Shape after dummy transformation: {dummies.shape}')


from sklearn.model_selection import train_test_split



# feature columns and target

X = data.drop("MRP_FCST_QTY", axis=1)

y = data.MRP_FCST_QTY



# Break off test and train set from data

X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                      test_size=0.3,

                                                      random_state=42)
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# RMSLE

from sklearn.metrics import mean_squared_log_error





# RandomForestRegressor

def score_dataset(X_train, X_test, y_train, y_test):

    model = RandomForestRegressor(n_estimators=1000, random_state=42)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return np.sqrt(mean_squared_log_error(y_test, preds))





#RMSLE = np.sqrt(mean_squared_log_error( y_test, predictions ))
# Drop categorical columns in training and testing data

drop_X_train = X_train.select_dtypes(exclude=['object'])

drop_X_test = X_test.select_dtypes(exclude=['object'])
# print("LR RMSLE from Approach 1 (Dropping categorical variables):")

# print(score_dataset_lr(drop_X_train, drop_X_test, y_train, y_test))



print("RMSLE from Approach 1 (Dropping categorical variables):")

print(score_dataset(drop_X_train, drop_X_test, y_train, y_test))
# All categorical columns

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(X_train[col]) == set(X_test[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))

        

print('Categorical columns that will be label encoded:', good_label_cols)

print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
from sklearn.preprocessing import LabelEncoder



# Drop categorical columns that will not be encoded

label_X_train = X_train.drop(bad_label_cols, axis=1)

label_X_test = X_test.drop(bad_label_cols, axis=1)



# Apply label encoder

label_encoder = LabelEncoder()

for col in set(good_label_cols):

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_test[col] = label_encoder.transform(X_test[col])
# Train & Score Model for Label Encoding

print("RMSLE from Approach 2 (Label Encoding):") 

print(score_dataset(label_X_train, label_X_test, y_train, y_test))
# Investigating cardinality



# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 30]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
from sklearn.preprocessing import OneHotEncoder





# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_test.index = X_test.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(object_cols, axis=1)

num_X_test = X_test.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
print("RMSLE from Approach 3 (One-Hot Encoding):") 

print(score_dataset(OH_X_train, OH_X_test, y_train, y_test))
y_test.head()

y_test.shape


# create dataset for lightgbm

lgb_train = lgb.Dataset(OH_X_train, y_train)

lgb_eval = lgb.Dataset(OH_X_test, y_test, reference=lgb_train)





# specify your configurations as a dict



params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'l2', 'l1'},

    'num_leaves': 31,

    'learning_rate': 0.005,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0

}





print('Starting training...')

# train

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=20,

                valid_sets=lgb_eval,

                early_stopping_rounds=5)



print('Saving model...')

# save model to file

gbm.save_model('model.txt')



print('Starting predicting...')

# predict

y_pred = gbm.predict(OH_X_test, num_iteration=gbm.best_iteration)

# eval

print('The rmlse of prediction is:', np.sqrt(mean_squared_log_error(y_test, y_pred)))
cat_dat = data.select_dtypes(include=['object']).copy()

cat_dat.head()
#The number of unique values??

for col in cat_dat.columns[0:]:

    print(col, " ------ ", data[col].nunique())
str_cols= cat_dat.loc[:, cat_dat.dtypes=='object'].columns.tolist()



str_cols
import category_encoders as ce



encoder = ce.BinaryEncoder(cols=str_cols)

df_binary = encoder.fit_transform(cat_dat)



df_binary.head()

df_binary.shape
data_c = data.drop(str_cols, axis=1)



data_c.shape
data_f = pd.concat([data_c, df_binary], axis=1)



data_f.shape
data_f.dtypes


from sklearn.model_selection import train_test_split



# feature columns and target

X = data_f.drop("MRP_FCST_QTY", axis=1)

y = data_f.MRP_FCST_QTY



# Break off test and train set from data

X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                      test_size=0.3,

                                                      random_state=42)


# RandomForestRegressor

def score_dataset_be(X_train, X_test, y_train, y_test):

    model = RandomForestRegressor(n_estimators=1000, random_state=42)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return np.sqrt(mean_squared_log_error(y_test, preds))





#RMSLE = np.sqrt(mean_squared_log_error( y_test, predictions ))




print("RMSLE from Binary Encoding:")

print(score_dataset_be(X_train, X_test, y_train, y_test))


# create dataset for lightgbm

lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)





# specify your configurations as a dict



params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'l2', 'l1'},

    'num_leaves': 31,

    'learning_rate': 0.005,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0

}





print('Starting training...')

# train

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=200,

                valid_sets=lgb_eval,

                early_stopping_rounds=50)



print('Saving model...')

# save model to file

gbm.save_model('model_be.txt')



print('Starting predicting...')

# predict

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# eval

print('The rmlse of prediction is:', np.sqrt(mean_squared_log_error(y_test, y_pred)))
#BackwardDifferenceEncoder



#encoder = ce.BackwardDifferenceEncoder(cols=str_cols)

#df_bd = encoder.fit_transform(cat_dat)



#df_bd.head()
#Polynomial Coding

#Helmert Coding

'''

import category_encoders as ce



encoder = ce.BackwardDifferenceEncoder(cols=[...])

encoder = ce.BaseNEncoder(cols=[...])

encoder = ce.BinaryEncoder(cols=[...])

encoder = ce.CatBoostEncoder(cols=[...])

encoder = ce.HashingEncoder(cols=[...])

encoder = ce.HelmertEncoder(cols=[...])

encoder = ce.JamesSteinEncoder(cols=[...])

encoder = ce.LeaveOneOutEncoder(cols=[...])

encoder = ce.MEstimateEncoder(cols=[...])

encoder = ce.OneHotEncoder(cols=[...])

encoder = ce.OrdinalEncoder(cols=[...])

encoder = ce.SumEncoder(cols=[...])

encoder = ce.PolynomialEncoder(cols=[...])

encoder = ce.TargetEncoder(cols=[...])

encoder = ce.WOEEncoder(cols=[...])



encoder.fit(X, y)

X_cleaned = encoder.transform(X_dirty)

'''
#http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/