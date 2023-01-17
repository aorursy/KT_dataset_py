import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import catboost as cboost

import csv



%matplotlib inline
df_train = pd.read_csv('../input/train.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)

df_test = pd.read_csv('../input/test.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
# We only use categorical features in this naive approach

categorical_features = ['item_condition_id', 'category_name', 'brand_name', 'shipping']



df_x_train = df_train[categorical_features].copy()

df_x_test = df_test[categorical_features].copy()

df_y = df_train['price']
# Factorize both train and test (avoid unseen categories in train)

def factorize(train, test, col):

    cat_ids = sorted(set(train[col].dropna().unique()) | set(test[col].dropna().unique()))



    cat_ids = {k:i for i, k in enumerate(cat_ids)}

    cat_ids[np.nan] = -1



    train[col] = train[col].map(cat_ids)

    test[col]  = test[col].map(cat_ids)



# Factorize string columns

factorize(df_x_train, df_x_test, 'category_name')

factorize(df_x_train, df_x_test, 'brand_name')
# Create train and test Pool of train

ptrain = cboost.Pool(df_x_train, df_y, cat_features=np.arange(len(categorical_features)),

                     column_description=categorical_features)



ptest = cboost.Pool(df_x_test, cat_features=np.arange(len(categorical_features)),

                     column_description=categorical_features)
# Tune your parameters here!

cboost_params = {

    'nan_mode': 'Min',

    'loss_function': 'RMSE',  # Try 'LogLinQuantile' as well

    'iterations': 150,

    'learning_rate': 0.75,

    'depth': 5,

    'verbose': True

}



best_iter = cboost_params['iterations']  # Initial 'guess' it not using CV
# Train model on full data

model = cboost.CatBoostRegressor(**dict(cboost_params, verbose=False, iterations=best_iter))



fit_model = model.fit(ptrain)
pred_1 = fit_model.predict(ptest)
import csv

df_train = pd.read_csv('../input/train.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)

df_test = pd.read_csv('../input/test.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)



median = df_train['price'].median()

train = df_train.groupby('category_name')['price'].median()

price_dict = train.to_dict()



pred_2 = []

for i, row in df_test.iterrows():

    category_name = row['category_name']

    if(category_name not in price_dict):

        pred_2.append(median)

    else:

        pred_2.append(price_dict[category_name])
preds = np.clip(0.67*pred_1 + 0.33*np.array(pred_2), 0, 10000000000)

preds
sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['price'] = preds

sub.to_csv('blend_sub.csv', index=False)