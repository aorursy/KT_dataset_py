import xgboost as xgb
import os
import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

#import tensorflow as tf
#import keras.preprocessing.image
#import sklearn.preprocessing
#import sklearn.model_selection
#import sklearn.metrics
#import sklearn.linear_model
#import sklearn.naive_bayes
#import sklearn.tree
#import sklearn.ensemble
#import cv2
#import seaborn as sns
#import matplotlib.cm as cm  


#display parent directory and working directory
print(os.path.dirname(os.getcwd())+':', os.listdir(os.path.dirname(os.getcwd())+'/input'));
print(os.getcwd()+':', os.listdir(os.getcwd()));
#load_data

load_success = True
if os.path.isfile('../input/sales_train.csv'):
    train_df = pd.read_csv('../input/sales_train.csv') # on kaggle
    #dtrain = xgb.DMatrix('../input/sales_train.csv?format=csv&label_column=0')
    print('sales_train.csv loaded: data_df({0[0]},{0[1]})'.format(train_df.shape))
else:
    load_success = False

if os.path.isfile('../input/test.csv'):
    test_df = pd.read_csv('../input/test.csv') # on kaggle
    #dtest = xgb.DMatrix('../input/test.csv?format=csv')
    print('test.csv loaded: data_df({0[0]},{0[1]})'.format(test_df.shape))
else:
    load_success = False    

if os.path.isfile('../input/items.csv'):
    item_df = pd.read_csv('../input/items.csv') # on kaggle
    #ditem = xgb.DMatrix('../input/items.csv?format=csv')
    print('items.csv: data_df({0[0]},{0[1]})'.format(item_df.shape))
else:
    load_success = False

if os.path.isfile('../input/shops.csv'):
    shop_df = pd.read_csv('../input/shops.csv') # on kaggle
    #dshops = xgb.DMatrix('../input/shops.csv?format=csv')
    print('shops.csv: data_df({0[0]},{0[1]})'.format(shop_df.shape))
else:
    load_success = False
    
if os.path.isfile('../input/item_categories.csv'):
    category_df = pd.read_csv('../input/item_categories.csv') # on kaggle
    #dcategory = xgb.DMatrix('../input/item_categories.csv?format=csv')
    print('item_categories.csv loaded: data_df({0[0]},{0[1]})'.format(category_df.shape))
else:
    load_success = False
    
    
if not load_success:
    print('Error: train.csv not found')
#train_df['shop_id']= train_df.shop_id.astype('str')
#train_df['item_id']= train_df.item_id.astype('str')
del train_df['date']
train_df.head()
dtrain = xgb.DMatrix(train_df)
dtest = xgb.DMatrix(test_df)
#ditem = xgb.Dmatrix(item_df)
#dshops = xgb.Dmatrix(shop_df)
#dcategory = xgb.Dmatrix(category_df)
progress=dict()
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 1000
bst = xgb.train(param, dtrain, num_round, evals=[(dtrain,'train'),(dtest,'test')], evals_result=progress)
#param = {'max_depth':10, 'subsample':1, 'min_child_weight':0.5, 'eta':0.3, 'num_round':1000,'seed':1, 'silent':0, 'eval_metric':'rmse'}
