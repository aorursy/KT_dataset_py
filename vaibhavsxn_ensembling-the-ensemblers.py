import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor



import sklearn.linear_model

from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import MinMaxScaler



import gc

import os

import time



import xgboost as xgb

import lightgbm as lgb

from catboost import CatBoostClassifier



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
def oof_regression_stacker(train_x, train_y, test_x,

                           estimators, 

                           pred_cols, 

                           train_eval_metric, 

                           compare_eval_metric,

                           n_folds = 3,

                           holdout_x=False,

                           debug = False):

    

    """

    Original script:

        Jovan Sardinha

        https://medium.com/weightsandbiases/an-introduction-to-model-ensembling-63effc2ca4b3

        

    Args:

        train_x, train_y, test_x (DataFrame).

        n_folds (int): The number of folds for crossvalidation.

        esdtimators (list): The list of estimator functions.

        pred_cols (list): The estimator related names of prediction columns.

        train_eval_metric (class): Fucntion for the train eval metric.

        compare_eval_metric (class): Fucntion for the crossvalidation eval metric.

        holdout_x (DataFrame): Holdout dataframe if you intend to stack/blend using holdout.

        

    Returns:

        train_blend, test_blend, model

    """

    

    if debug == True:

        train_x = train_x.sample(n=1000, random_state=seed_val)

        train_y = train_y.sample(n=1000, random_state=seed_val)

        

    # Start timer:

    start_time = time.time()

    

    # List to save models:

    model_list = []

    

    # Initializing blending data frames:

    with_holdout = isinstance(holdout_x, pd.DataFrame)

    if with_holdout: holdout_blend = pd.DataFrame(holdout_x.index)

    

    train_blend = pd.DataFrame(train_x.index)

    val_blend = pd.DataFrame(train_x.index)

    test_blend = pd.DataFrame(test_x.index)



    # Arrays to hold estimators' predictions:

    test_len = test_x.shape[0]

    train_len = train_x.shape[0]



    dataset_blend_train = np.zeros((train_len, len(estimators))) # Mean train prediction holder

    dataset_blend_val = np.zeros((train_len, len(estimators))) # Validfation prediction holder                   

    dataset_blend_test = np.zeros((test_len, len(estimators))) # Mean test prediction holder

    if with_holdout: dataset_blend_holdout = np.zeros((holdout_x.shape[0], len(estimators))) # Same for holdout

        

    # Note: StratifiedKFold splits into roughly 66% train 33% test  

    folds = StratifiedShuffleSplit(n_splits= n_folds, random_state=seed_val,

                                  test_size = 1/n_folds, train_size = 1-(1/n_folds))

        

    # For every estimator:

    for j, estimator in enumerate(estimators):

        

        # Array to hold folds number of predictions on test:

        dataset_blend_train_j = np.zeros((train_len, n_folds))

        dataset_blend_test_j = np.zeros((test_len, n_folds))

        if with_holdout: dataset_blend_holdout_j = np.zeros((holdout_x.shape[0], n_folds))

        

        # For every fold:

        for i, (train, test) in enumerate(folds.split(train_x, train_y)):

            trn_x = train_x.iloc[train, :] 

            trn_y = train_y.iloc[train].values.ravel()

            val_x = train_x.iloc[test, :] 

            val_y = train_y.iloc[test].values.ravel()

            

            # Estimators conditional training:

            if estimator == 'lgb':

                model = kfold_lightgbm(trn_x, trn_y)

                pred_val = model.predict(val_x)

                pred_test = model.predict(test_x)

                pred_train = model.predict(train_x)

                if with_holdout:

                    pred_holdout = model.predict(holdout_x)                

            elif estimator == 'xgb':

                model = kfold_xgb(trn_x, trn_y)

                pred_val = xgb_predict(val_x, model)

                pred_test = xgb_predict(test_x, model)

                pred_train = xgb_predict(train_x, model)

                if with_holdout:

                    pred_holdout = xgb_predict(holdout_x, model)

            elif estimator == 'f10_dnn':

                model = f10_dnn(trn_x, trn_y)

                pred_val = model.predict(val_x).ravel()

                pred_test = model.predict(test_x).ravel()

                pred_train = model.predict(train_x).ravel()

                if with_holdout:

                    pred_holdout = model.predict(holdout_x).ravel()

                #print(pred_val.shape, pred_test.shape, pred_train.shape)             

            elif estimator == 'ridge':

                model = ridge(trn_x, trn_y)

                pred_val = model.predict(val_x)

                pred_test = model.predict(test_x)

                pred_train = model.predict(train_x)

                if with_holdout:

                    pred_holdout = model.predict(holdout_x)                         

            else:

                model = kfold_cat(trn_x, trn_y)

                pred_val = model.predict_proba(val_x)[:,1]

                pred_test = model.predict_proba(test_x)[:,1]

                pred_train = model.predict_proba(train_x)[:,1]

                if with_holdout:

                    pred_holdout = model.predict_proba(holdout_x)[:,1]         

            

            dataset_blend_val[test, j] = pred_val

            dataset_blend_test_j[:, i] = pred_test

            dataset_blend_train_j[:, i] = pred_train

            if with_holdout: 

                dataset_blend_holdout_j[:, i] = pred_holdout

            

            print('fold:', i+1, '/', n_folds,

                  '; estimator:',  j+1, '/', len(estimators),

                  ' -> oof cv score:', compare_eval_metric(val_y, pred_val))



            del trn_x, trn_y, val_x, val_y

            gc.collect()

    

        # Save curent estimator's mean prediction for test, train and holdout:

        dataset_blend_test[:, j] = np.mean(dataset_blend_test_j, axis=1)

        dataset_blend_train[:, j] = np.mean(dataset_blend_train_j, axis=1)

        if with_holdout: dataset_blend_holdout[:, j] = np.mean(dataset_blend_holdout_j, axis=1)

        

        model_list += [model]

        

    #print('--- comparing models ---')

    for i in range(dataset_blend_val.shape[1]):

        print('model', i+1, ':', compare_eval_metric(train_y, dataset_blend_val[:,i]))

        

    for i, j in enumerate(estimators):

        val_blend[pred_cols[i]] = dataset_blend_val[:,i]

        test_blend[pred_cols[i]] = dataset_blend_test[:,i]

        train_blend[pred_cols[i]] = dataset_blend_train[:,i]

        if with_holdout: 

            holdout_blend[pred_cols[i]] = dataset_blend_holdout[:,i]

        else:

            holdout_blend = False

    

    end_time = time.time()

    print("Total Time usage: " + str(int(round(end_time - start_time))))

    return train_blend, val_blend, test_blend, holdout_blend, model_list
from sklearn.linear_model import Ridge

import sklearn.linear_model



def ridge(trn_x, trn_y):

    clf = Ridge(alpha=20, 

                copy_X=True, 

                fit_intercept=True, 

                solver='auto',max_iter=10000,

                normalize=False, 

                random_state=0,  

                tol=0.0025)

    clf.fit(trn_x, trn_y)

    return clf
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout, BatchNormalization

from keras.layers.advanced_activations import PReLU

from keras.optimizers import Adam

from sklearn.model_selection import KFold



import gc

import os



from sklearn.metrics import roc_auc_score

from keras.callbacks import Callback



class roc_callback(Callback):

    def __init__(self,training_data,validation_data):

        self.x = training_data[0]

        self.y = training_data[1]

        self.x_val = validation_data[0]

        self.y_val = validation_data[1]





    def on_train_begin(self, logs={}):

        return



    def on_train_end(self, logs={}):

        return



    def on_epoch_begin(self, epoch, logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict(self.x)

        roc = roc_auc_score(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)

        roc_val = roc_auc_score(self.y_val, y_pred_val)

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')

        return



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return



def f10_dnn(X_train, Y_train, nn_num_folds=10):

    

    folds = KFold(n_splits=nn_num_folds, shuffle=True, random_state=seed_val)



    for n_fold, (nn_trn_idx, nn_val_idx) in enumerate(folds.split(X_train)):

        nn_trn_x, nn_trn_y = X_train.iloc[nn_trn_idx,:], Y_train[nn_trn_idx]

        nn_val_x, nn_val_y = X_train.iloc[nn_val_idx,:], Y_train[nn_val_idx]



        print( 'Setting up neural network...' )

        nn = Sequential()

        nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = 718))

        nn.add(PReLU())

        nn.add(Dropout(.3))

        nn.add(Dense(units = 160 , kernel_initializer = 'normal'))

        nn.add(PReLU())

        nn.add(BatchNormalization())

        nn.add(Dropout(.3))

        nn.add(Dense(units = 64 , kernel_initializer = 'normal'))

        nn.add(PReLU())

        nn.add(BatchNormalization())

        nn.add(Dropout(.3))

        nn.add(Dense(units = 26, kernel_initializer = 'normal'))

        nn.add(PReLU())

        nn.add(BatchNormalization())

        nn.add(Dropout(.3))

        nn.add(Dense(units = 12, kernel_initializer = 'normal'))

        nn.add(PReLU())

        nn.add(BatchNormalization())

        nn.add(Dropout(.3))

        nn.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

        nn.compile(loss='binary_crossentropy', optimizer='adam')



        print( 'Fitting neural network...' )

        nn.fit(nn_trn_x, nn_trn_y, validation_data = (nn_val_x, nn_val_y), epochs=10, verbose=2,

              callbacks=[roc_callback(training_data=(nn_trn_x, nn_trn_y),validation_data=(nn_val_x, nn_val_y))])

        

        #print( 'Predicting...' )

        #sub_preds += nn.predict(X_test).flatten().clip(0,1) / folds.n_splits

    

        gc.collect()

        

        return nn
def kfold_lightgbm(trn_x, trn_y, num_folds=3):

       

    # Cross validation model

    in_folds = StratifiedShuffleSplit(n_splits= num_folds, random_state=seed_val)

        

    # Create arrays and dataframes to store results

    for train_idx, valid_idx in in_folds.split(trn_x, trn_y):

        dtrain = lgb.Dataset(data=trn_x.values[train_idx], 

                             label=trn_y[train_idx], 

                             free_raw_data=False, silent=True)

        dvalid = lgb.Dataset(data=trn_x.values[valid_idx], 

                             label=trn_y[valid_idx], 

                             free_raw_data=False, silent=True)



        # LightGBM parameters found by Bayesian optimization

        params = {

            'objective': 'binary',

            'boosting_type': 'gbdt',

            'nthread': 4,

            'learning_rate': 0.02,  # 02,

            'num_leaves': 20,

            'colsample_bytree': 0.9497036,

            'subsample': 0.8715623,

            'subsample_freq': 1,

            'max_depth': 8,

            'reg_alpha': 0.041545473,

            'reg_lambda': 0.0735294,

            'min_split_gain': 0.0222415,

            'min_child_weight': 60, # 39.3259775,

            'seed': seed_val,

            'verbose': -1,

            'metric': 'auc',

        }

        

        clf = lgb.train(

            params=params,

            train_set=dtrain,

            num_boost_round=10000,

            valid_sets=[dtrain, dvalid],

            early_stopping_rounds=200,

            verbose_eval=False

        )



        del dtrain, dvalid

        gc.collect()

    

    return clf
def xgb_predict(X, model):

    xgb_X = xgb.DMatrix(X.values)

    return model.predict(xgb_X)
def kfold_xgb(trn_x, trn_y, num_folds=3):

    

    # Cross validation model

    folds = StratifiedShuffleSplit(n_splits= num_folds, random_state=seed_val)

        

    # Create arrays and dataframes to store results

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(trn_x, trn_y)):

        dtrain = xgb.DMatrix(trn_x.values[train_idx], 

                             trn_y[train_idx])

        dvalid = xgb.DMatrix(trn_x.values[valid_idx], 

                             trn_y[valid_idx])



        # LightGBM parameters found by Bayesian optimization

        n_rounds = 2000

        

        xgb_params = {'eta': 0.05,

                      'max_depth': 6, 

                      'subsample': 0.85, 

                      'colsample_bytree': 0.85,

                      'colsample_bylevel': 0.632,

                      'min_child_weight' : 30,

                      'objective': 'binary:logistic', 

                      'eval_metric': 'auc', 

                      'seed': seed_val,

                      'lambda': 0,

                      'alpha': 0,

                      'silent': 1

                     }

        

        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

        xgb_model = xgb.train(xgb_params, 

                              dtrain, 

                              n_rounds, 

                              watchlist, 

                              verbose_eval=False,

                              early_stopping_rounds=200)



        del dtrain, dvalid

        gc.collect()

    

    return xgb_model
def kfold_cat(trn_x, trn_y, num_folds=3):

    

    # Cross validation model

    folds = StratifiedShuffleSplit(n_splits= num_folds, random_state=seed_val)

        

    # Create arrays and dataframes to store results

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(trn_x, trn_y)):

        cat_X_train, cat_y_train = trn_x.values[train_idx], trn_y[train_idx]

        cat_X_valid, cat_y_valid = trn_x.values[valid_idx], trn_y[valid_idx]



        # Catboost:

        #cb_model = CatBoostClassifier(iterations=1000,

                              #learning_rate=0.1,

                              #depth=7,

                              #l2_leaf_reg=40,

                              #bootstrap_type='Bernoulli',

                              #subsample=0.7,

                              #scale_pos_weight=5,

                              #eval_metric='AUC',

                              #metric_period=50,

                              #od_type='Iter',

                              #od_wait=45,

                              #random_seed=17,

                              #allow_writing_files=False)

        

        cb_model = CatBoostClassifier(iterations=2000,

                                      learning_rate=0.02,

                                      depth=6,

                                      l2_leaf_reg=40,

                                      bootstrap_type='Bernoulli',

                                      subsample=0.8715623,

                                      scale_pos_weight=5,

                                      eval_metric='AUC',

                                      metric_period=50,

                                      od_type='Iter',

                                      od_wait=45,

                                      random_seed=seed_val,

                                     allow_writing_files=False)

        

        cb_model.fit(cat_X_train, cat_y_train,

                     eval_set=(cat_X_valid, cat_y_valid),

                     use_best_model=True,

                     verbose=False)



        del cat_X_train, cat_y_train, cat_X_valid, cat_y_valid 

        gc.collect()

    

    return cb_model
def data_loader(to_load=False):

    

    if not to_load:

        

        df = data_builder()

        feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

        

        # Split to train and test:

        y = df['TARGET']

        X = df[feats]

        X = X.fillna(X.mean()).clip(-1e11,1e11)



        print("X shape: ", X.shape, "    y shape:", y.shape)

        print("\nPreparing data...")



        training = y.notnull()

        testing = y.isnull()

        

        X_train = X.loc[training,:]

        X_test = X.loc[testing,:]

        y_train = y.loc[training]

        

        # Scale:

        scaler = MinMaxScaler()

        scaler.fit(X)

        X_train.loc[:, X_train.columns] = scaler.transform(X_train[X_train.columns])

        X_test.loc[:, X_test.columns] = scaler.transform(X_test[X_test.columns])

        

        print(X_train.shape, X_test.shape, y_train.shape)

        df.to_pickle('df_low_mem.pkl.gz')

        

        del df, X, y, training, testing

        gc.collect()

    

    return X_train, X_test, y_train
print(os.listdir("../input/home-credit-default-risk"))
#app_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
import numpy as np

import pandas as pd

import gc

import time

from contextlib import contextmanager

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



@contextmanager

def timer(title):

    t0 = time.time()

    yield

    print("{} - done in {:.0f}s".format(title, time.time() - t0))



# One-hot encoding for categorical columns with get_dummies

def one_hot_encoder(df, nan_as_category = True):

    original_columns = list(df.columns)

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']

    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)

    new_columns = [c for c in df.columns if c not in original_columns]

    return df, new_columns



# Preprocess application_train.csv and application_test.csv

def application_train_test(num_rows = None, nan_as_category = False):

    # Read data and merge

    df = pd.read_csv('../input/home-credit-default-risk/application_train.csv', nrows= num_rows)

    test_df = pd.read_csv('../input/home-credit-default-risk/application_test.csv', nrows= num_rows)

    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))

    df = df.append(test_df).reset_index()

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)

    df = df[df['CODE_GENDER'] != 'XNA']

    

    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]

    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan

    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)



    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']



    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']

    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']

    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)

    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)

    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])

    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)

    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])

    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)

    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)

    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())

    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']

    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']

    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']

    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']

    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    

    # Categorical features with Binary encode (0 or 1; two categories)

    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:

        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # Categorical features with One-Hot encode

    df, cat_cols = one_hot_encoder(df, nan_as_category)

    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',

    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',

    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 

    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',

    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',

    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',

    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']

    df= df.drop(dropcolum,axis=1)

    del test_df

    gc.collect()

    return df



# Preprocess bureau.csv and bureau_balance.csv

def bureau_and_balance(num_rows = None, nan_as_category = True):

    bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv', nrows = num_rows)

    bb = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv', nrows = num_rows)

    bb, bb_cat = one_hot_encoder(bb, nan_as_category)

    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    

    # Bureau balance: Perform aggregations and merge with bureau.csv

    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}

    for col in bb_cat:

        bb_aggregations[col] = ['mean']

    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)

    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])

    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')

    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)

    del bb, bb_agg

    gc.collect()

    

    # Bureau and bureau_balance numeric features

    num_aggregations = {

        'DAYS_CREDIT': [ 'mean', 'var'],

        'DAYS_CREDIT_ENDDATE': [ 'mean'],

        'DAYS_CREDIT_UPDATE': ['mean'],

        'CREDIT_DAY_OVERDUE': ['mean'],

        'AMT_CREDIT_MAX_OVERDUE': ['mean'],

        'AMT_CREDIT_SUM': [ 'mean', 'sum'],

        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],

        'AMT_CREDIT_SUM_OVERDUE': ['mean'],

        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],

        'AMT_ANNUITY': ['max', 'mean'],

        'CNT_CREDIT_PROLONG': ['sum'],

        'MONTHS_BALANCE_MIN': ['min'],

        'MONTHS_BALANCE_MAX': ['max'],

        'MONTHS_BALANCE_SIZE': ['mean', 'sum']

    }

    # Bureau and bureau_balance categorical features

    cat_aggregations = {}

    for cat in bureau_cat: cat_aggregations[cat] = ['mean']

    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})

    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Bureau: Active credits - using only numerical aggregations

    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]

    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)

    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])

    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')

    del active, active_agg

    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations

    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]

    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)

    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])

    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')

    del closed, closed_agg, bureau

    gc.collect()

    return bureau_agg



# Preprocess previous_applications.csv

def previous_applications(num_rows = None, nan_as_category = True):

    prev = pd.read_csv('../input/home-credit-default-risk/previous_application.csv', nrows = num_rows)

    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)

    # Days 365.243 values -> nan

    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)

    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)

    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)

    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)

    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

    # Add feature: value ask / value received percentage

    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # Previous applications numeric features

    num_aggregations = {

        'AMT_ANNUITY': [ 'max', 'mean'],

        'AMT_APPLICATION': [ 'max','mean'],

        'AMT_CREDIT': [ 'max', 'mean'],

        'APP_CREDIT_PERC': [ 'max', 'mean'],

        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],

        'AMT_GOODS_PRICE': [ 'max', 'mean'],

        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],

        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],

        'DAYS_DECISION': [ 'max', 'mean'],

        'CNT_PAYMENT': ['mean', 'sum'],

    }

    # Previous applications categorical features

    cat_aggregations = {}

    for cat in cat_cols:

        cat_aggregations[cat] = ['mean']

    

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})

    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Previous Applications: Approved Applications - only numerical features

    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]

    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)

    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])

    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Previous Applications: Refused Applications - only numerical features

    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]

    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)

    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])

    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    del refused, refused_agg, approved, approved_agg, prev

    gc.collect()

    return prev_agg



# Preprocess POS_CASH_balance.csv

def pos_cash(num_rows = None, nan_as_category = True):

    pos = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv', nrows = num_rows)

    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)

    # Features

    aggregations = {

        'MONTHS_BALANCE': ['max', 'mean', 'size'],

        'SK_DPD': ['max', 'mean'],

        'SK_DPD_DEF': ['max', 'mean']

    }

    for cat in cat_cols:

        aggregations[cat] = ['mean']

    

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)

    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # Count pos cash accounts

    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()

    del pos

    gc.collect()

    return pos_agg

    

# Preprocess installments_payments.csv

def installments_payments(num_rows = None, nan_as_category = True):

    ins = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv', nrows = num_rows)

    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)

    # Percentage and difference paid in each installment (amount paid and installment value)

    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']

    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    # Days past due and days before due (no negative values)

    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']

    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']

    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)

    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    # Features: Perform aggregations

    aggregations = {

        'NUM_INSTALMENT_VERSION': ['nunique'],

        'DPD': ['max', 'mean', 'sum','min','std' ],

        'DBD': ['max', 'mean', 'sum','min','std'],

        'PAYMENT_PERC': [ 'max','mean',  'var','min','std'],

        'PAYMENT_DIFF': [ 'max','mean', 'var','min','std'],

        'AMT_INSTALMENT': ['max', 'mean', 'sum','min','std'],

        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum','std'],

        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum','std']

    }

    for cat in cat_cols:

        aggregations[cat] = ['mean']

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)

    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    # Count installments accounts

    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()

    del ins

    gc.collect()

    return ins_agg



# Preprocess credit_card_balance.csv

def credit_card_balance(num_rows = None, nan_as_category = True):

    cc = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv', nrows = num_rows)

    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)

    # General aggregations

    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)

    cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'sum', 'var'])

    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

    # Count credit card lines

    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

    del cc

    gc.collect()

    return cc_agg
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtypes

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        #else: df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
def data_builder():

    

    num_rows = None

    df = application_train_test(num_rows)

    with timer("Process bureau and bureau_balance"):

        bureau = bureau_and_balance(num_rows)

        print("Bureau df shape:", bureau.shape)

        df = df.join(bureau, how='left', on='SK_ID_CURR')

        del bureau

        gc.collect()

    with timer("Process previous_applications"):

        prev = previous_applications(num_rows)

        print("Previous applications df shape:", prev.shape)

        df = df.join(prev, how='left', on='SK_ID_CURR')

        del prev

        gc.collect()

    with timer("Process POS-CASH balance"):

        pos = pos_cash(num_rows)

        print("Pos-cash balance df shape:", pos.shape)

        df = df.join(pos, how='left', on='SK_ID_CURR')

        del pos

        gc.collect()

    with timer("Process installments payments"):

        ins = installments_payments(num_rows)

        print("Installments payments df shape:", ins.shape)

        df = df.join(ins, how='left', on='SK_ID_CURR')

        del ins

        gc.collect()

    with timer("Process credit card balance"):

        cc = credit_card_balance(num_rows)

        print("Credit card balance df shape:", cc.shape)

        df = df.join(cc, how='left', on='SK_ID_CURR')

        del cc

        gc.collect()

        

    df.set_index('SK_ID_CURR', inplace=True, drop=False)

    df = df.drop(labels='index', axis=1)

    df = reduce_mem_usage(df)    

    df.to_pickle('df_low_mem.pkl.gz')

    

    return df
# Fix random seed:

seed_val = 42



# Load data:

train_x, test_x, train_y = data_loader()
estimators = ['cat','lgb', 'xgb','ridge','f10_dnn']

pred_cols = ['pred_cat','pred_lgb','pred_xgb','ridge','f10_dnn']
#Holdout

from sklearn.model_selection import train_test_split

x_train, x_hold, y_train, y_hold = train_test_split(train_x, train_y, test_size=0.1, random_state=seed_val)
n_folds = 2

tr_blend, va_blend, tst_blend, hold_blend, m_list = oof_regression_stacker(x_train, y_train, test_x, 

                                                                           n_folds = 2, 

                                                                           estimators=estimators, 

                                                                           pred_cols = pred_cols,

                                                                           train_eval_metric=roc_auc_score,

                                                                           compare_eval_metric=roc_auc_score,

                                                                           debug = True, holdout_x = x_hold)