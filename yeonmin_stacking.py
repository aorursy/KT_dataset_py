# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgbm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, BatchNormalization, Activation 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

import regex as re
import gc
# Any results you write to the current directory are saved as output.
baseline_tree_score = 0.23092278864723115
baseline_neuralnetwork_score = 0.5480561937041435
train = pd.read_csv('../input/kaggletutorial/covertype_train.csv')
test = pd.read_csv('../input/kaggletutorial/covertype_test.csv')
train_index = train.shape[0]
lgbm_param =  {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    "learning_rate": 0.06,
    "num_leaves": 16,
    "max_depth": 6,
    "colsample_bytree": 0.7,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "nthread":8
}
def keras_model(input_dims):
    model = Sequential()
    
    model.add(Dense(input_dims, input_dim=input_dims))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(input_dims//2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    # output layer (y_pred)
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    # compile this model
    model.compile(loss='binary_crossentropy', # one may use 'mean_absolute_error' as alternative
                  optimizer='adam', metrics=['accuracy'])
    return model

def keras_history_plot(history):
    plt.plot(history.history['loss'], 'y', label='train loss')
    plt.plot(history.history['val_loss'], 'r', label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()
def baseline_tree_cv(train):
    train_df = train.copy()
    y_value = train_df["Cover_Type"]
    del train_df["Cover_Type"], train_df["ID"]
    
    NFOLD = 5
    folds = StratifiedKFold(n_splits= NFOLD, shuffle=True, random_state=2018)

    total_score = 0
    best_iteration = 0
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, y_value)):
        train_x, train_y = train_df.iloc[train_idx], y_value.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], y_value.iloc[valid_idx]

        evals_result_dict = {} 
        dtrain = lgbm.Dataset(train_x, label=train_y)
        dvalid = lgbm.Dataset(valid_x, label=valid_y)

        clf = lgbm.train(lgbm_param, train_set=dtrain, num_boost_round=3000, valid_sets=[dtrain, dvalid],
                               early_stopping_rounds=200, evals_result=evals_result_dict, verbose_eval=500)

        predict = clf.predict(valid_x)
        cv_score = log_loss(valid_y, predict )
        total_score += cv_score
        best_iteration = max(best_iteration, clf.best_iteration)
        print('Fold {} LogLoss : {}'.format(n_fold + 1, cv_score ))
        lgbm.plot_metric(evals_result_dict)
        plt.show()
        
    print("Best Iteration", best_iteration)
    print("Total LogLoss", total_score / NFOLD)
    print("Baseline model Score Diff", total_score / NFOLD - baseline_tree_score)
    
    del train_df
    
    return best_iteration

def baseline_keras_cv(train):
    train_df = train.copy()
    y_value = train_df['Cover_Type']
    del train_df['Cover_Type'], train_df['ID']
    
    model = keras_model(train_df.shape[1])
    callbacks = [
            EarlyStopping(
                patience=10,
                verbose=10)
        ]

    NFOLD = 5
    folds = StratifiedKFold(n_splits= NFOLD, shuffle=True, random_state=2018)

    total_score = 0
    best_epoch = 0
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, y_value)):
        train_x, train_y = train_df.iloc[train_idx], y_value.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], y_value.iloc[valid_idx]

        history = model.fit(train_x.values, train_y.values, nb_epoch=30, batch_size = 64, validation_data=(valid_x.values, valid_y.values), 
                            verbose=1, callbacks=callbacks)

        keras_history_plot(history)
        predict = model.predict(valid_x.values)
        null_count = np.sum(pd.isnull(predict) )
        if null_count > 0:
            print("Null Prediction Error: ", null_count)
            predict[pd.isnull(predict)] = predict[~pd.isnull(predict)].mean()

        cv_score = log_loss(valid_y, predict )
        total_score += cv_score
        best_epoch = max(best_epoch, np.max(history.epoch))
        print('Fold {} LogLoss : {}'.format(n_fold + 1, cv_score ))
        
    print("Best Epoch: ", best_epoch)
    print("Total LogLoss", total_score/NFOLD)
    print("Baseline model Score Diff", total_score/NFOLD - baseline_neuralnetwork_score)
def outlier_binary(frame, col, outlier_range):
    outlier_feature = col + '_Outlier'
    frame[outlier_feature] = 0
    frame.loc[frame[col] > outlier_range, outlier_feature] = 1
    return frame

def outlier_divide_ratio(frame, col, outlier_range):
    outlier_index = frame[col] >= outlier_range
    outlier_median =  frame.loc[outlier_index, col].median()
    normal_median = frame.loc[frame[col] < outlier_range, col].median()
    outlier_ratio = outlier_median / normal_median
    
    frame.loc[outlier_index, col] = frame.loc[outlier_index, col]/outlier_ratio
    return frame

def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0] 
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequncy'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')

def binning_category_combine_feature(frame, col1, col2, col1_quantile, col2_quantile):
    print(col1, ' ', col2, 'Bining Combine')
    col1_quantile = np.arange(0,1.1,col1_quantile)
    col2_quantile = np.arange(0,1.1,col2_quantile)
    
    col1_label = '{}_quantile_label'.format(col1)
    frame[col1_label] = pd.qcut(frame[col1], q=col1_quantile, labels = ['{}_quantile_{:.1f}'.format(col1, col) for col in col1_quantile][1:])
    
    col2_label = '{}_quantile_label'.format(col2)
    frame[col2_label] = pd.qcut(frame[col2], q=col2_quantile, labels = ['{}_quantile_{:.1f}'.format(col2, col) for col in col2_quantile][1:])
    
    combine_label = 'Binnig_{}_{}_Combine'.format(col1, col2)
    frame[combine_label] = frame[[col1_label, col2_label]].apply(lambda row: row[col1_label] +'_'+ row[col2_label] ,axis=1)
    for col in [col1_label, col2_label, combine_label]:
        frame[col] = frame[col].factorize()[0]
    
    # del frame[col1_label], frame[col2_label]
    gc.collect()
    return frame, [col1_label, col2_label, combine_label]
def tree_data_preprocessing(train, test):
    train_index = train.shape[0]
    all_data = pd.concat([train, test])
    del all_data['oil_Type']

    all_column_set = set(all_data.columns)
    category_feature = []
    for col in all_data.loc[:, all_data.dtypes=='object'].columns:
        all_data[col] = all_data[col].factorize()[0]
        category_feature.append(col)

    numerical_feature = list(all_column_set - set(category_feature) - set(['Cover_Type','ID']))

    all_data['Elevation'] = np.log1p(all_data['Elevation'])

    all_data = outlier_binary(all_data, 'Horizontal_Distance_To_Fire_Points', 10000)
    all_data = outlier_binary(all_data, 'Horizontal_Distance_To_Roadways', 10000)

    all_data = outlier_divide_ratio(all_data, 'Horizontal_Distance_To_Fire_Points', 10000)
    all_data = outlier_divide_ratio(all_data, 'Horizontal_Distance_To_Roadways', 10000)

    all_data = frequency_encoding(all_data, 'Soil_Type')
    all_data = frequency_encoding(all_data, 'Wilderness_Area')

    aspect_train = all_data.loc[all_data['Aspect'].notnull()]
    aspect_test = all_data.loc[all_data['Aspect'].isnull()]
    del aspect_train["Cover_Type"], aspect_train['ID']
    del aspect_test["Cover_Type"], aspect_test['ID']

    numerical_feature_woaspect = numerical_feature[:]
    numerical_feature_woaspect.remove('Aspect')

    sc = StandardScaler()
    aspect_train[numerical_feature_woaspect] = sc.fit_transform(aspect_train[numerical_feature_woaspect])
    aspect_test[numerical_feature_woaspect] = sc.transform(aspect_test[numerical_feature_woaspect] )

    y_value = aspect_train['Aspect']
    del aspect_train['Aspect'], aspect_test['Aspect']
    
    knn = KNeighborsRegressor(n_neighbors=7)
    knn.fit(aspect_train,y_value)
    predict = knn.predict(aspect_test)
    
    sns.distplot(predict)
    sns.distplot(all_data['Aspect'].dropna())
    plt.title('KNN Aspect Null Imputation')
    plt.show()
    
    all_data.loc[all_data['Aspect'].isnull(),'Aspect'] = predict
    
    all_data['Horizontal_Distance_To_Hydrology'] = all_data['Horizontal_Distance_To_Hydrology']/1000
    all_data['HF1'] = all_data['Horizontal_Distance_To_Hydrology'] + all_data['Horizontal_Distance_To_Fire_Points']
    all_data['HF2'] = all_data['Horizontal_Distance_To_Hydrology'] - all_data['Horizontal_Distance_To_Fire_Points']
    all_data['HF3'] = np.log1p(all_data['Horizontal_Distance_To_Hydrology'] * all_data['Horizontal_Distance_To_Fire_Points'])
    all_data['HF4'] = all_data['Horizontal_Distance_To_Hydrology'] / all_data['Horizontal_Distance_To_Fire_Points']

    all_data['HR1'] = all_data['Horizontal_Distance_To_Hydrology'] + all_data['Horizontal_Distance_To_Roadways']
    all_data['HR2'] = all_data['Horizontal_Distance_To_Hydrology'] - all_data['Horizontal_Distance_To_Roadways']
    all_data['HR3'] = np.log1p(all_data['Horizontal_Distance_To_Hydrology'] * all_data['Horizontal_Distance_To_Roadways'])
    all_data['HR4'] = all_data['Horizontal_Distance_To_Hydrology'] / all_data['Horizontal_Distance_To_Roadways']

    all_data['HH1'] = all_data['Horizontal_Distance_To_Hydrology'] + all_data['Vertical_Distance_To_Hydrology']
    all_data['HH2'] = all_data['Horizontal_Distance_To_Hydrology'] - all_data['Vertical_Distance_To_Hydrology']
    all_data['HH3'] = np.log1p(abs(all_data['Horizontal_Distance_To_Hydrology'] * all_data['Vertical_Distance_To_Hydrology']))
    all_data['HH4'] = all_data['Horizontal_Distance_To_Hydrology'] / all_data['Vertical_Distance_To_Hydrology']

    all_data['FR1'] = all_data['Horizontal_Distance_To_Fire_Points'] + all_data['Horizontal_Distance_To_Roadways']
    all_data['FR2'] = all_data['Horizontal_Distance_To_Fire_Points'] - all_data['Horizontal_Distance_To_Roadways']
    all_data['FR3'] = np.log1p(all_data['Horizontal_Distance_To_Fire_Points'] * all_data['Horizontal_Distance_To_Roadways'])
    all_data['FR4'] = all_data['Horizontal_Distance_To_Fire_Points'] / all_data['Horizontal_Distance_To_Roadways']
    
    all_data['Direct_Distance_Hydrology'] = (all_data['Horizontal_Distance_To_Hydrology']**2+all_data['Vertical_Distance_To_Hydrology']**2)**0.5
    
    all_data.loc[np.isinf(all_data['HF4']),'HF4'] = 0
    all_data.loc[np.isinf(all_data['HR4']),'HR4'] = 0
    all_data.loc[np.isinf(all_data['HH4']),'HH4'] = 0
    all_data.loc[np.isinf(all_data['FR4']),'FR4'] = 0
    all_data[['HF4','HH4']] = all_data[['HF4','HH4']].fillna(0)
    
    all_data, new_col = binning_category_combine_feature(all_data, 'Elevation', 'Aspect', 0.1, 0.1) 
    for col in new_col:
        all_data = frequency_encoding(all_data, col)
        
    train_df = all_data.iloc[:train_index]
    test_df = all_data.iloc[train_index:]
    
    soil_mean_encoding = train_df.groupby(['Soil_Type'])['Cover_Type'].agg({'Soil_Type_Mean':'mean', 
                                                                        'Soil_Type_Std':'std', 
                                                                        'Soil_Type_Size':'size', 
                                                                        'Soil_Type_Sum':'sum'}).reset_index()
    train_df = train_df.merge(soil_mean_encoding, on='Soil_Type', how='left')
    test_df = test_df.merge(soil_mean_encoding, on='Soil_Type', how='left')
    
    wildness_mean_encoding = train_df.groupby(['Wilderness_Area'])['Cover_Type'].agg({'Wilderness_Area_Mean':'mean', 
                                                                              'Wilderness_Area_Std':'std', 
                                                                              'Wilderness_Area_Size':'size', 
                                                                              'Wilderness_Area_Sum':'sum'}).reset_index()
    train_df = train_df.merge(wildness_mean_encoding, on='Wilderness_Area', how='left')
    test_df = test_df.merge(wildness_mean_encoding, on='Wilderness_Area', how='left')
    
    del all_data, predict, aspect_train, aspect_test
    gc.collect()
    
    return train_df, test_df
def nn_data_preprocessing(train, test):
    train_index = train.shape[0]
    all_data = pd.concat([train, test])
    del all_data['oil_Type']

    all_column_set = set(all_data.columns)
    category_feature = []
    for col in all_data.loc[:, all_data.dtypes=='object'].columns:
        all_data[col] = all_data[col].factorize()[0]
        category_feature.append(col)
    
    numerical_feature = list(all_column_set - set(category_feature) - set(['Cover_Type','ID']))
    
    all_data['Elevation'] = np.log1p(all_data['Elevation'])

    all_data = outlier_binary(all_data, 'Horizontal_Distance_To_Fire_Points', 10000)
    all_data = outlier_binary(all_data, 'Horizontal_Distance_To_Roadways', 10000)

    all_data = outlier_divide_ratio(all_data, 'Horizontal_Distance_To_Fire_Points', 10000)
    all_data = outlier_divide_ratio(all_data, 'Horizontal_Distance_To_Roadways', 10000)

    all_data = frequency_encoding(all_data, 'Soil_Type')
    all_data = frequency_encoding(all_data, 'Wilderness_Area')

    aspect_train = all_data.loc[all_data['Aspect'].notnull()]
    aspect_test = all_data.loc[all_data['Aspect'].isnull()]
    del aspect_train["Cover_Type"], aspect_train['ID']
    del aspect_test["Cover_Type"], aspect_test['ID']

    numerical_feature_woaspect = numerical_feature[:]
    numerical_feature_woaspect.remove('Aspect')

    sc = StandardScaler()
    aspect_train[numerical_feature_woaspect] = sc.fit_transform(aspect_train[numerical_feature_woaspect])
    aspect_test[numerical_feature_woaspect] = sc.transform(aspect_test[numerical_feature_woaspect] )

    y_value = aspect_train['Aspect']
    del aspect_train['Aspect'], aspect_test['Aspect']

    knn = KNeighborsRegressor(n_neighbors=7)
    knn.fit(aspect_train,y_value)
    predict = knn.predict(aspect_test)

    sns.distplot(predict)
    sns.distplot(all_data['Aspect'].dropna())
    plt.title('KNN Aspect Null Imputation')
    plt.show()

    all_data.loc[all_data['Aspect'].isnull(),'Aspect'] = predict
    
    all_data['Horizontal_Distance_To_Hydrology'] = all_data['Horizontal_Distance_To_Hydrology']/1000
    all_data['HF1'] = all_data['Horizontal_Distance_To_Hydrology'] + all_data['Horizontal_Distance_To_Fire_Points']
    all_data['HF2'] = all_data['Horizontal_Distance_To_Hydrology'] - all_data['Horizontal_Distance_To_Fire_Points']
    all_data['HF3'] = np.log1p(all_data['Horizontal_Distance_To_Hydrology'] * all_data['Horizontal_Distance_To_Fire_Points'])
    all_data['HF4'] = all_data['Horizontal_Distance_To_Hydrology'] / all_data['Horizontal_Distance_To_Fire_Points']

    all_data['HR1'] = all_data['Horizontal_Distance_To_Hydrology'] + all_data['Horizontal_Distance_To_Roadways']
    all_data['HR2'] = all_data['Horizontal_Distance_To_Hydrology'] - all_data['Horizontal_Distance_To_Roadways']
    all_data['HR3'] = np.log1p(all_data['Horizontal_Distance_To_Hydrology'] * all_data['Horizontal_Distance_To_Roadways'])
    all_data['HR4'] = all_data['Horizontal_Distance_To_Hydrology'] / all_data['Horizontal_Distance_To_Roadways']

    all_data['HH1'] = all_data['Horizontal_Distance_To_Hydrology'] + all_data['Vertical_Distance_To_Hydrology']
    all_data['HH2'] = all_data['Horizontal_Distance_To_Hydrology'] - all_data['Vertical_Distance_To_Hydrology']
    all_data['HH3'] = np.log1p(abs(all_data['Horizontal_Distance_To_Hydrology'] * all_data['Vertical_Distance_To_Hydrology']))
    all_data['HH4'] = all_data['Horizontal_Distance_To_Hydrology'] / all_data['Vertical_Distance_To_Hydrology']

    all_data['FR1'] = all_data['Horizontal_Distance_To_Fire_Points'] + all_data['Horizontal_Distance_To_Roadways']
    all_data['FR2'] = all_data['Horizontal_Distance_To_Fire_Points'] - all_data['Horizontal_Distance_To_Roadways']
    all_data['FR3'] = np.log1p(all_data['Horizontal_Distance_To_Fire_Points'] * all_data['Horizontal_Distance_To_Roadways'])
    all_data['FR4'] = all_data['Horizontal_Distance_To_Fire_Points'] / all_data['Horizontal_Distance_To_Roadways']

    all_data['Direct_Distance_Hydrology'] = (all_data['Horizontal_Distance_To_Hydrology']**2+all_data['Vertical_Distance_To_Hydrology']**2)**0.5
    
    all_data.loc[np.isinf(all_data['HF4']),'HF4'] = 0
    all_data.loc[np.isinf(all_data['HR4']),'HR4'] = 0
    all_data.loc[np.isinf(all_data['HH4']),'HH4'] = 0
    all_data.loc[np.isinf(all_data['FR4']),'FR4'] = 0
    
    all_data[['HF4','HH4']] = all_data[['HF4','HH4']].fillna(0)
    
    all_data, new_col = binning_category_combine_feature(all_data, 'Elevation', 'Aspect', 0.1, 0.1)
    
    for col in new_col:
        all_data = frequency_encoding(all_data, col)
        
    all_data.drop(columns=new_col,axis=1,inplace=True)
    
    before_one_hot = set(all_data.columns)
    for col in category_feature:
        all_data = pd.concat([all_data,pd.get_dummies(all_data[col],prefix=col)],axis=1)
        
    one_hot_feature = set(all_data.columns) - before_one_hot
    
    train_df = all_data.iloc[:train_index]
    test_df = all_data.iloc[train_index:]
    
    soil_mean_encoding = train_df.groupby(['Soil_Type'])['Cover_Type'].agg({'Soil_Type_Mean':'mean', 
                                                                        'Soil_Type_Std':'std', 
                                                                        'Soil_Type_Size':'size', 
                                                                        'Soil_Type_Sum':'sum'}).reset_index()
    train_df = train_df.merge(soil_mean_encoding, on='Soil_Type', how='left')
    test_df = test_df.merge(soil_mean_encoding, on='Soil_Type', how='left')
    
    wildness_mean_encoding = train_df.groupby(['Wilderness_Area'])['Cover_Type'].agg({'Wilderness_Area_Mean':'mean', 
                                                                              'Wilderness_Area_Std':'std', 
                                                                              'Wilderness_Area_Size':'size', 
                                                                              'Wilderness_Area_Sum':'sum'}).reset_index()
    train_df = train_df.merge(wildness_mean_encoding, on='Wilderness_Area', how='left')
    test_df = test_df.merge(wildness_mean_encoding, on='Wilderness_Area', how='left')
    
    train_df.drop(columns=category_feature, axis=1, inplace=True)
    test_df.drop(columns=category_feature, axis=1, inplace=True)
    
    scale_feature = list(set(train_df.columns)-one_hot_feature-set(['Cover_Type','ID']))
    sc = StandardScaler()
    train_df[scale_feature] = sc.fit_transform(train_df[scale_feature])
    test_df[scale_feature] = sc.transform(test_df[scale_feature] )
    
    return train_df, test_df
org_train_df, org_test_df = tree_data_preprocessing(train, test)
train_df = org_train_df.copy()
test_df = org_test_df.copy()
lgbm_param =  {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    "learning_rate": 0.03,
    "num_leaves": 24,
    "max_depth": 6,
    "colsample_bytree": 0.65,
    "subsample": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "nthread":8
}
y_value = train_df["Cover_Type"]
del train_df["Cover_Type"], train_df["ID"]
del test_df["Cover_Type"], test_df["ID"]
""" 시간관계상 CV를 돌리지는 않겠습니다.
NFOLD = 5
folds = StratifiedKFold(n_splits= NFOLD, shuffle=True, random_state=2018)

total_score = 0
best_iteration = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, y_value)):
    train_x, train_y = train_df.iloc[train_idx], y_value.iloc[train_idx]
    valid_x, valid_y = train_df.iloc[valid_idx], y_value.iloc[valid_idx]

    evals_result_dict = {} 
    dtrain = lgbm.Dataset(train_x, label=train_y)
    dvalid = lgbm.Dataset(valid_x, label=valid_y)

    clf = lgbm.train(lgbm_param, train_set=dtrain, num_boost_round=5000, valid_sets=[dtrain, dvalid],
                           early_stopping_rounds=200, evals_result=evals_result_dict, verbose_eval=500)

    predict = clf.predict(valid_x)
    cv_score = log_loss(valid_y, predict )
    total_score += cv_score
    best_iteration = max(best_iteration, clf.best_iteration)
    print('Fold {} LogLoss : {}'.format(n_fold + 1, cv_score ))
    lgbm.plot_metric(evals_result_dict)
    plt.show()

print("Best Iteration", best_iteration)
print("Total LogLoss", total_score / NFOLD)
print("Baseline model Score Diff", total_score / NFOLD - baseline_tree_score)
"""
dtrain = lgbm.Dataset(train_df, label=y_value)
clf = lgbm.train(lgbm_param, train_set=dtrain, num_boost_round=5000)
predict = clf.predict(test_df)
submission = pd.read_csv('../input/kaggletutorial/sample_submission.csv')
submission['Cover_Type'] = predict
submission.to_csv('lgbm_last.csv', index=False)
org_train_df, org_test_df = nn_data_preprocessing(train, test)
train_df = org_train_df.copy()
test_df = org_test_df.copy()
def keras_model(input_dims):
    model = Sequential()
    
    model.add(Dense(input_dims, input_dim=input_dims))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(input_dims))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(input_dims//2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(input_dims//5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    # output layer (y_pred)
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    # compile this model
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
    return model


y_value = train_df['Cover_Type']
del train_df['Cover_Type'], train_df['ID']
del test_df['Cover_Type'], test_df['ID']

model = keras_model(train_df.shape[1])
callbacks = [
        EarlyStopping(
            patience=10,
            verbose=10)
    ]
""" 시간관계상 CV를 돌리지는 않겠습니다.
NFOLD = 5
folds = StratifiedKFold(n_splits= NFOLD, shuffle=True, random_state=2018)

total_score = 0
best_epoch = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, y_value)):
    train_x, train_y = train_df.iloc[train_idx], y_value.iloc[train_idx]
    valid_x, valid_y = train_df.iloc[valid_idx], y_value.iloc[valid_idx]

    history = model.fit(train_x.values, train_y.values, nb_epoch=30, batch_size = 64, validation_data=(valid_x.values, valid_y.values), 
                        verbose=1, callbacks=callbacks)

    keras_history_plot(history)
    predict = model.predict(valid_x.values)
    null_count = np.sum(pd.isnull(predict) )
    if null_count > 0:
        print("Null Prediction Error: ", null_count)
        predict[pd.isnull(predict)] = predict[~pd.isnull(predict)].mean()

    cv_score = log_loss(valid_y, predict )
    total_score += cv_score
    best_epoch = max(best_epoch, np.max(history.epoch))
    print('Fold {} LogLoss : {}'.format(n_fold + 1, cv_score ))

print("Best Epoch: ", best_epoch)
print("Total LogLoss", total_score/NFOLD)
print("Baseline model Score Diff", total_score/NFOLD - baseline_neuralnetwork_score)
"""
history = model.fit(train_df.values, y_value.values, nb_epoch=30, batch_size = 64, verbose=1)
predict = model.predict(test_df.values)
submission_nn = pd.read_csv('../input/kaggletutorial/sample_submission.csv')
submission_nn['Cover_Type'] = predict
submission_nn.to_csv('nn_last.csv', index=False)
def calculate_correlation(base_df, target_df):
    source = base_df.copy()
    source = source.merge(target_df,on='ID')
    corr_df = source.corr()
    corr = corr_df.ix['Cover_Type_x']['Cover_Type_y']
    del corr_df, source
    return corr
source = submission.copy()
source = source.merge(submission_nn,on='ID')
source
calculate_correlation(submission, submission_nn)
class SklearnWrapper(object):
    def __init__(self, clf, params=None, **kwargs):
        seed = kwargs.get('seed', 0)
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train, x_cross=None, y_cross=None):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]
class LgbmWrapper(object):
    def __init__(self, params=None, **kwargs):
        seed = kwargs.get('seed', 0)
        num_rounds = kwargs.get('num_rounds', 1000)
        early_stopping = kwargs.get('ealry_stopping', 100)
        eval_function = kwargs.get('eval_function', None)
        verbose_eval = kwargs.get('verbose_eval', 100)

        self.param = params
        self.param['seed'] = seed
        self.num_rounds = num_rounds
        self.early_stopping = early_stopping
        self.eval_function = eval_function
        self.verbose_eval = verbose_eval

    def train(self, x_train, y_train, x_cross=None, y_cross=None):
        need_cross_validation = True
        if x_cross is None:
            need_cross_validation = False

        if isinstance(y_train, pd.DataFrame) is True:
            y_train = y_train[y_train.columns[0]]
            if need_cross_validation is True:
                y_cross = y_cross[y_cross.columns[0]]

        if need_cross_validation is True:
            dtrain = lgbm.Dataset(x_train, label=y_train, silent=True)
            dvalid = lgbm.Dataset(x_cross, label=y_cross, silent=True)
            self.clf = lgbm.train(self.param, train_set=dtrain, num_boost_round=self.num_rounds, valid_sets=dvalid,
                                  feval=self.eval_function, early_stopping_rounds=self.early_stopping,
                                  verbose_eval=self.verbose_eval)
        else:
            dtrain = lgbm.Dataset(x_train, label=y_train, silent= True)
            self.clf = lgbm.train(self.param, dtrain, self.num_rounds)

    def predict(self, x):
        return self.clf.predict(x, num_iteration=self.clf.best_iteration)

    def get_params(self):
        return self.param
def get_oof(clf, x_train, y_train, x_test, eval_func, **kwargs):
    nfolds = kwargs.get('NFOLDS', 5)
    kfold_shuffle = kwargs.get('kfold_shuffle', True)
    kfold_random_state = kwargs.get('kfold_random_sate', 0)

    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]

    kf = StratifiedKFold(n_splits= nfolds, shuffle=kfold_shuffle, random_state=kfold_random_state)

    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((nfolds, ntest))

    cv_sum = 0
    try:
        if clf.clf is not None:
            print(clf.clf)
    except:
        print(clf)
        print(clf.get_params())
    
    for i, (train_index, cross_index) in enumerate(kf.split(x_train, y_train)):
        x_tr, x_cr = None, None
        y_tr, y_cr = None, None
        if isinstance(x_tr, pd.DataFrame):
            x_tr, x_cr = x_train.iloc[train_index], x_train.iloc[cross_index]
            y_tr, y_cr = y_train.iloc[train_index], y_train.iloc[cross_index]
        else:
            x_tr, x_cr = x_train[train_index], x_train[cross_index]
            y_tr, y_cr = y_train[train_index], y_train[cross_index]

        clf.train(x_tr, y_tr, x_cr, y_cr)
        oof_train[cross_index] = clf.predict(x_cr)
        cv_score = eval_func(y_cr, oof_train[cross_index])

        print('Fold %d / ' % (i+1), 'CV-Score: %.6f' % cv_score)
        cv_sum = cv_sum + cv_score

    score = cv_sum / nfolds
    print("Average CV-Score: ", score)
    # Using All Dataset, retrain
    clf.train(x_train, y_train)
    oof_test = clf.predict(x_test)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
lgbm_param1 =  {
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': 'binary_logloss',
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 7,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "nthread":8,
    'drop_rate':0.1, 
    'skip_drop':0.5,
    'max_drop':50, 
    'top_rate':0.1, 
    'other_rate':0.1
}

lgbm_param2 =  {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    "learning_rate": 0.03,
    "num_leaves": 10,
    "max_depth": 4,
    "colsample_bytree": 0.5,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "nthread":8
}

lgbm_param3 =  {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    "learning_rate": 0.03,
    "num_leaves": 24,
    "max_depth": 6,
    "colsample_bytree": 0.5,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "nthread":8
}

rf_params = {
    'criterion':'gini', 'max_leaf_nodes':24, 'n_estimators':200, 'min_impurity_split':0.0000001,
    'max_features':0.4, 'max_depth':6, 'min_samples_leaf':20, 'min_samples_split':2,
    'min_weight_fraction_leaf':0.0, 'bootstrap':True,
    'random_state':1, 'verbose':False
    
}

et_parmas = {
    'criterion':'gini', 'max_leaf_nodes':31, 'n_estimators':200, 'min_impurity_split':0.0000001,
    'max_features':0.6, 'max_depth':10, 'min_samples_leaf':20, 'min_samples_split':2,
    'min_weight_fraction_leaf':0.0, 'bootstrap':True,
    'random_state':1, 'verbose':False 
}
org_train_df, org_test_df = tree_data_preprocessing(train, test)
train_df = org_train_df.copy()
test_df = org_test_df.copy()
y_value = train_df["Cover_Type"]
del train_df["Cover_Type"], train_df["ID"]
del test_df["Cover_Type"], test_df["ID"]
et_model = SklearnWrapper(clf = ExtraTreesClassifier, params=et_parmas)
rf_model = SklearnWrapper(clf = RandomForestClassifier, params=rf_params)
et_train, et_test = get_oof(et_model, train_df.values, y_value, test_df.values, log_loss, NFOLDS=3)
rf_train, rf_test = get_oof(rf_model, train_df.values, y_value, test_df.values, log_loss, NFOLDS=3)
x_train_second_layer = np.concatenate((rf_train, et_train), axis=1)
x_test_second_layer = np.concatenate((rf_test, et_test), axis=1)
x_train = pd.DataFrame(x_train_second_layer)
x_test = pd.DataFrame(x_test_second_layer)
lgbm_meta_params = {
    'boosting':'gbdt', 'num_leaves':28, 'learning_rate':0.03, 'min_sum_hessian_in_leaf':0.1,
    'max_depth':7, 'feature_fraction':0.6, 'min_data_in_leaf':30, 'poission_max_delta_step':0.7,
    'bagging_fraction':0.8, 'min_gain_to_split':0, 
    'objective':'binary', 'seed':1,'metric': 'binary_logloss'
}

NFOLD = 3
folds = StratifiedKFold(n_splits= NFOLD, shuffle=True, random_state=2018)

total_score = 0
best_iteration = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x_train, y_value)):
    train_x, train_y = train_df.iloc[train_idx], y_value.iloc[train_idx]
    valid_x, valid_y = train_df.iloc[valid_idx], y_value.iloc[valid_idx]

    evals_result_dict = {} 
    dtrain = lgbm.Dataset(train_x, label=train_y)
    dvalid = lgbm.Dataset(valid_x, label=valid_y)

    clf = lgbm.train(lgbm_meta_params, train_set=dtrain, num_boost_round=5000, valid_sets=[dtrain, dvalid],
                           early_stopping_rounds=200, evals_result=evals_result_dict, verbose_eval=500)

    predict = clf.predict(valid_x)
    cv_score = log_loss(valid_y, predict )
    total_score += cv_score
    best_iteration = max(best_iteration, clf.best_iteration)
    print('Fold {} LogLoss : {}'.format(n_fold + 1, cv_score ))
    lgbm.plot_metric(evals_result_dict)
    plt.show()

print("Best Iteration", best_iteration)
print("Total LogLoss", total_score / NFOLD)
print("Baseline model Score Diff", total_score / NFOLD - baseline_tree_score)
dtrain = lgbm.Dataset(x_train, label=y_value)
clf = lgbm.train(lgbm_meta_params, train_set=dtrain, num_boost_round=5000)
predict_stacking = clf.predict(x_test)
submission_stacking = pd.read_csv('../input/kaggletutorial/sample_submission.csv')
submission_stacking['Cover_Type'] = predict_stacking
submission_stacking.to_csv('submission_stacking.csv', index=False)
submission_et = pd.read_csv('../input/kaggletutorial/sample_submission.csv')
submission_et['Cover_Type'] = et_test
submission_et.to_csv('submission_et.csv', index=False)
submission_rf = pd.read_csv('../input/kaggletutorial/sample_submission.csv')
submission_rf['Cover_Type'] = rf_test
submission_rf.to_csv('submission_rf.csv', index=False)