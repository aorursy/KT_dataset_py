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
train_df, test_df = tree_data_preprocessing(train, test)
soil_mean_encoding = train_df.groupby(['Soil_Type'])['Cover_Type'].agg({'Soil_Type_Mean':'mean', 
                                                                        'Soil_Type_Std':'std', 
                                                                        'Soil_Type_Size':'size', 
                                                                        'Soil_Type_Sum':'sum'}).reset_index()
train_df = train_df.merge(soil_mean_encoding, on='Soil_Type', how='left')
test_df = test_df.merge(soil_mean_encoding, on='Soil_Type', how='left')
baseline_tree_cv(train_df)
wildness_mean_encoding = train_df.groupby(['Wilderness_Area'])['Cover_Type'].agg({'Wilderness_Area_Mean':'mean', 
                                                                              'Wilderness_Area_Std':'std', 
                                                                              'Wilderness_Area_Size':'size', 
                                                                              'Wilderness_Area_Sum':'sum'}).reset_index()
wildness_mean_encoding
train_df = train_df.merge(wildness_mean_encoding, on='Wilderness_Area', how='left')
test_df = test_df.merge(wildness_mean_encoding, on='Wilderness_Area', how='left')
baseline_tree_cv(train_df)
nn_train_df, nn_test_df = nn_data_preprocessing(train, test)
baseline_keras_cv(nn_train_df)