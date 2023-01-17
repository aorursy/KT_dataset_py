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

    # all_data['Aspect'].fillna(all_data['Aspect'].mean(), inplace=True)
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
    
    train_df = all_data.iloc[:train_index]
    test_df = all_data.iloc[train_index:]
    
    del all_data, predict, aspect_train, aspect_test
    gc.collect()
    
    return train_df, test_df
train_df, test_df = tree_data_preprocessing(train, test)
all_data = pd.concat([train_df, test_df])
distance_feature = [col for col in train.columns if col.find('Distance') != -1 ]
distance_feature
all_data[distance_feature].head()
all_data['Horizontal_Distance_To_Hydrology'] = all_data['Horizontal_Distance_To_Hydrology']/1000
sns.pairplot(all_data[distance_feature + ['Cover_Type']], hue='Cover_Type', x_vars=distance_feature, y_vars=distance_feature, size=3)
plt.show()
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
np.isinf(all_data).sum()
all_data.loc[np.isinf(all_data['HF4']),'HF4'] = 0
all_data.loc[np.isinf(all_data['HR4']),'HR4'] = 0
all_data.loc[np.isinf(all_data['HH4']),'HH4'] = 0
all_data.loc[np.isinf(all_data['FR4']),'FR4'] = 0
np.isinf(all_data).sum().sum()
all_data.isnull().sum()
all_data[['HF4','HH4']] = all_data[['HF4','HH4']].fillna(0)
def target_disturibution(frame, col):
    sns.distplot(frame.loc[frame['Cover_Type']==0, col])
    sns.distplot(frame.loc[frame['Cover_Type']==1, col])
    plt.title(col)
    plt.show()
train_df = all_data.iloc[:train_index]
test_df = all_data.iloc[train_index:]
for col in train_df.columns:
    if col.find('HF') != -1 or col.find('HH') != -1 or col.find('FR') != -1 or col.find('HR') != -1:
        target_disturibution(train_df,col)
baseline_tree_cv(train_df)
other_numerical_feature = ['Aspect', 'Elevation', 'Slope', 'Hillshade_3pm', 'Hillshade_9am', 'Hillshade_Noon']
sns.pairplot(all_data[other_numerical_feature + ['Cover_Type']], hue='Cover_Type', x_vars=other_numerical_feature, y_vars=other_numerical_feature, size=3)
plt.show()
sns.scatterplot('Elevation', 'Aspect', hue='Cover_Type',data=all_data)
plt.show()
def scatter_quantile_graph(frame, col1, col2):
    col1_quantile = np.arange(0,1.1,0.1)
    col2_quantile = np.arange(0,1.1,0.2)
    
    sns.scatterplot(col1, col2, hue='Cover_Type',data=frame)
    for quantile_value in frame[col1].quantile(col1_quantile):
        plt.axvline(quantile_value, color='red')
    for quantile_value in frame[col2].quantile(col2_quantile):
        plt.axhline(quantile_value, color='blue')

    plt.title('{} - {}'.format(col1,col2))
    plt.show()
scatter_quantile_graph(all_data, 'Elevation', 'Aspect')
scatter_quantile_graph(all_data, 'Elevation', 'Slope')
scatter_quantile_graph(all_data, 'Elevation', 'Hillshade_3pm')
scatter_quantile_graph(all_data, 'Elevation', 'Hillshade_9am')
scatter_quantile_graph(all_data, 'Elevation', 'Hillshade_Noon')
all_data_binning = all_data.copy()
quantile_10 = np.arange(0,1.1,0.1)
quantile_5 = np.arange(0,1.1,0.2)
    
all_data_binning['Elevation_quantile_label'] = pd.qcut(
                                            all_data_binning['Elevation'], 
                                            q=quantile_10, labels = ['Ele_quantile_{:.1f}'.format(col) for col in quantile_10][1:])

all_data_binning['Aspect_quantile_label'] = pd.qcut(
                                            all_data_binning['Aspect'], 
                                            q=quantile_5, labels = ['Aspect_quantile_{:.1f}'.format(col) for col in quantile_5][1:])
all_data_binning['Ele_Asp_Combine'] = all_data_binning[['Elevation_quantile_label','Aspect_quantile_label']].apply(lambda row: row['Elevation_quantile_label'] +'_'+ row['Aspect_quantile_label'] ,axis=1)
all_data_binning['Ele_Asp_Combine'].nunique()
for col in ['Elevation_quantile_label','Aspect_quantile_label','Ele_Asp_Combine']:
    all_data_binning[col] = all_data_binning[col].factorize()[0]
train_df_binning = all_data_binning.iloc[:train_index]
test_df_binning = all_data_binning.iloc[train_index:]
baseline_tree_cv(train_df_binning)
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
all_data, new_col = binning_category_combine_feature(all_data, 'Elevation', 'Aspect', 0.1, 0.1)

""" 나머지 부분은 Feature 추가하면서 성능 검토 바랍니다~!
all_data, new_col = binning_category_combine_feature(all_data, 'Elevation', 'Slope', 0.1, 0.2)
for col in new_col:
    all_data = frequency_encoding(all_data, col)

all_data, new_col = binning_category_combine_feature(all_data, 'Elevation', 'Hillshade_3pm', 0.1, 0.2)
for col in new_col:
    all_data = frequency_encoding(all_data, col)

all_data, new_col = binning_category_combine_feature(all_data, 'Elevation', 'Hillshade_9am', 0.1, 0.2)
for col in new_col:
    all_data = frequency_encoding(all_data, col)

all_data, new_col = binning_category_combine_feature(all_data, 'Elevation', 'Hillshade_Noon', 0.1, 0.2)
for col in new_col:
    all_data = frequency_encoding(all_data, col)
"""
all_data.head()
train_df = all_data.iloc[:train_index]
test_df = all_data.iloc[train_index:]
baseline_tree_cv(train_df)
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
    
    before_one_hot = set(all_data.columns)
    for col in category_feature:
        all_data = pd.concat([all_data,pd.get_dummies(all_data[col],prefix=col)],axis=1)
        del all_data[col]
    one_hot_feature = set(all_data.columns) - before_one_hot
    
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
    
    scale_feature = list(set(all_data.columns)-one_hot_feature-set(['Cover_Type','ID']))
    
    train_df = all_data.iloc[:train_index]
    test_df = all_data.iloc[train_index:]

    sc = StandardScaler()
    train_df[scale_feature] = sc.fit_transform(train_df[scale_feature])
    test_df[scale_feature] = sc.transform(test_df[scale_feature] )
    
    return train_df, test_df
nn_train_df, nn_test_df = nn_data_preprocessing(train, test)
baseline_keras_cv(nn_train_df)