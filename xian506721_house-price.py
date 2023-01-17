# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
path = '../input/'

train_data = pd.read_csv(path+'train.csv',index_col=0)

test_data = pd.read_csv(path+'test.csv',index_col=0)

test_sub = pd.read_csv(path+'sample_submission.csv',index_col=0)



train_y = train_data['SalePrice']

train_data.drop(['SalePrice'],axis=1,inplace=True)



data = train_data.append(test_data,sort=False)

features = data.columns

sns.set_style('whitegrid')
#1.特征缺失比率

def Na_rate(data):

    #特征缺失比率

    na_df = pd.DataFrame(columns=['feature_name','na_rate'])

    na_rate = []

    for i in data.columns:

        rate = data[i].isnull().sum()*1.0/len(data[i])

        na_rate.append(rate)

        

    na_df['feature_name'] = features

    na_df['na_rate'] = na_rate

    na_df = na_df[na_df['na_rate']!=0].sort_values(by='na_rate',ascending=False)

    print('miss_features_num',na_df.shape[0])

    return na_df





#2.区分类别和数值型变量

def Dis_features(data):

    #区分类别和数值型变量

    cat_features = []

    num_features = []

    for i in data.columns:

        if data[i].dtype == 'object':

            cat_features.append(i)

        else:

            num_features.append(i)

    if 'Id' in num_features:

        num_features.remove('Id')

    print('类别型变量共有：',len(cat_features))

    print('数值型变量有：',len(num_features))

    

    return cat_features,num_features



#3.对有原因缺失分-1和1填充

def Fill_reason_missing(data,features):

    #给数据集、需要处理的特征

    data1 = data.copy()

    for i in features:

        data1[i] = data1[i].fillna(-1)

        data1[i] = data1[i].map(lambda x:x if x==-1 else 1)

    return data1



#4.对数值型变量缺失值进行简单填充

def Fill_num_miss(data,col):

    data1 = data.copy()

    for i in col:

        data1[i] = SimpleImputer().fit_transform(data1[i].values.reshape(-1,1))

    return data1



#5.对类别型变量进行众数填充

def Fill_cat_miss(data,col):

    data1 = data.copy()

    for i in col:

        data1[i] = data1[i].fillna(data1[i].dropna().mode()[0])

    return data1



#6.对类别型变量进行编码

def Cat_encoder(data,col):

    data1 = data.copy()

    for i in col:

        data1[i] = pd.factorize(data1[i])[0]

    return data1



#7.对数值型变量进行归一化

def Standard_num(data,col):

    data1 = data.copy()

    for i in col:

        data1[i] = StandardScaler().fit_transform(data1[i].values.reshape(-1,1))

    return data1
#回归问题

print('偏度：',train_y.skew())

print('峰度：',train_y.kurt())

train_y.hist(bins=50)

plt.show()

#目标变量大致是一个右偏的正态分布
#由于目标变量右偏，因此选择λ=0的box-cox变换，可以减小不可观测误差和变量的相关性，降低数据分布的偏度，使得数据更接近正态分布

log_train_y = np.log(1+train_y)

print('偏度：',log_train_y.skew())

print('峰度：',log_train_y.kurt())

log_train_y.hist(bins=50)

plt.show()
cat_all, num_all = Dis_features(data)
na_train = Na_rate(train_data)

print(na_train)

#合并总数据的缺失情况

na_all = Na_rate(data)

print(na_all)

#对于超过40%缺失的这几个变量进行观察，发现都是房子的某种设施是什么类型的，缺失是没有该设施，是有意义的缺失，因此决定将这几个特征按照是否缺失分成两类来处理

miss_40_features = na_all[na_all['na_rate']>0.4]['feature_name'].values
#对缺失超过4成的变量进行是否缺失的（-1,1）的编码

data = Fill_reason_missing(data,miss_40_features)

#此时的Na_rate

na_all1 = Na_rate(data)

print(na_all1)
#剩下的缺失数值型变量

print('训练集：')

missing_data_train = train_data[na_train['feature_name'].values]

cat_miss_train, num_miss_train = Dis_features(missing_data_train)

print(num_miss_train)



print('全部数据集：')

missing_data = data[na_all1['feature_name'].values]

cat_miss, num_miss = Dis_features(missing_data)#得到数值型缺失变量

print(num_miss)
#对于上面训练集和所有数据集共同缺失的数据，三个数值型缺失变量，

#1.LotFrontage表示房产距离街边的距离，此变量用简单填充

data['LotFrontage'] = SimpleImputer().fit_transform(data['LotFrontage'].values.reshape(-1,1))

#2.GarageYrBlt车库修建年数，MasVnrArea贴砖面积，缺失是因为没有车库，没有贴砖，以及其他变量因为没有车库和地下室的缺失造成，因此单独编码为-1

for i in num_miss:

    data[i] = data[i].fillna(-1)

    

#此时的Na_rate

na_all2 = Na_rate(data)

print(na_all2)

print('全部数据集：')

missing_data = data[na_all2['feature_name'].values]

Dis_features(missing_data)#得到缺失变量

#此时就剩这18个类别型变量，车库和地下室相关的变量缺失都是有原因，MasVnrType表面砌体这个变量，没有对缺失解释，因此用众数填充,后面的变量都是这种情况

#需要补缺的类别型变量有：['MasVnrType','MSZoning','Functional','Utilities','KitchenQual','Electrical','Exterior2nd','Exterior1st','SaleType']

fill_cat = na_all2[na_all2['na_rate']<0.02]['feature_name'].values

print(data['MasVnrType'].value_counts())

print('==========')

print(fill_cat)

data = Fill_cat_miss(data,fill_cat)#类别型变量补缺

na_all3 = Na_rate(data)

print('==========')

print(na_all3)
#对于剩下的变量，因为是有原因缺失，可以直接进行pd.factorize编码，缺失会自动填为-1



print('cat：',cat_all)

print('='*50)

print('num:',num_all)

data = Cat_encoder(data,cat_all)#对所有类别型变量进行编码
#查看现在得缺失情况,发现没有缺失了

Na_rate(data)
data = Standard_num(data,num_all)
#在考虑要不要把取值过多的类别型变量进行分箱

for i in cat_all:

    length = len(set(data[i].values))

    print(i,length)
train_x = data.loc[train_data.index]

test_x = data.loc[test_data.index]
from sklearn import model_selection

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor



def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):



    # random forest

    rf_est = RandomForestRegressor(random_state=0)

    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}

    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)

    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)

    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))

    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))

    print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))

    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),

                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)

    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']

    print('Sample 10 Features from RF')

    print(str(features_top_n_rf[:10]))



    # AdaBoost

    ada_est =AdaBoostRegressor(random_state=0)

    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}

    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)

    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)

    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))

    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))

    print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))

    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),

                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)

    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']

    print('Sample 10 Feature from Ada:')

    print(str(features_top_n_ada[:10]))



    # ExtraTree

    et_est = ExtraTreesRegressor(random_state=0)

    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}

    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)

    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)

    print('Top N Features Best ET Params:' + str(et_grid.best_params_))

    print('Top N Features Best ET Score:' + str(et_grid.best_score_))

    print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))

    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),

                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)

    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']

    print('Sample 10 Features from ET:')

    print(str(features_top_n_et[:10]))

    

    # GradientBoosting

    gb_est =GradientBoostingRegressor(random_state=0)

    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}

    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)

    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)

    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))

    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))

    print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))

    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),

                                           'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)

    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']

    print('Sample 10 Feature from GB:')

    print(str(features_top_n_gb[:10]))

    

    # DecisionTree

    dt_est = DecisionTreeRegressor(random_state=0)

    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}

    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)

    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)

    print('Top N Features Best DT Params:' + str(dt_grid.best_params_))

    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))

    print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))

    feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),

                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)

    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']

    print('Sample 10 Features from DT:')

    print(str(features_top_n_dt[:10]))

    

    # merge the three models

    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt], 

                               ignore_index=True).drop_duplicates()

    

    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et, 

                                   feature_imp_sorted_gb, feature_imp_sorted_dt],ignore_index=True)

    

    return features_top_n , features_importance

feature_to_pick = 30

feature_top_n, feature_importance = get_top_n_features(train_x, log_train_y, feature_to_pick)

train_x = pd.DataFrame(train_x[feature_top_n])

test_x = pd.DataFrame(test_x[feature_top_n])



# rf.fit(train_x,log_train_y)

# log_y_hat = rf.predict(test_x)

# # rf.fit(train_x,train_y)

# # y_hat = rf.predict(test_x)

rf_feature_imp = feature_importance[:10]

Ada_feature_imp = feature_importance[32:32+10].reset_index(drop=True)



# make importances relative to max importance

rf_feature_importance = 100.0 * (rf_feature_imp['importance'] / rf_feature_imp['importance'].max())

Ada_feature_importance = 100.0 * (Ada_feature_imp['importance'] / Ada_feature_imp['importance'].max())



# Get the indexes of all features over the importance threshold

rf_important_idx = np.where(rf_feature_importance)[0]

Ada_important_idx = np.where(Ada_feature_importance)[0]



# Adapted from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html

pos = np.arange(rf_important_idx.shape[0]) + .5



plt.figure(1, figsize = (18, 8))



plt.subplot(121)

plt.barh(pos, rf_feature_importance[rf_important_idx][::-1])

plt.yticks(pos, rf_feature_imp['feature'][::-1])

plt.xlabel('Relative Importance')

plt.title('RandomForest Feature Importance')



plt.subplot(122)

plt.barh(pos, Ada_feature_importance[Ada_important_idx][::-1])

plt.yticks(pos, Ada_feature_imp['feature'][::-1])

plt.xlabel('Relative Importance')

plt.title('AdaBoost Feature Importance')



plt.show()
from sklearn.model_selection import KFold



# Some useful parameters which will come in handy later on

ntrain = train_x.shape[0]

ntest = test_x.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 7 # set folds for out-of-fold prediction

kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)



def get_out_fold(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.fit(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


from sklearn.svm import SVR



rf = RandomForestRegressor(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=6, 

                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)



ada = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)



et = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)



gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)



dt = DecisionTreeRegressor(max_depth=8)



svm = SVR(kernel='linear', C=0.025)
x_train = train_x.values # Creates an array of the train data

x_test = test_x.values # Creats an array of the test data

y_train = log_train_y.values
rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test) # AdaBoost 

et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test) # Extra Trees

gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test) # Gradient Boost

dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test) # Decision Tree

svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test) # Support Vector



print("Training is complete")
x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, svm_oof_train), axis=1)

x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, svm_oof_test), axis=1)
from xgboost import XGBRegressor



gbm = XGBRegressor( n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, 

                        colsample_bytree=0.8, nthread= -1, scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)

y_hat = np.expm1(predictions)

submission = test_sub.copy()

submission['SalePrice'] = y_hat

submission.to_csv('submission.csv')
submission
# y_hat = np.expm1(log_y_hat)
# test_sub['SalePrice']

# submission = test_sub.copy()

# submission['SalePrice'] = y_hat

# submission.to_csv('submission.csv')
