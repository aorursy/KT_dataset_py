target_name = 'question_not_really_a_question'
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import eli5

from eli5 import show_prediction



# preprocessing

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold

import pandas_profiling as pp



# models

from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV

from sklearn.svm import SVR, LinearSVR

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor 

from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, VotingRegressor 

from sklearn.neural_network import MLPRegressor

from sklearn.tree import DecisionTreeRegressor

import sklearn.model_selection

from sklearn.model_selection import cross_val_predict as cvp

from sklearn import metrics

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import xgboost as xgb

import lightgbm as lgb



# model tuning

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval



import warnings

warnings.filterwarnings("ignore")



pd.set_option('max_columns',100)

pd.set_option('max_rows',100)
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
valid_part = 0.2
train0 = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

test0 = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')
train0['t_len'] = train0['question_title'].str.len()

train0['q_len'] = train0['question_body'].str.len()

train0['a_len'] = train0['answer'].str.len()
test0['t_len'] = test0['question_title'].str.len()

test0['q_len'] = test0['question_body'].str.len()

test0['a_len'] = test0['answer'].str.len()
train0['qa_len'] = train0['q_len'] + train0['a_len']

test0['qa_len'] = test0['q_len'] + test0['a_len']
train0.head(1)
test0.head(1)
pd.set_option('max_colwidth', 600)
train0[train0[target_name] == 0].head(3)
train0[train0[target_name] == 1].head(3)
train0.info()
answ_start = ["Can", "Could", "Did", "Does", "Do", "Has", "How", "Will", "Is", "What", "Where", "Why", "When"]

answ_dict = dict.fromkeys(answ_start, 0)

answ_dict_test = dict.fromkeys(answ_start, 0)
train0['Is_question_in_body'] = False

for i in range(len(train0)):

    if i % 1000 == 0:

        print(i)

    if train0.loc[i, target_name] > 0:

        for s in answ_start:

            if train0.loc[i, 'question_body'].find(s) > 0:

                train0.loc[i,'Is_question_in_body'] = True

                answ_dict[s] += 1

answ_dict
test0['Is_question_in_body'] = False

for i in range(len(test0)):

    for s in answ_start:

        if test0.loc[i, 'question_body'].find(s) > 0:

            test0.loc[i,'Is_question_in_body'] = True

            answ_dict_test[s] += 1

answ_dict_test
train0['Is_question_body_sign'] = train0['question_body'].str.contains("?", regex=False)

train0['Is_question_body_sign'].sum()
test0['Is_question_body_sign'] = test0['question_body'].str.contains("?", regex=False)

test0['Is_question_body_sign'].sum()
train0 = train0[['Is_question_in_body', 'Is_question_body_sign', 'qa_len', 'question_not_really_a_question']]

test0 = test0[['Is_question_in_body', 'Is_question_body_sign', 'qa_len']]
train0.head(1)
test0.head(1)
train0 = reduce_mem_usage(train0)
test0 = reduce_mem_usage(test0)
pp.ProfileReport(train0)
pp.ProfileReport(test0)
train0 = train0.fillna(-1)

test0 = test0.fillna(-1)
seed = 0
# For boosting model

train0b = train0

train_target0b = train0b[target_name]

train0b = train0b.drop([target_name], axis=1)



# Synthesis valid as test for selection models

trainb, testb, targetb, target_testb = train_test_split(train0b, train_target0b, test_size=valid_part, random_state=seed)
# For boosting model

test0b = test0
train_target0 = train0[target_name]

train0 = train0.drop([target_name], axis=1)
#For models from Sklearn

scaler = StandardScaler()

train0 = pd.DataFrame(scaler.fit_transform(train0), columns = train0.columns)

test0 = pd.DataFrame(scaler.fit_transform(test0), columns = test0.columns)
train0.head(3)
len(train0)
# Synthesis valid as test for selection models

train, test, target, target_test = train_test_split(train0, train_target0, test_size=valid_part, random_state=seed)
train.head(3)
test.head(3)
train.info()
test.info()
acc_train_r2 = []

acc_test_r2 = []

acc_train_d = []

acc_test_d = []

acc_train_rmse = []

acc_test_rmse = []
def acc_d(y_meas, y_pred):

    # Relative error between predicted y_pred and measured y_meas values

    return mean_absolute_error(y_meas, y_pred)*len(y_meas)/sum(abs(y_meas))



def acc_rmse(y_meas, y_pred):

    # RMSE between predicted y_pred and measured y_meas values

    return (mean_squared_error(y_meas, y_pred))**0.5
def acc_boosting_model(num,model,train,test,num_iteration=0):

    # Calculation of accuracy of boosting model by different metrics

    

    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse

    

    if num_iteration > 0:

        ytrain = model.predict(train, num_iteration = num_iteration)  

        ytest = model.predict(test, num_iteration = num_iteration)

    else:

        ytrain = model.predict(train)  

        ytest = model.predict(test)



    print('target = ', targetb[:5].values)

    print('ytrain = ', ytrain[:5])



    acc_train_r2_num = round(r2_score(targetb, ytrain) * 100, 2)

    print('acc(r2_score) for train =', acc_train_r2_num)   

    acc_train_r2.insert(num, acc_train_r2_num)



    acc_train_d_num = round(acc_d(targetb, ytrain) * 100, 2)

    print('acc(relative error) for train =', acc_train_d_num)   

    acc_train_d.insert(num, acc_train_d_num)



    acc_train_rmse_num = round(acc_rmse(targetb, ytrain) * 100, 2)

    print('acc(rmse) for train =', acc_train_rmse_num)   

    acc_train_rmse.insert(num, acc_train_rmse_num)



    print('target_test =', target_testb[:5].values)

    print('ytest =', ytest[:5])

    

    acc_test_r2_num = round(r2_score(target_testb, ytest) * 100, 2)

    print('acc(r2_score) for test =', acc_test_r2_num)

    acc_test_r2.insert(num, acc_test_r2_num)

    

    acc_test_d_num = round(acc_d(target_testb, ytest) * 100, 2)

    print('acc(relative error) for test =', acc_test_d_num)

    acc_test_d.insert(num, acc_test_d_num)

    

    acc_test_rmse_num = round(acc_rmse(target_testb, ytest) * 100, 2)

    print('acc(rmse) for test =', acc_test_rmse_num)

    acc_test_rmse.insert(num, acc_test_rmse_num)
def acc_model(num,model,train,test):

    # Calculation of accuracy of model from Sklearn by different metrics   

  

    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse

    

    ytrain = model.predict(train)  

    ytest = model.predict(test)



    print('target = ', target[:5].values)

    print('ytrain = ', ytrain[:5])



    acc_train_r2_num = round(r2_score(target, ytrain) * 100, 2)

    print('acc(r2_score) for train =', acc_train_r2_num)   

    acc_train_r2.insert(num, acc_train_r2_num)



    acc_train_d_num = round(acc_d(target, ytrain) * 100, 2)

    print('acc(relative error) for train =', acc_train_d_num)   

    acc_train_d.insert(num, acc_train_d_num)



    acc_train_rmse_num = round(acc_rmse(target, ytrain) * 100, 2)

    print('acc(rmse) for train =', acc_train_rmse_num)   

    acc_train_rmse.insert(num, acc_train_rmse_num)



    print('target_test =', target_test[:5].values)

    print('ytest =', ytest[:5])

    

    acc_test_r2_num = round(r2_score(target_test, ytest) * 100, 2)

    print('acc(r2_score) for test =', acc_test_r2_num)

    acc_test_r2.insert(num, acc_test_r2_num)

    

    acc_test_d_num = round(acc_d(target_test, ytest) * 100, 2)

    print('acc(relative error) for test =', acc_test_d_num)

    acc_test_d.insert(num, acc_test_d_num)

    

    acc_test_rmse_num = round(acc_rmse(target_test, ytest) * 100, 2)

    print('acc(rmse) for test =', acc_test_rmse_num)

    acc_test_rmse.insert(num, acc_test_rmse_num)
# Linear Regression



linreg = LinearRegression()

linreg.fit(train, target)

acc_model(0,linreg,train,test)
coeff_linreg = pd.DataFrame(train.columns.delete(0))

coeff_linreg.columns = ['feature']

coeff_linreg["score_linreg"] = pd.Series(linreg.coef_)

coeff_linreg.sort_values(by='score_linreg', ascending=False)
# Eli5 visualization

eli5.show_weights(linreg, feature_names = train.columns.tolist())
# Support Vector Machines



svr = SVR()

svr.fit(train, target)

acc_model(1,svr,train,test)
# Linear SVR



linear_svr = LinearSVR()

linear_svr.fit(train, target)

acc_model(2,linear_svr,train,test)
# MLPRegressor



mlp = MLPRegressor()

param_grid = {'hidden_layer_sizes': [i for i in range(2,20)],

              'activation': ['relu'],

              'solver': ['adam'],

              'learning_rate': ['constant'],

              'learning_rate_init': [0.01],

              'power_t': [0.5],

              'alpha': [0.0001],

              'max_iter': [1000],

              'early_stopping': [True],

              'warm_start': [False]}

mlp_GS = GridSearchCV(mlp, param_grid=param_grid, 

                   cv=10, verbose=True, pre_dispatch='2*n_jobs')

mlp_GS.fit(train, target)

acc_model(3,mlp_GS,train,test)
# Stochastic Gradient Descent



sgd = SGDRegressor()

sgd.fit(train, target)

acc_model(4,sgd,train,test)
# Decision Tree Regression



decision_tree = DecisionTreeRegressor()

decision_tree.fit(train, target)

acc_model(5,decision_tree,train,test)
# Random Forest



random_forest = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'n_estimators': [100, 1000]}, cv=10)

random_forest.fit(train, target)

print(random_forest.best_params_)

acc_model(6,random_forest,train,test)
xgb_clf = xgb.XGBRegressor({'objective': 'reg:squarederror'}) 

parameters = {'n_estimators': [200, 300], 

              'learning_rate': [0.01, 0.02, 0.03],

              'max_depth': [10, 12]}

xgb_reg = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=5, n_jobs=-1).fit(trainb, targetb)

print("Best score: %0.3f" % xgb_reg.best_score_)

print("Best parameters set:", xgb_reg.best_params_)

acc_boosting_model(7,xgb_reg,trainb,testb)
#%% split training set to validation set

Xtrain, Xval, Ztrain, Zval = train_test_split(trainb, targetb, test_size=0.2, random_state=0)

train_set = lgb.Dataset(Xtrain, Ztrain, silent=False)

valid_set = lgb.Dataset(Xval, Zval, silent=False)
params = {

        'boosting_type':'gbdt',

        'objective': 'regression',

        'num_leaves': 31,

        'learning_rate': 0.01,

        'max_depth': -1,

        'subsample': 0.8,

        'bagging_fraction' : 1,

        'max_bin' : 5000 ,

        'bagging_freq': 20,

        'colsample_bytree': 0.6,

        'metric': 'rmse',

        'min_split_gain': 0.5,

        'min_child_weight': 1,

        'min_child_samples': 10,

        'scale_pos_weight':1,

        'zero_as_missing': False,

        'seed':0,        

    }

modelL = lgb.train(params, train_set = train_set, num_boost_round=10000,

                   early_stopping_rounds=2000,verbose_eval=500, valid_sets=valid_set)
acc_boosting_model(8,modelL,trainb,testb,modelL.best_iteration)
def hyperopt_gb_score(params):

    clf = GradientBoostingRegressor(**params)

    current_score = cross_val_score(clf, train, target, cv=10).mean()

    print(current_score, params)

    return current_score 

 

space_gb = {

            'n_estimators': hp.choice('n_estimators', range(100, 1000)),

            'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int))            

        }

 

best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=10)

print('best:')

print(best)
params = space_eval(space_gb, best)

params
# Gradient Boosting Regression



gradient_boosting = GradientBoostingRegressor(**params)

gradient_boosting.fit(train, target)

acc_model(9,gradient_boosting,train,test)
# Ridge Regressor



ridge = RidgeCV(cv=10)

ridge.fit(train, target)

acc_model(10,ridge,train,test)
# Bagging Regressor



bagging = BaggingRegressor()

bagging.fit(train, target)

acc_model(11,bagging,train,test)
# Extra Trees Regressor



etr = ExtraTreesRegressor()

etr.fit(train, target)

acc_model(12,etr,train,test)
# AdaBoost Regression



Ada_Boost = AdaBoostRegressor()

Ada_Boost.fit(train, target)

acc_model(13,Ada_Boost,train,test)
Voting_Reg = VotingRegressor(estimators=[('lin', linreg), ('ridge', ridge), ('sgd', sgd)])

Voting_Reg.fit(train, target)

acc_model(14,Voting_Reg,train,test)
models = pd.DataFrame({

    'Model': ['Linear Regression', 'Support Vector Machines', 'Linear SVR', 

              'MLPRegressor', 'Stochastic Gradient Decent', 

              'Decision Tree Regressor', 'Random Forest',  'XGB', 'LGBM',

              'GradientBoostingRegressor', 'RidgeRegressor', 'BaggingRegressor', 'ExtraTreesRegressor', 

              'AdaBoostRegressor', 'VotingRegressor'],

    

    'r2_train': acc_train_r2,

    'r2_test': acc_test_r2,

    'd_train': acc_train_d,

    'd_test': acc_test_d,

    'rmse_train': acc_train_rmse,

    'rmse_test': acc_test_rmse

                     })
pd.options.display.float_format = '{:,.2f}'.format
print('Prediction accuracy for models by R2 criterion - r2_test')

models.sort_values(by=['r2_test', 'r2_train'], ascending=False)
print('Prediction accuracy for models by relative error - d_test')

models.sort_values(by=['d_test', 'd_train'], ascending=True)
print('Prediction accuracy for models by RMSE - rmse_test')

models.sort_values(by=['rmse_test', 'rmse_train'], ascending=True)
# Plot

plt.figure(figsize=[25,6])

xx = models['Model']

plt.tick_params(labelsize=14)

plt.plot(xx, models['r2_train'], label = 'r2_train')

plt.plot(xx, models['r2_test'], label = 'r2_test')

plt.legend()

plt.title('R2-criterion for 15 popular models for train and test datasets')

plt.xlabel('Models')

plt.ylabel('R2-criterion, %')

plt.xticks(xx, rotation='vertical')

plt.savefig('graph.png')

plt.show()
# Plot

plt.figure(figsize=[25,6])

xx = models['Model']

plt.tick_params(labelsize=14)

plt.plot(xx, models['d_train'], label = 'd_train')

plt.plot(xx, models['d_test'], label = 'd_test')

plt.legend()

plt.title('Relative errors for 15 popular models for train and test datasets')

plt.xlabel('Models')

plt.ylabel('Relative error, %')

plt.xticks(xx, rotation='vertical')

plt.savefig('graph.png')

plt.show()
# Plot

plt.figure(figsize=[25,6])

xx = models['Model']

plt.tick_params(labelsize=14)

plt.plot(xx, models['rmse_train'], label = 'rmse_train')

plt.plot(xx, models['rmse_test'], label = 'rmse_test')

plt.legend()

plt.title('RMSE for 15 popular models for train and test datasets')

plt.xlabel('Models')

plt.ylabel('RMSE, %')

plt.xticks(xx, rotation='vertical')

plt.savefig('graph.png')

plt.show()
pred = xgb_reg.predict(test0)