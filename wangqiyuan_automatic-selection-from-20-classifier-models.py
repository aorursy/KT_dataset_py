import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



# preprocessing

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold

import pandas_profiling as pp



# models

from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

import xgboost as xgb

from xgboost import XGBClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_val_predict as cvp

from sklearn import metrics

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score



# model tuning

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval



import warnings

warnings.filterwarnings("ignore")
path = "../input/titanic/"
test_train_split_part = 0.2
metrics_all = {1 : 'r2_score', 2 : 'relative_error', 3 : 'rmse'}

metrics_now = [1, 2, 3] # you can only select some numbers of metrics from metrics_all
traindf = pd.read_csv(path + 'train.csv').set_index('PassengerId')

testdf = pd.read_csv(path + 'test.csv').set_index('PassengerId')

submission = pd.read_csv(path + 'gender_submission.csv')
traindf.head(10)
target_name = 'Survived'
traindf.info()
testdf.info()
#Thanks to:

# https://www.kaggle.com/mauricef/titanic

# https://www.kaggle.com/vbmokin/titanic-top-3-one-line-of-the-prediction-code

df = pd.concat([traindf, testdf], axis=0, sort=False)

df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))

df['LastName'] = df.Name.str.split(',').str[0]

family = df.groupby(df.LastName).Survived

df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())

df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)

df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())

df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount - \

                                    df.Survived.fillna(0), axis=0)

df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)

df.WomanOrBoyCount = df.WomanOrBoyCount.replace(np.nan, 0)

df['Alone'] = (df.WomanOrBoyCount == 0)

df = pd.concat([df.WomanOrBoySurvived.fillna(0), df.Alone.fillna(0), 

                df.Sex.replace({'male': 0, 'female': 1}), df.Survived], axis=1)
target0 = df[target_name].loc[traindf.index]

df = df.drop([target_name], axis=1)

train0, test0 = df.loc[traindf.index], df.loc[testdf.index]
train0.head(3)
pp.ProfileReport(train0)
pp.ProfileReport(test0)
# Determination categorical features

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

categorical_columns = []

features = train0.columns.values.tolist()

for col in features:

    if train0[col].dtype in numerics: continue

    categorical_columns.append(col)

categorical_columns
# Encoding categorical features

for col in categorical_columns:

    if col in train0.columns:

        le = LabelEncoder()

        le.fit(list(train0[col].astype(str).values) + list(test0[col].astype(str).values))

        train0[col] = le.transform(list(train0[col].astype(str).values))

        test0[col] = le.transform(list(test0[col].astype(str).values))
train0.info()
# For boosting model

train0b = train0.copy()

test0b = test0.copy()

# Synthesis valid as "test" for selection models

trainb, testb, targetb, target_testb = train_test_split(train0b, target0, test_size=test_train_split_part, random_state=0)
#For models from Sklearn

scaler = StandardScaler()

train0 = pd.DataFrame(scaler.fit_transform(train0), columns = train0.columns)

test0 = pd.DataFrame(scaler.transform(test0), columns = test0.columns)
train0.head(3)
# Synthesis valid as test for selection models

train, test, target, target_test = train_test_split(train0, target0, test_size=test_train_split_part, random_state=0)
train.head(3)
test.head(3)
train.info()
test.info()
# list of accuracy of all model - amount of metrics_now * 2 (train & test datasets)

num_models = 20

acc_train = []

acc_test = []

acc_all = np.empty((len(metrics_now)*2, 0)).tolist()

acc_all
N_best_models = 5 # the amount of the best models which to need selection

acc_all_pred = np.empty((len(metrics_now), 0)).tolist()

acc_all_pred
def acc_d(y_meas, y_pred):

    # Relative error between predicted y_pred and measured y_meas values

    return mean_absolute_error(y_meas, y_pred)*len(y_meas)/sum(abs(y_meas))



def acc_rmse(y_meas, y_pred):

    # RMSE between predicted y_pred and measured y_meas values

    return (mean_squared_error(y_meas, y_pred))**0.5
def acc_metrics_calc(num,model,train,test,target,target_test):

    # The models selection stage

    # Calculation of accuracy of model by different metrics

    global acc_all



    ytrain = model.predict(train).astype(int)

    ytest = model.predict(test).astype(int)

    print('target = ', target[:5].values)

    print('ytrain = ', ytrain[:5])

    print('target_test =', target_test[:5].values)

    print('ytest =', ytest[:5])

    

    num_acc = 0

    for x in metrics_now:

        if x == 1:

            #r2_score criterion

            acc_train = round(r2_score(target, ytrain) * 100, 2)

            acc_test = round(r2_score(target_test, ytest) * 100, 2)

        elif x == 2:

            #relative error criterion

            acc_train = round(acc_d(target, ytrain) * 100, 2)

            acc_test = round(acc_d(target_test, ytest) * 100, 2)

        elif x == 3:

            #rmse criterion

            acc_train = round(acc_rmse(target, ytrain) * 100, 2)

            acc_test = round(acc_rmse(target_test, ytest) * 100, 2)

        

        print('acc of', metrics_all[x], 'for train =', acc_train)

        print('acc of', metrics_all[x], 'for test =', acc_test)

        acc_all[num_acc].append(acc_train) #train

        acc_all[num_acc+1].append(acc_test) #test

        num_acc += 2      
def acc_metrics_calc_pred(num,model,train,test,target):

    # The prediction stage

    # Calculation of accuracy of model for all different metrics and creates of submission file

    global acc_all_pred



    ytrain = model.predict(train).astype(int)

    ytest = model.predict(test).astype(int)

    print('target = ', target[:5].values)

    print('ytrain = ', ytrain[:5])

    print('ytest =', ytest[:5])

    

    num_acc = 0

    for x in metrics_now:

        if x == 1:

            #r2_score criterion

            acc_train = round(r2_score(target, ytrain) * 100, 2)

        elif x == 2:

            #relative error criterion

            acc_train = round(acc_d(target, ytrain) * 100, 2)

        elif x == 3:

            #rmse criterion

            acc_train = round(acc_rmse(target, ytrain) * 100, 2)



        print('acc of', metrics_all[x], 'for train =', acc_train)

        acc_all_pred[num_acc].append(acc_train) #train

        num_acc += 1          

    

    submission[target_name] = ytest

    submission.to_csv('submission'+str(num)+'.csv', index=False)
# Linear Regression



linreg = LinearRegression()

linreg.fit(train, target)

acc_metrics_calc(0,linreg,train,test,target,target_test)
# Support Vector Machines



svr = SVC()

svr.fit(train, target)

acc_metrics_calc(1,svr,train,test,target,target_test)
# Linear SVR



linear_svc = LinearSVC()

linear_svc.fit(train, target)

acc_metrics_calc(2,linear_svc,train,test,target,target_test)
# MLPClassifier



mlp = MLPClassifier()

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

acc_metrics_calc(3,mlp_GS,train,test,target,target_test)
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(train, target)

acc_metrics_calc(4,sgd,train,test,target,target_test)
# Decision Tree Classifier



decision_tree = DecisionTreeClassifier()

decision_tree.fit(train, target)

acc_metrics_calc(5,decision_tree,train,test,target,target_test)
# Random Forest



random_forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': range(100, 1000)}, cv=5)

random_forest.fit(train, target)

print(random_forest.best_params_)

acc_metrics_calc(6,random_forest,train,test,target,target_test)
xgb_clf = xgb.XGBClassifier({'objective': 'reg:squarederror'}) 

parameters = {'n_estimators': [50, 60, 70, 80, 90, 95, 100, 200], 

              'learning_rate': [0.001, 0.0025, 0.005, 0.01, 0.05],

              'max_depth': [3, 4, 5, 6, 7],

              'reg_lambda': [0.05, 0.1, 0.3, 0.5]}

xgb_reg = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=5, n_jobs=-1).fit(trainb, targetb)

print("Best score: %0.3f" % xgb_reg.best_score_)

print("Best parameters set:", xgb_reg.best_params_)

acc_metrics_calc(7,xgb_reg,train,test,target,target_test)
#%% split training set to validation set

Xtrain, Xval, Ztrain, Zval = train_test_split(trainb, targetb, test_size=0.2, random_state=0)

modelL = lgb.LGBMClassifier(n_estimators=1000)

modelL.fit(Xtrain, Ztrain, eval_set=[(Xval, Zval)], early_stopping_rounds=50, verbose=False)
acc_metrics_calc(8,modelL,trainb,testb,targetb,target_testb)
fig =  plt.figure(figsize = (5,5))

axes = fig.add_subplot(111)

lgb.plot_importance(modelL,ax = axes,height = 0.5)

plt.show();

plt.close()
def hyperopt_gb_score(params):

    clf = GradientBoostingClassifier(**params)

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
# Gradient Boosting Classifier



gradient_boosting = GradientBoostingClassifier(**params)

gradient_boosting.fit(train, target)

acc_metrics_calc(9,gradient_boosting,train,test,target,target_test)
# Ridge Classifier



ridge = RidgeClassifier()

ridge.fit(train, target)

acc_ridge_classifier = round(ridge.score(train, target) * 100, 2)

acc_ridge_classifier
acc_test_ridge_classifier = round(ridge.score(test, target_test) * 100, 2)

acc_test_ridge_classifier
acc_metrics_calc(10,ridge,train,test,target,target_test)
# Bagging Classifier



bagging = BaggingClassifier()

bagging.fit(train, target)

acc_metrics_calc(11,bagging,train,test,target,target_test)
# Extra Trees Classifier



etr = ExtraTreesClassifier()

etr.fit(train, target)

acc_metrics_calc(12,etr,train,test,target,target_test)
# AdaBoost Classifier



Ada_Boost = AdaBoostClassifier()

Ada_Boost.fit(train, target)

acc_metrics_calc(13,Ada_Boost,train,test,target,target_test)
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(train, target)

acc_metrics_calc(14,logreg,train,test,target,target_test)
# k-Nearest Neighbors algorithm



knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_neighbors': range(2, 5)}, cv=5).fit(train, target)

print(knn.best_params_)

acc_metrics_calc(15,knn,train,test,target,target_test)
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(train, target)

acc_metrics_calc(16,gaussian,train,test,target,target_test)
# Perceptron



perceptron = Perceptron()

perceptron.fit(train, target)

acc_metrics_calc(17,perceptron,train,test,target,target_test)
# Gaussian Process Classification



gpc = GaussianProcessClassifier()

gpc.fit(train, target)

acc_metrics_calc(18,gpc,train,test,target,target_test)
Voting_ens = VotingRegressor(estimators=[('lin', linreg), ('mlp', mlp_GS), ('sgd', sgd)])

Voting_ens.fit(train, target)

acc_metrics_calc(19,Voting_ens,train,test,target,target_test)
models = pd.DataFrame({

    'Model': ['Linear Regression', 'Support Vector Machines', 'Linear SVC', 

              'MLPClassifier', 'Stochastic Gradient Decent', 

              'Decision Tree Classifier', 'Random Forest',  'XGB', 'LGBMClassifier',

              'GradientBoostingClassifier', 'RidgeClassifier', 'BaggingClassifier', 'ExtraTreesClassifier', 

              'AdaBoostClassifier', 'Logistic Regression',

              'KNN', 'Naive Bayes', 'Perceptron', 'Gaussian Process Classification',

              'VotingRegressor']})
for x in metrics_now:

    xs = metrics_all[x]

    models[xs + '_train'] = acc_all[(x-1)*2]

    models[xs + '_test'] = acc_all[(x-1)*2+1]

models
print('Prediction accuracy for models')

ms = metrics_all[metrics_now[0]] # the first from metrics

models.sort_values(by=[(ms + '_test'), (ms + '_train')], ascending=False)
pd.options.display.float_format = '{:,.2f}'.format
for x in metrics_now:   

    # Plot

    xs = metrics_all[x]

    xs_train = metrics_all[x] + '_train'

    xs_test = metrics_all[x] + '_test'

    plt.figure(figsize=[25,6])

    xx = models['Model']

    plt.tick_params(labelsize=14)

    plt.plot(xx, models[xs_train], label = xs_train)

    plt.plot(xx, models[xs_test], label = xs_test)

    plt.legend()

    plt.title(str(xs) + ' criterion for ' + str(num_models) + ' popular models for train and test datasets')

    plt.xlabel('Models')

    plt.ylabel(xs + ', %')

    plt.xticks(xx, rotation='vertical')

    plt.show()
# Choose the number of metric by which the N_best_models will be determined =>  {1 : 'r2_score', 2 : 'relative_error', 3 : 'rmse'}

metrics_main = 1 

xs = metrics_all[metrics_main]

xs_train = metrics_all[metrics_main] + '_train'

xs_test = metrics_all[metrics_main] + '_test'

print('The best',N_best_models, 'models by the',xs,'criterion:')

direct_sort = False if (metrics_main == 1) else True

models_sort = models.sort_values(by=[xs_test, xs_train], ascending=direct_sort)
models_best = models_sort.iloc[range(N_best_models),:]

models_best
models_pred = pd.DataFrame(models_best.Model, columns = ['Model']) 
def model_fit(name_model,train,target):

    # Fitting name_model (from 20 options) for giver train and target

    # You can optionally add hyperparameters optimization in any model

    if name_model == 'Linear Regression':

        model = LinearRegression()

        model.fit(train, target)

        

    elif name_model == 'Support Vector Machines':

        model = SVC()

        model.fit(train, target)

        

    elif name_model == 'Linear SVC':

        model = LinearSVC()

        model.fit(train, target)

        

    elif name_model == 'MLPClassifier':

        mlp = MLPClassifier()

        param_mlp = {'hidden_layer_sizes': [i for i in range(2,20)],

                      'activation': ['relu'],

                      'solver': ['adam'],

                      'learning_rate': ['constant'],

                      'learning_rate_init': [0.01],

                      'power_t': [0.5],

                      'alpha': [0.0001],

                      'max_iter': [1000],

                      'early_stopping': [True],

                      'warm_start': [False]}

        model = GridSearchCV(mlp, param_grid=param_mlp, 

                           cv=10, verbose=True, pre_dispatch='2*n_jobs')

        model.fit(train, target)

        

    elif name_model == 'Stochastic Gradient Decent':

        model = SGDClassifier()

        model.fit(train, target)

        

    elif name_model == 'Decision Tree Classifier':

        model = DecisionTreeClassifier()

        model.fit(train, target)

        

    elif name_model == 'Random Forest':

        model = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [100, 1000]}, cv=5)

        model.fit(train, target)

        

    elif name_model == 'XGB':

        xgb_clf = xgb.XGBClassifier({'objective': 'reg:squarederror'}) 

        parameters_xgb = {'n_estimators': [60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 140], 

              'learning_rate': [0.005, 0.01, 0.05, 0.075, 0.1],

              'max_depth': [3, 5, 7, 9],

              'reg_lambda': [0.1, 0.3, 0.5]}

        model = GridSearchCV(estimator=xgb_clf, param_grid=parameters_xgb, cv=5, n_jobs=-1).fit(trainb, targetb)

        model.fit(train, target)

        

    elif name_model == 'LGBMClassifier':

        Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=0.2, random_state=0)

        model = lgb.LGBMClassifier(n_estimators=500)

        model.fit(Xtrain, Ztrain, eval_set=[(Xval, Zval)], early_stopping_rounds=50, verbose=False)

        

    elif name_model == 'GradientBoostingClassifier':

        def hyperopt_gb_score(params):

            clf = GradientBoostingClassifier(**params)

            current_score = cross_val_score(clf, train, target, cv=10).mean()

            print(current_score, params)

            return current_score 

        space_gb = {'n_estimators': hp.choice('n_estimators', range(100, 1000)),

                    'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int))}

        best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=10)

        params_gb = space_eval(space_gb, best)

        model = GradientBoostingClassifier(**params_gb)

        model.fit(train, target)

        

    elif name_model == 'RidgeClassifier':

        model = RidgeClassifier()

        model.fit(train, target)

        

    elif name_model == 'BaggingClassifier':

        model = BaggingClassifier()

        model.fit(train, target)

        

    elif name_model == 'ExtraTreesClassifier':

        model = ExtraTreesClassifier()

        model.fit(train, target)

        

    elif name_model == 'AdaBoostClassifier':

        model = AdaBoostClassifier()

        model.fit(train, target)

        

    elif name_model == 'Logistic Regression':

        model = LogisticRegression()

        model.fit(train, target)

        

    elif name_model == 'KNN':

        model = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_neighbors': [2, 3]}, cv=10).fit(train, target)

        model.fit(train, target)

        

    elif name_model == 'Naive Bayes':

        model = GaussianNB()

        model.fit(train, target)

        

    elif name_model == 'Perceptron':

        model = Perceptron()

        model.fit(train, target)

        

    elif name_model == 'Gaussian Process Classification':

        model = GaussianProcessClassifier()

        model.fit(train, target)

        

    elif name_model == 'VotingRegressor':

        model = VotingRegressor(estimators=[('lin', linreg), ('mlp', mlp_GS), ('sgd', sgd)])

        model.fit(train, target)

        

    return model
for i in range(N_best_models):

    name_model = models_best.iloc[i]['Model']

    if (name_model == 'XGB') | (name_model == 'LGBMClassifier'):

        # boosting model

        model = model_fit(name_model,train0b,target0)

        acc_metrics_calc_pred(i,model,train0b,test0b,target0)

    else:

        # model from Sklearn

        model = model_fit(name_model,train0,target0)

        acc_metrics_calc_pred(i,model,train0,test0,target0)
for x in metrics_now:

    xs = metrics_all[x]

    models_pred[xs + '_train'] = acc_all_pred[(x-1)]

models_pred