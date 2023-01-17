import pandas as pd
import sklearn as sk
import numpy as np

print(pd.__version__)
print(sk.__version__)

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
soil_type_cols = []
wilderness_area_cols = []
for col in train_data.columns:
    if 'Soil_Type' in col:
        soil_type_cols.append(col)
    elif 'Wilderness_Area' in col:
        wilderness_area_cols.append(col)
print(soil_type_cols)
print(wilderness_area_cols)
train_data_cate = train_data.loc[:, soil_type_cols + wilderness_area_cols]
test_data_cate = test_data.loc[:, soil_type_cols + wilderness_area_cols]
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

train_data_num = train_data.drop(['Id', 'Cover_Type'] + soil_type_cols + wilderness_area_cols, axis=1)
test_data_num = test_data.drop(['Id'] + soil_type_cols + wilderness_area_cols, axis=1)
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attribute_names]

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])
train_data_num.describe()
train_data_num_tr = pd.DataFrame(num_pipeline.fit_transform(train_data_num))
train_data_num_tr.describe()
test_data_num_tr = pd.DataFrame(num_pipeline.fit_transform(test_data_num))
test_data_num_tr.describe()
train_data_tr = train_data_num_tr.join(train_data_cate)
train_data_tr.describe()
X_data = train_data_tr.values
y_data = train_data['Cover_Type'].values

test_data_tr = test_data_num_tr.join(test_data_cate)
test_data_tr.describe()
from xgboost import XGBClassifier
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, gaussian_process, discriminant_analysis

MLA = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # Gaussian Processes
#     gaussian_process.GaussianProcessClassifier(),  Failed if features are over 10 or more

    # GLM
#     linear_model.PassiveAggressiveClassifier(),
#     linear_model.RidgeClassifierCV(),
#     linear_model.SGDClassifier(),
#     linear_model.Perceptron(),

    # Nearest Neighbor
#     neighbors.KNeighborsClassifier(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),


    # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()
]
from sklearn import model_selection
import seaborn as sns
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = train_data['Cover_Type']

# a = ensemble.ExtraTreesClassifier()
# a.fit(X_train, y_train)
# print(a.score(X_test, y_test))
row_index = 0
for alg in MLA:
#     fit = alg.fit(X_train, y_train)
#     predicted = fit.predict(X_test)
    # fp, tp, th = roc_curve(y_test, predicted)
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    print(MLA_name)
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, X_data, y_data, cv=cv_split, return_train_score=True)
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    print(MLA_compare.loc[row_index, 'MLA Time'])
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    print("MLA Train Accuracy Mean", MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'])
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    print("MLA Test Accuracy Mean", MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'])
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std() * 3  
    # let's know the worst that can happen!

    # save MLA predictions - see section 6 for usage
    alg.fit(X_data, y_data)
    MLA_predict[MLA_name] = alg.predict(X_data)

    row_index += 1
pd.set_option('display.max_columns', None)
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')
# This is an example how to use random search on GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

#Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_n_estimator = [50, 100]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]
print(X_data.shape[1])
print(y_data.shape[0])
print(len(test_data_tr.columns))
vote_test = [
    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
    ('etc',ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),
    
#     xgboost: http://xgboost.readthedocs.io/en/latest/model.html
   ('xgb', XGBClassifier())

]

vote_param = [{
    #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
    'etc__n_estimators': grid_n_estimator,
    'etc__criterion': grid_criterion,
    'etc__max_depth': grid_max_depth,
    'etc__random_state': grid_seed,


    #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
    'gbc__loss': ['deviance', 'exponential'],
    'gbc__learning_rate': grid_ratio,
    'gbc__n_estimators': grid_n_estimator,
    'gbc__criterion': ['friedman_mse', 'mse', 'mae'],
    'gbc__max_depth': grid_max_depth,
    'gbc__min_samples_split': grid_min_samples,
    'gbc__min_samples_leaf': grid_min_samples,      
    'gbc__random_state': grid_seed,

    #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    'rfc__n_estimators': grid_n_estimator,
    'rfc__criterion': grid_criterion,
    'rfc__max_depth': grid_max_depth,
    'rfc__min_samples_split': grid_min_samples,
    'rfc__min_samples_leaf': grid_min_samples,   
    'rfc__bootstrap': grid_bool,
    'rfc__oob_score': grid_bool, 
    'rfc__random_state': grid_seed,

    #http://xgboost.readthedocs.io/en/latest/parameter.html
    'xgb__learning_rate': grid_ratio,
    'xgb__max_depth': [2,4,6,8,10],
    'xgb__tree_method': ['exact', 'approx', 'hist'],
    'xgb__objective': ['reg:linear', 'reg:logistic', 'binary:logistic'],
    'xgb__seed': grid_seed    

}]

grid_param = [
    [{
        #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
        'n_estimators': grid_n_estimator, #default=10
        'criterion': grid_criterion, #default=”gini”
        'max_depth': grid_max_depth, #default=None
        'random_state': grid_seed
     }],
    [{
        #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
#         'loss': ['deviance', 'exponential'], #default=’deviance’
        'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
        'n_estimators': [100], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
#         'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
        'max_depth': grid_max_depth, #default=3   
        'random_state': grid_seed
    }],
    [{
        #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
        'n_estimators': grid_n_estimator, #default=10
        'criterion': grid_criterion, #default=”gini”
        'max_depth': grid_max_depth, #default=None
        'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
        'random_state': grid_seed
     }],
    [{
        #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
        'learning_rate': grid_learn, #default: .3
        'max_depth': [1,2,4,6,8,10], #default 2
        'n_estimators': grid_n_estimator, 
        'seed': grid_seed  
     }]
]

import time
start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter
for clf, param in zip(vote_test, grid_param): #https://docs.python.org/3/library/functions.html#zip

    #print(clf[1]) #vote_test is a list of tuples, index 0 is the name and index 1 is the algorithm
    #print(param)
    
    
    start = time.perf_counter()        
    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split)
    best_search.fit(X_data, y_data)
    run = time.perf_counter() - start

    best_param = best_search.best_params_
    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param) 


run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

print('-'*10)
#Hard Vote or majority rules w/Tuned Hyperparameters
grid_hard = ensemble.VotingClassifier(estimators = vote_test , voting = 'hard')
grid_hard_cv = model_selection.cross_validate(grid_hard, X_data, y_data, cv  = cv_split)
grid_hard.fit(X_data, y_data)
print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 
print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))
print('-'*10)

pred_type = grid_hard.predict(test_data_tr.values)
output_data = np.column_stack((test_data['Id'].values, pred_type))
np.savetxt("submit.csv", output_data, fmt='%i', delimiter=",", header='Id,Cover_Type', comments='')