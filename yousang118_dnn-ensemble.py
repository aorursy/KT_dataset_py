import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option('max_columns', 10, 'max_rows', 20)
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
sns.set(style='white', context='notebook', palette='deep')
from matplotlib import font_manager, rc
tr_train = pd.read_csv('../input/X_train.csv', encoding='cp949')
tr_test = pd.read_csv('../input/X_test.csv', encoding='cp949')
tr_train1 = tr_train.copy(deep=True)
tr = pd.concat([tr_train, tr_test])
tr

features = []
f = tr.groupby('cust_id')['amount'].agg([('총구매액', 'sum')]).reset_index()
features.append(f); f
f = tr.groupby('cust_id')['amount'].agg([('구매건수', 'size')]).reset_index()
features.append(f); f
f = tr.groupby('cust_id')['amount'].agg([('평균구매가격', 'mean')]).reset_index()
features.append(f); f
n = tr.gds_grp_nm.nunique()
f = tr.groupby('cust_id')['gds_grp_nm'].agg([('구매상품다양성', lambda x: len(x.unique()) / n)]).reset_index()
features.append(f); f
tr['sales_date'] = tr.tran_date.str[:10]
f = tr.groupby(by = 'cust_id')['sales_date'].agg([('내점일수','nunique')]).reset_index()
features.append(f); f
def weekday(x):
    w = x.dayofweek 
    if w < 4:
        return 1 # 주중
    else:
        return 0 # 주말
f = tr.groupby(by = 'cust_id')['sales_date'].agg([('요일구매패턴', lambda x : pd.to_datetime(x).apply(weekday).value_counts().index[0])]).reset_index()
features.append(f); f
def f1(x):
    k = x.month
    if 3 <= k <= 5 :
        return('봄-구매건수')
    elif 6 <= k <= 8 :
        return('여름-구매건수')
    elif 9 <= k <= 11 :    
        return('가을-구매건수')
    else :
        return('겨울-구매건수')    
    
tr['season'] = pd.to_datetime(tr.sales_date).apply(f1)
f = pd.pivot_table(tr, index='cust_id', columns='season', values='amount', 
                   aggfunc=np.size, fill_value=0).reset_index()
features.append(f); f
tr['ones']= np.ones(len(tr))
cor = pd.pivot_table(tr, values='ones', index='cust_id', columns='gds_grp_mclas_nm',aggfunc=sum,fill_value=0)
f = cor.reset_index()
features.append(f)
f.head()

f = tr.groupby('cust_id')['amount'].agg([('최대구매액', 'max')]).reset_index()
features.append(f)
f
f = tr.groupby('cust_id')['amount'].agg([('최소구매액', 'min')]).reset_index()
features.append(f)
f
f = tr.groupby('cust_id')['amount'].agg([('구매액 중간값', 'median')]).reset_index()
features.append(f)
f
tr['ones']= np.ones(len(tr))
store = pd.pivot_table(tr, values='ones', index='cust_id', columns='store_nm',aggfunc=sum,fill_value=0)
# store['주거래 지점'] = store.idxmax(axis=1)
f = store.reset_index()
features.append(f)
tr['ym'] = tr.tran_date.str[:4] + tr.tran_date.str[5:7]
dm_piv = pd.pivot_table(tr, values='amount', index='cust_id', columns='ym', aggfunc=np.mean, fill_value=0)
dm_piv = dm_piv.reset_index()
f = dm_piv
features.append(f)
tr['ym'] = tr.tran_date.str[:4] + tr.tran_date.str[5:7]
dm_piv = pd.pivot_table(tr, values='amount', index='cust_id', columns='ym', aggfunc=np.mean, fill_value=0)
f= pd.DataFrame(dm_piv)
f=f[f>0]
f['1년 중 거래하는 개월 수'] = f.nunique(axis=1)
f = f['1년 중 거래하는 개월 수'] 
f = f.reset_index()
features.append(f)
tr['ym'] = tr.tran_date.str[:4] + tr.tran_date.str[5:7]
dm_piv = pd.pivot_table(tr, values='ones', index='cust_id', columns='ym', aggfunc=sum, fill_value=0)
dm_piv = dm_piv.reset_index()
f = dm_piv
features.append(f)
tr['ym'] = tr.tran_date.str[:4] + tr.tran_date.str[5:7]
dm_piv = pd.pivot_table(tr, values='ones', index='cust_id', columns='ym', aggfunc=sum, fill_value=0)
f= pd.DataFrame(dm_piv)
f['최대 거래 빈도 월'] = f.idxmax(axis=1)
f = f['최대 거래 빈도 월']
f= pd.to_numeric(f, downcast='float')
f = f.reset_index()
features.append(f)
f.dtypes
tr['dym'] = tr.tran_date.str[:4] + tr.tran_date.str[5:7] + tr.tran_date.str[8:10]
f = tr.groupby('cust_id')['dym'].agg([('최초 구매일', 'min')]).reset_index()
import datetime
f1 = tr.groupby('cust_id')['dym'].agg([('최종 구매일', 'max')]).reset_index()
f2 = tr.groupby('cust_id')['dym'].agg([('최초 구매일', 'min')]).reset_index()
f3 = pd.merge(f1,f2,on='cust_id')
f3['max-min'] = np.zeros(len(f1)) 

for k in range(len(f3)):
    asd = datetime.datetime.strptime(f3.iloc[k,1],"%Y%m%d")-datetime.datetime.strptime(f3.iloc[k,2],"%Y%m%d")
    f3.iloc[k,3] = asd.days
    
features.append(f3)
del f3['최종 구매일']
del f3['최초 구매일']
f3

f3 = pd.merge(f1,f2,on='cust_id')
f3['max-min'] = np.zeros(len(f1)) 

for k in range(len(f3)):
    asd = datetime.datetime.strptime(f3.iloc[k,1],"%Y%m%d")-datetime.datetime.strptime(f3.iloc[k,2],"%Y%m%d")
    f3.iloc[k,3] = asd.days
    
f = tr.groupby('cust_id')['dym'].agg([('내점일 수', 'nunique')]).reset_index()
features.append(f)

f4 = pd.merge(f3,f, on='cust_id')
f4['거래주기'] = f4['max-min']/f4['내점일 수']
del f4['최종 구매일']
del f4['최초 구매일']
del f4['max-min']
del f4['내점일 수']
#features.append(f4)
f4
f = tr.groupby('cust_id')['store_nm'].agg([('한번이라도 이용한 지점 수', 'nunique')]).reset_index()
features.append(f)
f = tr.groupby('cust_id')['gds_grp_nm'].agg([('한번이라도 이용한 코너 수', 'nunique')]).reset_index()
features.append(f)
f
f1 = tr.groupby('cust_id')['gds_grp_mclas_nm'].agg([('한번이라도 이용한 코너 수', 'nunique')]).reset_index()
f2 = tr.groupby('cust_id')['gds_grp_nm'].agg([('한번이라도 이용한 브랜드 수', 'nunique')]).reset_index()
f3 = pd.merge(f1,f2,on='cust_id')
f3['브랜드 충성도'] = f3['한번이라도 이용한 브랜드 수']/f3['한번이라도 이용한 코너 수']
del f3['한번이라도 이용한 브랜드 수']
del f3['한번이라도 이용한 코너 수']
features.append(f3)
f3
f = tr.groupby(['cust_id','dym'])['amount'].agg([('total', 'sum')]).reset_index()
##동일고객,동일날짜 별로 구매액 합계
f = f.groupby('cust_id')['total'].agg([('일 평균 구매액', 'mean')]).astype(int).reset_index()
#features.append(f)
f
f = tr.groupby('cust_id')['gds_grp_mclas_nm'].agg([('주구매코너', lambda x: x.value_counts().index[0])]).reset_index()
f = pd.get_dummies(f, columns=['주구매코너'])  # This method performs One-hot-encoding
f=f[['cust_id','주구매코너_골프','주구매코너_남성 캐주얼','주구매코너_남성정장','주구매코너_농산물','주구매코너_디자이너','주구매코너_시티웨어','주구매코너_주방가전','주구매코너_화장품']]
features.append(f); f
X_train = DataFrame({'cust_id': tr_train.cust_id.unique()})
for f in features :
    X_train = pd.merge(X_train, f, how='left')


X_test = DataFrame({'cust_id': tr_test.cust_id.unique()})
for f in features :
    X_test = pd.merge(X_test, f, how='left')


X_train.info()
y_train = pd.read_csv('../input/y_train.csv')
data1 = pd.merge(X_train,y_train,on='cust_id') 
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()     
    ]



#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = data1['gender']

#index through MLA and save performance to table
row_index = 0

for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, X_train, y = data1['gender'], cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
      
    #save MLA predictions - see section 6 for usage
    alg.fit(X_train, data1['gender'])
    MLA_predict[MLA_name] = alg.predict(X_train)
    
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
vote_est = [
    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc',ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),

    #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    
    #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ('lr', linear_model.LogisticRegressionCV()),
    
    #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),
    
    #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
    ('knn', neighbors.KNeighborsClassifier()),
    
    #SVM: http://scikit-learn.org/stable/modules/svm.html
    ('svc', svm.SVC(probability=True)),
    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
   ('xgb', XGBClassifier())

]


#Hard Vote or majority rules
vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, X_train, y=data1['gender'], cv  = cv_split)
vote_hard.fit(X_train, data1['gender'])

print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 
print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))
print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))
print('-'*10)


#Soft Vote or weighted probabilities
vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, X_train, y=data1['gender'], cv  = cv_split)
vote_soft.fit(X_train, data1['gender'])

print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 
print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))
print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))
print('-'*10)

#WARNING: Running is very computational intensive and time expensive.
#Code is written for experimental/developmental purposes and not production ready!


#Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]


grid_param = [
            [{
            #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
            'n_estimators': grid_n_estimator, #default=50
            'learning_rate': grid_learn, #default=1
            #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
            'random_state': grid_seed
            }],
       
    
            [{
            #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
            'n_estimators': grid_n_estimator, #default=10
            'max_samples': grid_ratio, #default=1.0
            'random_state': grid_seed
             }],

    
            [{
            #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
            'n_estimators': grid_n_estimator, #default=10
            'criterion': grid_criterion, #default=”gini”
            'max_depth': grid_max_depth, #default=None
            'random_state': grid_seed
             }],


            [{
            #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
            #'loss': ['deviance', 'exponential'], #default=’deviance’
            'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
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
            #GaussianProcessClassifier
            'max_iter_predict': grid_n_estimator, #default: 100
            'random_state': grid_seed
            }],
        
    
            [{
            #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
            'fit_intercept': grid_bool, #default: True
            #'penalty': ['l1','l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs
            'random_state': grid_seed
             }],
            
    
            [{
            #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
            'alpha': grid_ratio, #default: 1.0
             }],
    
    
            #GaussianNB - 
            [{}],
    
            [{
            #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
            'n_neighbors': [1,2,3,4,5,6,7], #default: 5
            'weights': ['uniform', 'distance'], #default = ‘uniform’
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }],
            
    
            [{
            #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
            #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
            #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [1,2,3,4,5], #default=1.0
            'gamma': grid_ratio, #edfault: auto
            'decision_function_shape': ['ovo', 'ovr'], #default:ovr
            'probability': [True],
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



start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter
for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip

    #print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm
    #print(param)
    
    
    start = time.perf_counter()        
    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'roc_auc')
    best_search.fit(X_train, data1['gender'])
    run = time.perf_counter() - start

    best_param = best_search.best_params_
    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param) 


run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

print('-'*10)
#Hard Vote or majority rules w/Tuned Hyperparameters
grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
grid_hard_cv = model_selection.cross_validate(grid_hard, X_train, data1['gender'], cv  = cv_split)
grid_hard.fit(X_train, data1['gender'])

print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 
print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))
print('-'*10)

#Soft Vote or weighted probabilities w/Tuned Hyperparameters
grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
grid_soft_cv = model_selection.cross_validate(grid_soft, X_train, data1['gender'] , cv  = cv_split)
grid_soft.fit(X_train, data1['gender'])

print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 
print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))
print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))
print('-'*10)


#prepare data for modeling
print(X_test.info())
print("-"*10)
#data_val.sample(10)

X_test['gender1'] = vote_hard.predict(X_test)
X_test['gender2'] = grid_hard.predict(X_test)
X_test['gender3'] = vote_soft.predict(X_test)
X_test['gender4'] = grid_soft.predict(X_test)
submit = X_test[['cust_id','gender1']]
submit.to_csv("submit1.csv", index='cust_id')

submit2 = X_test[['cust_id','gender2']]
submit2.to_csv("submit2.csv", index='cust_id')

submit3 = X_test[['cust_id','gender3']]
submit3.to_csv("submit3.csv", index='cust_id')

submit4 = X_test[['cust_id','gender4']]
submit5.to_csv("submit4.csv", index='cust_id')

print('Validation Data Distribution: \n', X_test['gender'].value_counts(normalize = True))
submit.sample(10)














