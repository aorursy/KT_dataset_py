import pandas as pd                 # -> to operate on dataset

import numpy as np                  # -> to modify dataset

import matplotlib.pyplot as plt     # -> to plot some dataset

import seaborn as sns               # -> to make plotting fun



# Some of these imports aren't used YET

# Here we import methods to analyze classifiers

from sklearn.feature_selection import RFE

from sklearn.feature_selection import SelectKBest

from sklearn.decomposition import KernelPCA, PCA

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV

from sklearn.model_selection import learning_curve, validation_curve, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report



# Here we import predictors

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF



# Stand-alone library - XGBoost

from xgboost import XGBClassifier



# Here we import stuff to create Neural Network for classification

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Activation, GaussianNoise, Input

from keras.optimizers import Adagrad, Adam, RMSprop, TFOptimizer

from keras.metrics import binary_crossentropy, mse

from keras.callbacks import CSVLogger, ModelCheckpoint

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
dataframe = pd.read_csv('../input/adult.csv')



dataframe = dataframe[dataframe['occupation'] != '?']

dataframe = dataframe[dataframe['workclass'] != '?']

dataframe = dataframe[dataframe['native.country'] != '?']
for name in dataframe.columns.values:            #  ->  All column names

    print(name,"unique values:")

    print(np.sort(dataframe[name].unique()))     #  ->  Sort the unique values

    print("\n")                                  #      of column



print(dataframe.shape)
map_list = []      #  -> list of maps

to_map = ['workclass', 'education', 'marital.status', 'occupation', 'relationship',

         'race', 'sex', 'native.country', 'income']
for column in to_map:

    mapper = {}                   #  ->  we create map for every column

    for index, unique in enumerate(np.sort(dataframe[column].unique())):

        mapper[unique] = index    #  ->  every unique value of column will

    map_list.append(mapper)       #     have unique index
map_list
for column, mapper in zip(to_map, map_list):

    dataframe[column] = dataframe[column].map(mapper)
"""Simple check if everything is OK"""



for name in dataframe.columns.values:

    print(name,"unique values:")

    print(np.sort(dataframe[name].unique()))

    print("\n")



print(dataframe.shape)
target = dataframe['income']

data = dataframe.drop(['income'], axis=1)
plt.figure(figsize=(11,11))

colormap = plt.cm.viridis_r

sns.heatmap(data.corr(), vmax=1.0, cmap=colormap, annot=True)

plt.show()
plt.figure(figsize=(1,1))

sns.pairplot(dataframe[dataframe.columns.values], hue='income',)

plt.show()
data = data.as_matrix()

target = target.as_matrix()

del dataframe
print(data.shape)

print(target.shape)
rand_state=42

C_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]



svc_params = {

    "C":C_range,

    "kernel":['rbf', 'sigmoid'],

    "degree":[2,3,4],

    "gamma":['auto',0.001, 0.02, 0.4],

    "coef0":[-2.,0.,0.001,0.2,1,2],

    "shrinking":[False,True],

    "tol":[1e-3, 1e-6,1e-4,0.01],

    "cache_size":[600],

    "max_iter":[-1],

    "random_state":[rand_state]

}



log_params = [{

    "C":C_range,

    "penalty":['l1'],

    "dual":[False],

    "fit_intercept":[False, True],

    "solver":['liblinear'],

    "intercept_scaling":[0.01, 0.1, 1., 10., 100.],

    "tol":[1e-4, 1e-6, 1e-3, 0.01, 0.1],

    "warm_start":[False, True],

    "random_state":[rand_state]

},            {

    "C":C_range,

    "penalty":["l2"],

    "dual":[False],

    "fit_intercept":[False, True],

    "intercept_scaling":[0.01, 0.1, 1., 10., 100.],

    "solver": ["lbfgs", "sag", "newton-cg"],

    "max_iter":[100, 200, 500, 1000],

    "tol":[1e-4, 1e-6, 1e-5, 1e-3, 0.1],

    "warm_start":[False, True],

    "random_state":[rand_state]

}]



ridge_params = {

    "alpha":[0.01, 0.1, 1.0, 10., 100.],

    "copy_X":[False, True],

    "fit_intercept":[False, True],

    "max_iter":[1000, 2000, 5000, 10000, 15000],

    "normalize":[False, True],

    "solver":['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],

    "tol":[1e-4, 1e-6, 1e-5, 0.01, 0.1],

    "random_state":[rand_state]

}



sgd_params = {

    'loss':['hinge', 'log', 'modified_huber', 'squared_hinge'],

    "penalty":['l1', 'l2', 'elasticnet'],

    "alpha":[0.001, 0.01, 0.1, 1.0, 10.],

    "l1_ratio":[0.15, 0.30, 0.5, 0.70, 0.9],

    "fit_intercept":[False, True],

    "shuffle":[True],

    "random_state":[rand_state],

    "learning_rate":["constant", "optimal", "invscaling"],

    "eta0":[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],

    "power_t":[1,2,3,4,5],

    "warm_start":[False, True],

    "average":[False, True]

}



ada_params = {

    "n_estimators":[50, 100, 200, 500, 1000],

    "learning_rate":[0.001, 0.01, 0.1, 1.0, 10.],

    "random_state":[rand_state]

}



grad_params = {

    "loss":["deviance", "exponential"],

    "learning_rate":[0.001, 0.01, 0.1, 1.0, 10.],

    "max_depth":[2, 3, 4, 6, 8, 12],

    "criterion":["friedman_mse", "mse", "mae"],

    "min_samples_split":[2, 4, 6, 8],

    "min_samples_leaf":[1, 2, 4, 6, 8],

    "max_features":["auto", "sqrt", "log2", None],

    "max_leaf_nodes":[8, 10, 12, 15, 30, 40],

    "min_impurity_split":[1e-6, 1e-5, 1e-3, 0.1, 1.],

    "random_state":[rand_state]

}



et_params = {

    "n_estimators":[10, 20, 50, 100, 300, 800, 1200],

    "criterion": ["gini", "entropy"],

    "max_features":["auto", "sqrt", "log2", None],

    "max_depth":[3, 5, 8, 10, 15],

    "min_samples_split":[2, 3, 4, 6, 8],

    "min_samples_leaf":[1, 2, 4, 6, 8],

    "max_leaf_nodes":[8, 10, 12, 15, 30, 40],

    "min_impurity_split":[1e-7, 1e-9, 1e-5, 1e-3, 0.1, 1.],

    "bootstrap":[False, True],

    "oob_score":[False, True],

    "random_state":[rand_state],

    "warm_start":[False, True]

}



kn_params = {

    "n_neighbors":[10, 30, 50, 100, 150],

    "weights":["uniform", "distance"],

    "algorithm":["auto", "ball_tree", "kd_tree"],

    "leaf_size":[30, 50, 80, 150, 200],

    "p":[2, 3, 4, 5],

}



gpc_params = {

    "kernel":[1.0*RBF(), 0.5*RBF(), 0.1*RBF(), 1.5*RBF()],

    "n_restarts_optimizer":[0, 1, 5, 10],

    "max_iter_predict":[100, 200, 500, 1000, 1500],

    "warm_start":[False, True],

    "copy_X_train":[False, True],

    "random_state":[rand_state]

}



xgb_params = {

    "max_depth":[3,5,10,15,30],

    "learning_rate":[0.001, 0.01, 0.1, 1.0, 10],

    "n_estimators":[100, 200, 500, 1000, 1500],

    "silent":[True],

    "gamma":[0.0, 0.0001, 0.001, 0.1],

    "min_child_weight":[1,2,4,6,8],

    "max_delta_step":[0,1,2,4],

    "reg_alpha":[0.0, 0.15, 0.35, 0.65, 0.80],

    "reg_lambda":[0.0, 0.15, 0.35, 0.65, 0.80],

    "seed":[rand_state]

}
scaler = StandardScaler()           #  -> For me SVC need to have normalized data

data = scaler.fit_transform(data)



data_1 = data

target_1 = target



X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33)



data = X_train

target = y_train



cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=rand_state)

rand_iter = 10
clf = SVC()

score = RandomizedSearchCV(clf, param_distributions=svc_params, n_iter=rand_iter, verbose=2, cv=cv, n_jobs = -1)

score.fit(data, target)

print("\n\nBest score: %f Best parameters: %s" % (score.best_score_,score.best_params_))

print("\n")

means = score.cv_results_['mean_test_score']

stds = score.cv_results_['std_test_score']

params = score.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
clf = LogisticRegression()

score = RandomizedSearchCV(clf, param_distributions=log_params[1], n_iter=rand_iter, verbose=2, cv=cv,n_jobs = -1)

score.fit(data, target)

print("\n\nBest score: %f Best parameters: %s" % (score.best_score_,score.best_params_))

print("\n")

means = score.cv_results_['mean_test_score']

stds = score.cv_results_['std_test_score']

params = score.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
clf = RidgeClassifier()

score = RandomizedSearchCV(clf, param_distributions=ridge_params, n_iter=rand_iter, verbose=2, cv=cv,n_jobs = -1)

score.fit(data, target)

print("\n\nBest score: %f Best parameters: %s" % (score.best_score_,score.best_params_))

print("\n")

means = score.cv_results_['mean_test_score']

stds = score.cv_results_['std_test_score']

params = score.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
clf = AdaBoostClassifier()

score = RandomizedSearchCV(clf, param_distributions=ada_params, n_iter=rand_iter, verbose=2, cv=cv,n_jobs = -1)

score.fit(data, target)

print("\n\nBest score: %f Best parameters: %s" % (score.best_score_,score.best_params_))

print("\n")

means = score.cv_results_['mean_test_score']

stds = score.cv_results_['std_test_score']

params = score.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
clf = GradientBoostingClassifier()

score = RandomizedSearchCV(clf, param_distributions=grad_params, n_iter=rand_iter, verbose=2, cv=cv, n_jobs = -1)

score.fit(data, target)        # I HAVE CHANGED THE N_ITER VALUE

print("\n\nBest score: %f Best parameters: %s" % (score.best_score_,score.best_params_))

print("\n")

means = score.cv_results_['mean_test_score']   

stds = score.cv_results_['std_test_score']

params = score.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
clf = KNeighborsClassifier()

score = RandomizedSearchCV(clf, param_distributions=kn_params, n_iter=rand_iter, verbose=2, cv=cv, n_jobs = -1)

score.fit(data, target)                    # I HAVE CHANGED THE N_ITER VALUE FOR PROCCESSING SAKE

print("\n\nBest score: %f Best parameters: %s" % (score.best_score_,score.best_params_))

print("\n")

means = score.cv_results_['mean_test_score']

stds = score.cv_results_['std_test_score']

params = score.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
clf = SGDClassifier()

score = RandomizedSearchCV(clf, param_distributions=sgd_params, n_iter=rand_iter, verbose=2, cv=cv, n_jobs = -1)

score.fit(data, target)

print("\n\nBest score: %f Best parameters: %s" % (score.best_score_,score.best_params_))

print("\n")

means = score.cv_results_['mean_test_score']

stds = score.cv_results_['std_test_score']

params = score.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))






