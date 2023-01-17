###########################################

# Suppress matplotlib user warnings

# Necessary for newer version of matplotlib

import warnings

warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")

#

# Display inline matplotlib plots with IPython

from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')

###########################################



import matplotlib.pyplot as pl

import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import ShuffleSplit, train_test_split, learning_curve, validation_curve, KFold, cross_val_score, GridSearchCV

import seaborn as sns

import pandas as pd

from pandas.tools.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor

from sklearn.metrics import mean_squared_error,make_scorer,r2_score
# Pretty display for notebooks

%matplotlib inline



# Load the Boston housing dataset

filename = ("../input/housing.csv")

names = ['CRIME', 'ResZoNe', 'N-Retail Bussiness', 'CHARS River', 'NitrOxd', 'Rooms/dwell', 'AGE', 'Dis to comp', 'Dist to highway',

         'TAX', 'PTRATIO', 'B','LSTAT', 'MedV']

dataset = pd.read_csv(filename, delim_whitespace=True, names=names)

prices = dataset['MedV']

features = dataset.drop('MedV', axis = 1)

# Success

print("Boston housing dataset has {} data points with {} variables each.".format(*dataset.shape))
features.head(10)
# histograms

dataset.hist(figsize=(9,7),grid=False);
features=features.drop(['CRIME', 'ResZoNe', 'CHARS River', 'NitrOxd', 'AGE', 'Dis to comp',  'TAX', 'B',], axis=1)
features.head()
sns.distplot(prices)
corr=dataset.corr()

pl.figure(figsize=(10, 10))

sns.heatmap(corr, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="white")

pl.title('Correlation between features');
features=features.drop(['N-Retail Bussiness','Dist to highway'], axis=1)
features.head()


# Feature observation

pl.figure(figsize=(20, 5))

for i, col in enumerate(features.columns):

    pl.subplot(1, 13, i+1)

    pl.plot(dataset[col], prices, 'o')

    pl.title(col)

    pl.xlabel(col)

    pl.ylabel('prices')
X_train, X_test, y_train, y_test = train_test_split(features, prices,test_size=0.2, random_state=31)



print("Training and testing split was successful.")
# Test options and evaluation metric using Root Mean Square error method

num_folds = 10

seed = 7

RMS = 'neg_mean_squared_error'
# Spot Check Algorithms

models = []

models.append(('LR', LinearRegression()))

models.append(('LASSO', Lasso()))

models.append(('EN', ElasticNet()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('CART', DecisionTreeRegressor()))

models.append(('SVR', SVR()))
results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=RMS)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
def boxplot():

    fig = pl.figure()

    fig.suptitle('Algorithm Comparison')

    ax = fig.add_subplot(111)

    pl.boxplot(results)

    ax.set_xticklabels(names)

    pl.show()

boxplot()
pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))

pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))

pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))

pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))

results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=RMS)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
boxplot()
scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])

param_grid = dict(n_neighbors=k_values)

model = KNeighborsRegressor()

kfold = KFold(n_splits=num_folds, random_state=seed)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold)

grid_result = grid.fit(rescaledX, y_train)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
ensembles = []

ensembles.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor(n_neighbors=7))])))

ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))

ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))

ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor(n_estimators=100))])))

ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))

results = []

names = []

for name, model in ensembles:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=RMS)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
boxplot()
scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

param_grid = dict(n_estimators=range(21,41,1))

model_train = GradientBoostingRegressor(random_state=seed)

kfold = KFold(n_splits=num_folds, random_state=seed)

grid = GridSearchCV(estimator=model_train, param_grid=param_grid, scoring=RMS, cv=kfold)

grid_result = grid.fit(rescaledX, y_train)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
def PlotLearningCurve(X, y):

    """ Calculates the performance of several models with varying sizes of training data.

        The learning and testing scores for each model are then plotted. """



    # Create 10 cross-validation sets for training and testing

    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)



    # Generate the training set sizes increasing by 50

    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)



    # Create the figure window

    fig = pl.figure(figsize=(10,7))



    # Create three different models based on n_estimators

    for k, n_est in enumerate([20,25,30,35]):

        

        regressor=GradientBoostingRegressor(n_estimators=n_est)

        # Calculate the training and testing scores

        sizes, train_scores, test_scores = learning_curve(regressor, X, y, \

            cv = cv, train_sizes = train_sizes, scoring = 'r2')



        # Find the mean and standard deviation for smoothing

        train_std = np.std(train_scores, axis = 1)

        train_mean = np.mean(train_scores, axis = 1)

        test_std = np.std(test_scores, axis = 1)

        test_mean = np.mean(test_scores, axis = 1)



        # Subplot the learning curve

        ax = fig.add_subplot(2, 2, k+1)

        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')

        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')

        ax.fill_between(sizes, train_mean - train_std, \

            train_mean + train_std, alpha = 0.15, color = 'r')

        ax.fill_between(sizes, test_mean - test_std, \

            test_mean + test_std, alpha = 0.15, color = 'g')



        # Labels

        ax.set_title('n_estimators = %s'%(n_est))

        ax.set_xlabel('Number of Training Points')

        ax.set_ylabel('Score')

        ax.set_xlim([0, X.shape[0]*0.8])

        ax.set_ylim([-0.05, 1.05])



    # Visual aesthetics

    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)

    fig.suptitle('Learning Algorithm LC Performances', fontsize = 16, y = 1.03)

    fig.tight_layout()

    fig.show()
def PLotComplexityCurve(X, y):

    """ Calculates the performance of the model as model complexity increases.

        The learning and testing errors rates are then plotted. """



    # Create 10 cross-validation sets for training and testing

    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)



    # Vary the n_estimators parameter from 1 to 100

    n_estimators = np.arange(1,100)



    # Calculate the training and testing scores

    train_scores, test_scores = validation_curve(GradientBoostingRegressor(), X, y, \

        param_name = "n_estimators", param_range = n_estimators, cv = cv, scoring = 'r2')



    # Find the mean and standard deviation for smoothing

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)



    # Plot the validation curve

    pl.figure(figsize=(7, 5))

    pl.title('Learning algorithm Complexity Performance')

    pl.plot(n_estimators, train_mean, 'o-', color = 'r', label = 'Training Score')

    pl.plot(n_estimators, test_mean, 'o-', color = 'g', label = 'Validation Score')

    pl.fill_between(n_estimators, train_mean - train_std, \

        train_mean + train_std, alpha = 0.15, color = 'r')

    pl.fill_between(n_estimators, test_mean - test_std, \

        test_mean + test_std, alpha = 0.15, color = 'g')



    # Visual aesthetics

    pl.legend(loc = 'lower right')

    pl.xlabel('N Estimators')

    pl.ylabel('Score')

    pl.ylim([-0.05,1.05])

    pl.show() 
PlotLearningCurve(X_train, y_train)
PLotComplexityCurve(X_train, y_train)
# prepare the model

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

model = GradientBoostingRegressor(random_state=seed, n_estimators=28)

model.fit(rescaledX, y_train)

# transform the validation dataset

rescaledValidationX = scaler.transform(X_test)

predictions = model.predict(rescaledValidationX)

print(mean_squared_error(y_test, predictions))
predictions=predictions.astype(int)

submission = pd.DataFrame({

        "Org House Price": y_test,

        "Pred House Price": predictions

    })



print(submission.head(10))