import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.style as style

import matplotlib.gridspec as gridspec

%matplotlib inline

import scipy.stats as stats

from scipy.stats import zscore



#import sklearn packages for modelling

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import mean_squared_error
myData = pd.read_csv('../input/concrete.csv')

myData.head()
myData.info()
myData.describe().T
myData = myData.apply(zscore)
(myData.corr()*2)['strength'].sort_values(ascending = False)[1:]
plt.figure(figsize = (15,10))



cor = myData.corr()

mask = np.zeros_like(cor)

mask[np.triu_indices_from(mask)] = True



with sns.axes_style('white'):

    sns.heatmap(cor, annot = True, linewidth = 2, mask = mask)

    

plt.title('Correlation Representation')

plt.show()
def plotting_3_chart(myData, feature):

    style.use('fivethirtyeight')



    fig = plt.figure(constrained_layout = True, figsize = (15,10))

    grid = gridspec.GridSpec(ncols = 3, nrows = 3, figure = fig)



    ax1 = fig.add_subplot(grid[0, :2])

    ax1.set_title('Histogram')

    sns.distplot(myData.loc[:, feature], norm_hist = True, ax = ax1)



    ax2 = fig.add_subplot(grid[1, :2])

    ax2.set_title('QQ plot')

    stats.probplot(myData.loc[:, feature], plot = ax2)



    ax3 = fig.add_subplot(grid[:,2])

    ax3.set_title('Boxplot')

    sns.boxplot(myData.loc[:, feature], orient = 'v', ax = ax3)

    

plotting_3_chart(myData, 'strength')
print('Skewness: '+ str(myData['strength'].skew()))

print('Kurtosis: ' + str(myData['strength'].kurt()))
q1 = round(np.quantile(myData['strength'], 0.25),2)

q2 = round(np.quantile(myData['strength'], 0.50),2)

q3 = round(np.quantile(myData['strength'], 0.75),2)



IQR = q3-q1



print('Quartile q1: ', q1)

print('Quartile q2: ', q2)

print('Quartile q3: ', q3)

print('Interquartile range: ', IQR)



print('Strength above ', myData['strength'].quantile(0.75) + (1.5*IQR), 'are outliers.')

print('Number of outliers', myData[myData['strength']> 2.64]['strength'].shape[0])
import itertools

cols = [i for i in myData.columns if i not in 'strength']

length = len(cols)

cs = ['b','r','g','c','m','k','lime','c']

fig = plt.figure(figsize = (13,25))



for i,j,k in itertools.zip_longest(cols,range(length), cs):

    plt.subplot(4,2,j+1)

    ax = sns.scatterplot(x = myData[i], y = myData['strength'] ,color = k)

    ax.set_facecolor('w')

    plt.axvline(myData[i].mean(), linestyle = 'dashed', label = 'mean', color = 'k')

    plt.legend(loc = 'best')

    plt.title(i, color = 'navy')

    plt.xlabel('')
cols = [i for i in myData.columns if i not in 'strength']

length = len(cols)

cs = ["b","r","g","c","m","k","lime","c"]

fig = plt.figure(figsize=(13,25))



for i,j,k in itertools.zip_longest(cols,range(length),cs):

    plt.subplot(4,2,j+1)

    ax = sns.distplot(myData[i],color=k, rug = True)

    ax.set_facecolor("w")

    plt.axvline(myData[i].mean(),linestyle="dashed",label="mean",color="k")

    plt.legend(loc="best")

    plt.title(i,color="navy")

    plt.xlabel("")
X = myData.drop('strength', axis = 1)

y = myData[['strength']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

X_train,X_val,y_train, y_val = train_test_split(X_train,y_train, test_size = 0.30, random_state = 1)
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score



target = 'strength'

def model(algorithm, d_X_train,d_y_train,d_X_test,d_y_test, of_type):

    print(algorithm)

    print("***************************************************************************")

    algorithm.fit(d_X_train, d_y_train)

    prediction = algorithm.predict(d_X_test)

    print('ROOT MEAN SQUARED ERROR: ', np.sqrt(mean_squared_error(d_y_test, prediction)))

    print("***************************************************************************")

    prediction = pd.DataFrame(prediction)

    cross_val = cross_val_score(algorithm, d_X_train, d_y_train, cv = 20, scoring = 'neg_mean_squared_error')

    cross_val = cross_val.ravel()

    print('CROSS VALIDATION SCORE')

    print("***************************************************************************")

    print('cv-mean: ', cross_val.mean())

    print('cv-std: ', cross_val.std())

    print('cv-max: ', cross_val.max())

    print('cv-min: ', cross_val.min())

    

    plt.figure(figsize = (13,28))

    plt.subplot(211)

    

    y_test = d_y_test.reset_index()['strength']

    

    ax = y_test.plot(label = 'originals', figsize = (12,13), linewidth = 2)

    ax = prediction[0].plot(label = 'predictions', figsize = (12,13), linewidth = 2)

    plt.legend(loc = 'best')

    plt.title('ORIGINALS vs PREDICTIONS')

    plt.xlabel('index')

    plt.ylabel('values')

    ax.set_facecolor('k')

    

    plt.subplot(212)

    

    if of_type == 'coef':

        coef = pd.DataFrame(algorithm.coef_.ravel())

        coef['feat'] = d_X_train.columns

        ax1 = sns.barplot(coef['feat'], coef[0], palette = 'jet_r', linewidth = 2, edgecolor = 'k' * coef['feat'].nunique())

        ax1.set_facecolor('lightgrey')

        ax1.axhline(0, color = 'k', linewidth = 2)

        plt.ylabel('coefficients')

        plt.xlabel('features')

        plt.title('FEATURE IMPORTANCES')

        

    elif of_type == 'feat':

        coef = pd.DataFrame(algorithm.feature_importances_)

        coef['feat'] = d_X_train.columns

        ax2 = sns.barplot(coef['feat'], coef[0], palette = 'jet_r', linewidth = 2, edgecolor = 'k' * coef['feat'].nunique())

        ax2.set_facecolor('lightgrey')

        ax2.axhline(0, color = 'k', linewidth = 2)

        plt.ylabel('coefficients')

        plt.xlabel('features')

        plt.title('FEATURE IMPORTANCES')

        

import warnings

warnings.filterwarnings('ignore')

    
from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

model(linreg, X_train, y_train, X_val, y_val,'coef')

print('Accuracy score:',linreg.score(X_val,y_val))
from sklearn.linear_model import Lasso, Ridge

ls = Lasso()

model(ls, X_train,y_train, X_val, y_val,'coef')

print('Accuracy score: ',ls.score(X_val,y_val))
from sklearn.linear_model import Ridge

rd = Ridge()

model(rd, X_train,y_train, X_val, y_val,'coef')

print('Accuracy score:',rd.score(X_val,y_val))
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

model(dtr, X_train,y_train, X_val, y_val,'feat')

print('Accuracy score:',dtr.score(X_val,y_val))
from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor()

model(ada, X_train,y_train, X_val, y_val,'feat')

print('Accuracy score:',ada.score(X_val,y_val))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

model(rf, X_train,y_train, X_val, y_val,'feat')

print('Accuracy score:',rf.score(X_val,y_val))
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()

model(gbr, X_train,y_train, X_val, y_val,'feat')

print('Accuracy score:',gbr.score(X_val,y_val))
print(pd.DataFrame(gbr.feature_importances_, columns = ['Imp'], index = X_train.columns))
from sklearn import model_selection



gbr = GradientBoostingRegressor()

gbr_kfold = model_selection.KFold(n_splits = 10, random_state = 1)



gbr_kfold_results = cross_val_score(gbr,X,y, cv = gbr_kfold)

print(gbr_kfold_results)
print('Accuracy: %.3f%%, (%.3f%%)' % (gbr_kfold_results.mean() * 100.0, gbr_kfold_results.std()))
from mlxtend.feature_selection import SequentialFeatureSelector as sfs



sfs1 = sfs(gbr, k_features=5, forward=True, scoring='r2', cv=5)

sfs1 = sfs1.fit(X_train.values, y_train.values)
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs



fig = plot_sfs(sfs1.get_metric_dict())



plt.title('Sequential Forward Selection (w. R^2)')

plt.grid()

plt.show()
sfs1.get_metric_dict()
columnList = list(X_train.columns)

feat_cols = list(sfs1.k_feature_idx_)

print(feat_cols)
subsetColumnList = [columnList[i] for i in feat_cols] 

print(subsetColumnList)
gbr = GradientBoostingRegressor()

gbr.fit(X_train[subsetColumnList], y_train)

y_train_pred = gbr.predict(X_train[subsetColumnList])

print('Training accuracy: %.3f' % gbr.score(X_train[subsetColumnList], y_train))
y_test_pred = gbr.predict(X_test[subsetColumnList])

print('Testing accuracy: %.3f' % gbr.score(X_test[subsetColumnList], y_test))
print("\n model_hyperparameters \n" , gbr.get_params() )
from scipy.stats import randint as sp_randint

from sklearn.model_selection import RandomizedSearchCV

param_dist = {'alpha': 0.9, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.1, 'loss': 'ls', 'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_iter_no_change': None, 'presort': 'auto', 'random_state': None, 'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
samples = 10

randomCV = RandomizedSearchCV(gbr, param_distributions=param_dist, n_iter=samples)
from sklearn.cluster import KMeans

cluster_range = range( 2, 6 )   # expect 3 to four clusters from the pair panel visual inspection hence restricting from 2 to 6

cluster_errors = []

for num_clusters in cluster_range:

    clusters = KMeans(num_clusters, n_init = 5)

    clusters.fit(X_train)

    labels = clusters.labels_

    centroids = clusters.cluster_centers_

    cluster_errors.append( clusters.inertia_ )

clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

clusters_df[0:15]
plt.figure(figsize=(12,6))

plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
from sklearn.cluster import KMeans

from scipy.stats import zscore



cluster = KMeans( n_clusters = 4, random_state = 2354 )

cluster.fit(X_train)

prediction = cluster.predict(X_train)

X_train["GROUP"] = prediction     # Creating a new column "GROUP" which will hold the cluster id of each record



#X_train_copy = X_train.copy(deep = True)  # Creating a mirror copy for later re-use instead of building repeatedly
centroids = cluster.cluster_centers_

centroids
import matplotlib.pylab as plt



X_train.boxplot(by = 'GROUP',  layout=(2,4), figsize=(15, 10))