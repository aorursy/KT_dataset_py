# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







import pandas as pd

import numpy as np

import random

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV

import pickle

import warnings

from scipy.stats import skew

from scipy.stats import norm

from scipy import stats

#Model 

from sklearn.preprocessing import StandardScaler 

warnings.filterwarnings('ignore')

sns.set(style='white', context='notebook', palette='deep')

%config InlineBackend.figure_format = 'retina'







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Housing.csv', sep = ';', encoding= 'latin-1', index_col = 0)
data.head(5)
data.columns
data.info()
data.shape
data.dtypes
ax = data.plot(x='lotsize', y= 'price', kind= 'scatter')

ax.set_title("Prix de vente d'une maison par rapport au m2")
data.loc[:, data.columns != 'ID'].hist(figsize=(30, 20), bins=50);
data.isnull().sum()
data.describe()
#Verifions que la distribution de la variable cible price s'approche d'une gaussienne
sns.distplot(data['price'], fit=norm);

(mu, sigma) = norm.fit(data['price'])

print('\n mu={:.2f} and sigma= {:.2f} \n'.format(mu,sigma))

plt.legend(['Normal dist.($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu,sigma)], loc = 'best')

plt.ylabel('Frequency')

plt.title  (' Price Distribution')

fig = plt.figure()

res = stats.probplot(data['price'],plot=plt)

plt.show()

print("skewness: %f" % data['price'].skew())

print("kurtosis: %f" % data['price'].kurt())
data['log_price']=np.log(data['price'])

data = data.drop(['price'], axis=1)
sns.distplot(data['log_price'], fit=norm)

(mu,sigma)=norm.fit(data['log_price'])

print('\n mu={:.2f} and sigma={:.2f}\n'.format(mu,sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)], loc='best')

plt.ylabel('Frequnecy')

plt.title('Price Distribution')

plt.show()

fig = plt.figure()

res = stats.probplot(data['log_price'],plot=plt)

plt.show()

print("skewness: %f" % data['log_price'].skew())

print("kurtosis: %f" % data['log_price'].kurt())
#On visualise les distributions des différentes variables numériques avec des histogrammes 

data.hist(bins=50, figsize=(20, 15))

plt.show()
sns.set(style= 'whitegrid', context ='notebook')

sns.pairplot(data, height=3)

plt.show()
for col_name in data.select_dtypes(object).columns:

    plt.figure()

    sns.boxplot(x=col_name, y="log_price", data = data )
def heatMap(df, mirror):



   # On calcule la matrice de corrélation

   corr = df.corr()

   # On affiche la figure 

   fig, ax = plt.subplots(figsize=(10, 10))

   # On genère le grille de couleur 

   colormap = sns.diverging_palette(220, 10, as_cmap=True)

   

   if mirror == True:

      

      sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")

      # xticks

      plt.xticks(range(len(corr.columns)), corr.columns);

      # yticks

      plt.yticks(range(len(corr.columns)), corr.columns)

      #show plot



   else:

      # Drop self-correlations

      dropSelf = np.zeros_like(corr)

      dropSelf[np.triu_indices_from(dropSelf)] = True

      # Generate Color Map

      colormap = sns.diverging_palette(220, 10, as_cmap=True)

      # Generate Heat Map, allow annotations and place floats in map

      sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)

      # Apply xticks

      plt.xticks(range(len(corr.columns)), corr.columns);

      # Apply yticks

      plt.yticks(range(len(corr.columns)), corr.columns)

   # show plot

   plt.show()
heatMap(data, False)
# On encode les variables non numériques 

for c in data.select_dtypes(object).columns:    

    encoder = LabelEncoder()    

    encoder.fit(list(data[c]))    

    data[c] = encoder.transform(list(data[c]))
heatMap(data, False)
import xgboost as xgb

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor 

from sklearn.linear_model import LinearRegression , Ridge , Lasso
#on définit la metrique utilisée pour évaluer l'algorithme.

def rmse(preds,targets, sample_weight=None, multioutput='uniform_average'):

    #on applique la fonction exp afin de revenir au price de depart et avoir une idée réaliste de l'erreur.

    preds = np.exp(preds)

    target= np.exp(targets)

    return np.sqrt(((preds - targets)**2).mean())
def r2(preds, targets):

    #on applique la fonction exp afin de revenir au price de depart et avoir une idée réaliste de l'erreur.

    preds = np.exp(preds)

    targets = np.exp(targets)

    return 1- (np.sum((targets-preds)**2)/np.sum((targets - np.mean(targets))**2))
def evaluate(model):

    preds_train = model.predict(X_train)

    preds_test = model.predict(X_test)

    # on affiche  preds = f(y)

    plt.figure()

    plt.scatter(y_test, preds_test)  

    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

    plt.xlabel('log_price')

    plt.ylabel('prediction')

    plt.title('pred = f(log_price)')

    # les résidus

    plt.figure()

    plt.scatter(y_test, abs(preds_test - y_test))  

    plt.xlabel('preds_log_price')

    plt.ylabel('risidual')

    plt.title('risidual = f(fitted_log_price)')   

    plt.show()

    rmse_train = rmse(preds_train, y_train)

    rmse_test = rmse(preds_test, y_test)

    r2_train = r2(preds_train, y_train)

    r2_test = r2(preds_test, y_test)   

    return rmse_train, rmse_test, r2_train, r2_test
col_names = ['model name', 'rmse_train', 'rmse_test', 'r2_train', 'r2_test']

results = pd.DataFrame(columns = col_names)

results
linear_model = LinearRegression()

linear_model.fit(X_train,y_train)

rmse_train, rmse_test, r2_train, r2_test = evaluate(linear_model)

new_row = {'model name':" linear regression", 'rmse_train':rmse_train, 'rmse_test':rmse_test, 'r2_train':r2_train, 'r2_test':r2_test}

results = results.append(new_row, ignore_index=True)

print("{} train train rmse : {:.3f} test rmse : {:.3f} train r2 : {:.3f} test r2  : {:.3f}".format(new_row['model name'],new_row['rmse_train'], new_row['rmse_test'] ,new_row['r2_train'], new_row['r2_test']))
randomforest_model = RandomForestRegressor(n_estimators = 150, random_state=0)

randomforest_model.fit(X_train,y_train)

rmse_train, rmse_test, r2_train, r2_test = evaluate(randomforest_model)

new_row = {'model name':"randomForest regressor", 'rmse_train':rmse_train, 'rmse_test':rmse_test, 'r2_train':r2_train, 'r2_test':r2_test}

results = results.append(new_row, ignore_index=True)

print("{} train train rmse : {:.3f} test rmse : {:.3f} train r2 : {:.3f} test r2  : {:.3f}".format(new_row['model name'],new_row['rmse_train'], new_row['rmse_test'] ,new_row['r2_train'], new_row['r2_test']))
xgboost_model = XGBRegressor(n_estimators=150, random_state=0)

xgboost_model.fit(X_train,y_train)

rmse_train, rmse_test, r2_train, r2_test = evaluate(xgboost_model)

new_row = {'model name':"xgboost regressorr", 'rmse_train':rmse_train, 'rmse_test':rmse_test, 'r2_train':r2_train, 'r2_test':r2_test}

results = results.append(new_row, ignore_index=True)

print("{} train train rmse : {:.3f} test rmse : {:.3f} train r2 : {:.3f} test r2  : {:.3f}".format(new_row['model name'],new_row['rmse_train'], new_row['rmse_test'] ,new_row['r2_train'], new_row['r2_test']))

results
#on prepare la liste des hyperparametres

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 50)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(3, 30, num = 3)]

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

# On crée  randomgrid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
# Tuning hyper-parameters by RandomizedSearchCV

tune_rf = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = random_grid, n_iter = 200, cv = 4, scoring = 'neg_mean_squared_error', verbose=2, random_state = 167, n_jobs = -1)

tune_rf.fit(X_train, y_train)
tune_rf.best_params_
# on stocke le score du meilleur estimateur 

best_estimator  =  tune_rf.best_estimator_ 

print("Name: {} train score : {} test score : {}".format(new_row['model name'],new_row['rmse_train'], new_row['rmse_test']))

rmse_train, rmse_test, r2_train, r2_test = evaluate(best_estimator)

new_row = {'model name':"tuned_randomForest regressor", 'rmse_train':rmse_train, 'rmse_test':rmse_test, 'r2_train':r2_train, 'r2_test':r2_test}

results = results.append(new_row, ignore_index=True)

print("{} train train rmse : {:.3f} test rmse : {:.3f} train r2 : {:.3f} test r2  : {:.3f}".format(new_row['model name'],new_row['rmse_train'], new_row['rmse_test'] ,new_row['r2_train'], new_row['r2_test']))



features = data.columns

importances = best_estimator.feature_importances_

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
# Tuning hyper-parameters by GridSearchCV

param_grid = {'max_depth': [int(x) for x in np.linspace(3, 30, 3)],

              'min_child_weight': [1, 3, 5],

              'subsample': [i/10.0 for i in range(6, 10)],

              }

tuned_xgboost  = GridSearchCV(estimator = xgb.XGBRegressor(learning_rate = 0.1, n_estimators = 200, colsample_bytree = 0.8, objective= 'reg:linear', 

                                            nthread = 4, scale_pos_weight = 1, seed = 27), param_grid = param_grid, scoring = 'neg_mean_squared_error',

                                            n_jobs = -1, iid = False, cv = 4)

tuned_xgboost.fit(X_train, y_train)
tuned_xgboost.best_params_
# on stocke le score du meilleur estimateur 

best_estimator  =  tuned_xgboost.best_estimator_ 

rmse_train, rmse_test, r2_train, r2_test = evaluate(best_estimator)

new_row = {'model name':"tuned XGboost regressor", 'rmse_train':rmse_train, 'rmse_test':rmse_test, 'r2_train':r2_train, 'r2_test':r2_test}

results = results.append(new_row, ignore_index=True)

print("{} train train rmse : {:.3f} test rmse : {:.3f} train r2 : {:.3f} test r2  : {:.3f}".format(new_row['model name'],new_row['rmse_train'], new_row['rmse_test'] ,new_row['r2_train'], new_row['r2_test']))



#Importance des variables 

xgb.plot_importance(best_estimator,max_num_features=20)

plt.show()
params = {'alpha': [i for i in range(100)]}

reg = GridSearchCV(Ridge(), param_grid = params, cv = 4, scoring = 'neg_mean_squared_error')

reg.fit(X_train, y_train)
# on stocke le score du meilleur estimateur 

best_estimator  =  reg.best_estimator_ 

rmse_train, rmse_test, r2_train, r2_test = evaluate(best_estimator)

new_row = {'model name':"tuned Ridge regressor", 'rmse_train':rmse_train, 'rmse_test':rmse_test, 'r2_train':r2_train, 'r2_test':r2_test}

results = results.append(new_row, ignore_index=True)

print("{} train train rmse : {:.3f} test rmse : {:.3f} train r2 : {:.3f} test r2  : {:.3f}".format(new_row['model name'],new_row['rmse_train'], new_row['rmse_test'] ,new_row['r2_train'], new_row['r2_test']))



reg = GridSearchCV(Lasso(), param_grid = params, cv = 4, scoring = 'neg_mean_squared_error')

reg.fit(X_train, y_train)




# on stocke le score du meilleur estimateur 

best_estimator  =  reg.best_estimator_ 

rmse_train, rmse_test, r2_train, r2_test = evaluate(best_estimator)

new_row = {'model name':"tuned Lasso regressor", 'rmse_train':rmse_train, 'rmse_test':rmse_test, 'r2_train':r2_train, 'r2_test':r2_test}

results = results.append(new_row, ignore_index=True)

print("{} train train rmse : {:.3f} test rmse : {:.3f} train r2 : {:.3f} test r2  : {:.3f}".format(new_row['model name'],new_row['rmse_train'], new_row['rmse_test'] ,new_row['r2_train'], new_row['r2_test']))



# on tri par score obtenu sur le jeu d'entrainement.

results =  results.sort_values(['rmse_test'], ascending = 1)

results