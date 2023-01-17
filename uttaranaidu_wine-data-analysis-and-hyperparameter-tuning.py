#Importing Data manipulation and plotting modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from xgboost import plot_importance

import seaborn as sns

import os
#Importing libraries for Data pre-processing

from sklearn.preprocessing import StandardScaler as ss
#Importing model for Dimentionality Reduction

from sklearn.decomposition import PCA
#Importing libraries for performance measures

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve
#Importing libraries For data splitting

from sklearn.model_selection import train_test_split
#Importing libraries for Model pipelining

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline
#Importing libraries for model parameter search and hyperparameter tuning

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from bayes_opt import BayesianOptimization

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import cross_val_score

import eli5

from eli5.sklearn import PermutationImportance
#Importing Miscelaneous libraries

import time

import os

from scipy.stats import uniform
#Display all rows

pd.set_option('display.max_columns', 100)
os.chdir("../input") 

data = pd.read_csv("winequalityN.csv")
data.head()
data.info()
data.shape
data.isnull().any()
data.dropna(axis=0,inplace=True)
data.isnull().any()
data.shape
g = sns.pairplot(data,palette="husl",diag_kind="kde",hue='type')
plt.figure(figsize=(12, 12))

sns.heatmap(data.corr(),annot=True,vmin=-1,cmap='YlGnBu')
plt.figure(figsize=(12,10))

sns.barplot(x=data.type, y=data.alcohol,data=data, hue=data.quality,palette='spring')

plt.xlabel('Type',fontsize=16)

plt.ylabel('Quality',fontsize=16)
#Assuming wines with quality greater than 6 are best



wine_data = data.shape[0]

print("Total number of wine data: "+str(wine_data))

high_quality = data.loc[(data.quality)> 6]

print("Best quality wine entries: "+str(high_quality.shape[0]))

good_quality = data.loc[(data.quality)<5]

print("Good quality wine entries: "+str(good_quality.shape[0]))

end_of_month_wine = data.loc[(data.quality) < 3]

print("Mediocre quality wine entries: "+str(good_quality.shape[0]))
table = data.pivot_table('alcohol',index='quality',columns='type')

q = table.quantile(0.90)

df = table[table < q]

plt.figure(figsize=(12,8))

plt.xticks(rotation=45)

ax = sns.boxplot(data=data)
table_fa = data.pivot_table('fixed acidity',index='quality',columns='type',aggfunc='sum').sum(axis=1)

table_va = data.pivot_table('volatile acidity',index='quality',columns='type',aggfunc='sum').sum(axis=1)

table_ca = data.pivot_table('citric acid',index='quality',columns='type',aggfunc='sum').sum(axis=1)

table_rs = data.pivot_table('residual sugar',index='quality',columns='type',aggfunc='sum').sum(axis=1)

table_chl = data.pivot_table('chlorides',index='quality',columns='type',aggfunc='sum').sum(axis=1)

table_tsd = data.pivot_table('total sulfur dioxide',index='quality',columns='type',aggfunc='sum').sum(axis=1)

table_den = data.pivot_table('density',index='quality',columns='type',aggfunc='sum').sum(axis=1)

table_pH = data.pivot_table('pH',index='quality',columns='type',aggfunc='sum').sum(axis=1)

table_sul = data.pivot_table('sulphates',index='quality',columns='type',aggfunc='sum').sum(axis=1)

table_alcohol = data.pivot_table('alcohol',index='quality',columns='type',aggfunc='sum').sum(axis=1)

qual = table_alcohol.index.astype(int)

lbl = ['fa','va','ca','rs','chl','tsd','den','pH','sul','alcohol']



plt.figure(figsize=(12,14))

ax = sns.pointplot(x=qual, y=table_fa, color='y',scale=0.7)

ax = sns.pointplot(x=qual, y=table_va, color='b',scale=0.7)

ax = sns.pointplot(x=qual, y=table_ca, color='m', scale=1.0)

ax = sns.pointplot(x=qual, y=table_rs, color='thistle', scale=1.0)

ax = sns.pointplot(x=qual, y=table_chl, color='g', scale=1.0)

ax = sns.pointplot(x=qual, y=table_tsd, color='coral',scale=0.7)

ax = sns.pointplot(x=qual, y=table_den, color='olive', scale=1.0)

ax = sns.pointplot(x=qual, y=table_pH, color='sienna', scale=1.0)

ax = sns.pointplot(x=qual, y=table_sul, color='m', scale=1.0)

ax = sns.pointplot(x=qual, y=table_alcohol, color='springgreen', scale=1.0)



ax.set_xlabel(xlabel='Quality', fontsize=16)

ax.set_ylabel(ylabel='Quantity of each component', fontsize=16)

ax.legend(handles=ax.lines[::len(qual)+1],labels=lbl,fontsize=12)

plt.show();
df = data.groupby(['quality'])

df_mean = df['fixed acidity','volatile acidity','citric acid'].aggregate(np.mean)

df_mean.plot(figsize=(10,8))

plt.xlabel('Quality',fontsize=16)

plt.ylabel('Quantity',fontsize=16)
fig = plt.figure(figsize=(12, 10));

ax = fig.add_subplot(1,1,1)

bp = data.boxplot(grid=False,layout=(2,2),column=['fixed acidity','volatile acidity','citric acid','alcohol'], by=['quality'],ax=ax)

#Last 13 columns as predictors

X = data.iloc[ :, 1:14]

X.head(2)
#First column as target

y = data.iloc[ : , 0]

y.head()
y = y.map({'white':1, 'red' : 0})

y.dtype
#Divide dataset into Training data and validation data

X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.35,

                                                    shuffle = True

                                                    )
xgb_list = [('ss',ss()),         #Scaling parameters

            ('pca',PCA()),       #Instantiate PCA

            ('xgb',XGBClassifier(silent = False,  #Instantiate XGB classifier with 2 cpu threads 

                                  n_jobs=2))]
#Instantiate Pipeline object

xgb_pipeline = Pipeline(xgb_list)

#A parameter grid for grid search

parameter_gs = {'xgb__learning_rate':  [0.04, 0.07],

              'xgb__n_estimators':   [150,  200],

              'xgb__max_depth':      [3,5],

              'pca__n_components' : [7,12]

              }
grid_search = GridSearchCV(xgb_pipeline,            # XGB pipeline object

                   parameter_gs,         # 2*2*2*2 parameter grid

                   n_jobs = 2,         # No. of parallel cpu threads

                   cv =2 ,             # No. of folds

                   verbose =2,      

                   scoring = ['accuracy', 'roc_auc'],  # Performance metrics

                   refit = 'roc_auc'   # Refitting final model those which maximise auc

                   )
#Fitting data to Pipeline

start = time.time()

grid_search.fit(X_train, y_train)   

end = time.time()

(end - start)/60 
f"Best score: {grid_search.best_score_} "
f"Best parameter set {grid_search.best_params_}"
plt.bar(grid_search.best_params_.keys(), grid_search.best_params_.values(), color='g')

plt.xticks(rotation=10)
y_pred = grid_search.predict(X_test)

y_pred
accuracy = accuracy_score(y_test, y_pred)
f"Accuracy: {accuracy * 100.0}"
#A parameter set for random search

parameter_rs = {'xgb__learning_rate':  uniform(0, 1),

              'xgb__n_estimators':   range(60,120),

              'xgb__max_depth':      range(4,7),

              'pca__n_components' : range(5,7)}
random_search = RandomizedSearchCV(xgb_pipeline,

                        param_distributions=parameter_rs,

                        scoring= ['roc_auc', 'accuracy'],

                        n_iter=15,          # Max combination of

                        verbose = 3,

                        refit = 'roc_auc',

                        n_jobs = 2,          # No. of parallel cpu threads

                        cv = 2               # No of folds.

                        )
#Fitting data to Pipeline

start = time.time()

random_search.fit(X_train, y_train)

end = time.time()

(end - start)/60
f"Best score: {random_search.best_score_} "
f"Best parameter set: {random_search.best_params_} "
plt.bar(random_search.best_params_.keys(), random_search.best_params_.values(), color='g')

plt.xticks(rotation=10)
y_pred = random_search.predict(X_test)

y_pred
accuracy = accuracy_score(y_test, y_pred)

f"Accuracy: {accuracy * 100.0}"
parameter_bo = {

           'learning_rate':  (0, 1),            

           'n_estimators':   (60,120),         

           'max_depth':      (4,7),            

           'n_components' :  (5,7)

            }


def xg_eval(learning_rate,n_estimators, max_depth,n_components):

    #Make Pipeling for BO

    pipe_xg1 = make_pipeline (ss(),

                              PCA(n_components=int(round(n_components))),

                              XGBClassifier(

                                           silent = False,

                                           n_jobs=2,

                                           learning_rate=learning_rate,

                                           max_depth=int(round(max_depth)),

                                           n_estimators=int(round(n_estimators))

                                           )

                             )

    #Fitting into pipeline 

    cv_result = cross_val_score(estimator = pipe_xg1,

                                X= X_train,

                                y = y_train,

                                cv = 2,

                                n_jobs = 2,

                                scoring = 'f1'

                                ).mean()             # taking mean of all results



    return cv_result       #Returning final mean of all results of cross val score
bayesian_opt = BayesianOptimization(

                             xg_eval,     

                             parameter_bo   

                             )
start = time.time()
bayesian_opt.maximize(init_points=5,

               n_iter=15,        

               )
f"Best parameter set: {bayesian_opt.max} "
bayesian_opt.max.values()
for features in bayesian_opt.max.values(): 

    print(features)
features
plt.bar(features.keys(), features.values(), color='g')

plt.xticks(rotation=10)
#Model with parameters of grid search

model_gs = XGBClassifier(

                    learning_rate = grid_search.best_params_['xgb__learning_rate'],

                    max_depth = grid_search.best_params_['xgb__max_depth'],

                    n_estimators=grid_search.best_params_['xgb__n_estimators']

                    )



#Model with parameters of random search

model_rs = XGBClassifier(

                    learning_rate = random_search.best_params_['xgb__learning_rate'],

                    max_depth = random_search.best_params_['xgb__max_depth'],

                    n_estimators=random_search.best_params_['xgb__n_estimators']

                    )



#Model with parameters of bayesian optimization

model_bo = XGBClassifier(

                    learning_rate = int(features['learning_rate']),

                    max_depth = int(features['max_depth']),

                    n_estimators=int(features['n_estimators'])

                    )
start = time.time()

model_gs.fit(X_train, y_train)

model_rs.fit(X_train, y_train)

model_bo.fit(X_train, y_train)
y_pred_gs = model_gs.predict(X_test)

y_pred_rs = model_rs.predict(X_test)

y_pred_bo = model_bo.predict(X_test)
accuracy_gs = accuracy_score(y_test, y_pred_gs)

accuracy_rs = accuracy_score(y_test, y_pred_rs)

accuracy_bo = accuracy_score(y_test, y_pred_gs)
print("Grid search Accuracy: "+str(accuracy_gs))

print("Grid search Accuracy: "+str(accuracy_rs))

print("Bayesian Optimization Accuracy: "+str(accuracy_bo))
model_gs.feature_importances_

model_rs.feature_importances_

model_bo.feature_importances_

plot_importance(model_gs)

plot_importance(model_rs)

plot_importance(model_bo)

plt.show()