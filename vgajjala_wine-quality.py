# Data manipulation

import numpy as np

import pandas as pd



# Data pre-processing

from sklearn.preprocessing import StandardScaler as ss



# Dimensionality reduction

from sklearn.decomposition import PCA



#  Data splitting and model parameter search

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



# Modeling modules

from xgboost.sklearn import XGBClassifier



# Model pipelining

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline



# Model evaluation metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score, f1_score



# Plotting

import matplotlib.pyplot as plt

import seaborn as sns

from xgboost import plot_importance



# For Bayes optimization

from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization



# Finding feature importance of ANY BLACK BOX estimator

import eli5

from eli5.sklearn import PermutationImportance



# Misc

import time

import os

import gc

import random

from scipy.stats import uniform
# Set option to dislay many rows

pd.set_option('display.max_columns', 100)
data = pd.read_csv("../input/winequalityN.csv")
data.shape
data.columns.values
data.dtypes.value_counts()
data.head(3)
data.describe()
data.type.value_counts()
data.isnull().sum()
data = data.dropna()
data.isnull().sum()
data.shape
data.describe()
plt.figure(figsize=(10, 10))

sns.heatmap(data.corr(),annot = True, linewidths = 0.8, cmap = 'PuOr')
# Plotting acidic factors

sns.pairplot(data, vars = ['fixed acidity', 'volatile acidity', 'citric acid', 'pH',], hue='type', height = 5, palette="prism")
# Plotting chemical factors

sns.pairplot(data, vars = ['chlorides', 'sulphates', 'free sulfur dioxide', 'total sulfur dioxide',], hue='type', height = 5, palette="prism")
# Checking relation between quality, alcohol and type

plt.figure(figsize=(10, 10))

sns.boxplot(x='quality', y = 'alcohol', hue = 'type' , data = data)
data.type.value_counts()  
plt.figure(figsize=(10, 10))

sns.countplot(x = 'quality', data=data, hue='type')
X = data.iloc[ :, 1:13]                       
X.head(2) 
y = data.iloc[ : , 0]
y.head()
y = y.map({'white':1, 'red' : 0})
y.head()
y.dtype
colnames = X.columns.tolist()
# Split dataset into train and validation parts

X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.35,

                                                    shuffle = True

                                                    )
data.shape
X_train.shape
X_test.shape
y_train.shape
y_test.shape 
#  Making Pipeline using xgboost

steps_xg = [('sts', ss() ),

            ('pca', PCA()),

            ('xg',  XGBClassifier(silent = False,

                                  n_jobs=2)       

            )

            ]
# Instantiate Pipeline object

pipe_xg = Pipeline(steps_xg)
#  Specify xgboost parameter-range                        

parameters = {'xg__learning_rate':  [0, 1], 

              'xg__n_estimators':   [50,  100],  

              'xg__max_depth':      [3,5],

              'pca__n_components' : [5,7]

              } 
##################### Grid Search #################

clf = GridSearchCV(pipe_xg,            

                   parameters,       

                   n_jobs = 2,        

                   cv =2 ,           

                   verbose =2,         

                   scoring = ['accuracy', 'roc_auc'], 

                   refit = 'roc_auc' 

                   )
# Start fitting data to pipeline

start = time.time()

clf.fit(X_train, y_train)

end = time.time()

(end - start)/60 
# Evaluate

f"Best score: {clf.best_score_} "
f"Best parameter set {clf.best_params_}"
#  Find feature importance of any BLACK Box model

# Instantiate the importance object

perm_gs = PermutationImportance(

                            clf,

                            random_state=1

                            )
# fit data & learn

start = time.time()

perm_gs.fit(X_test, y_test)

end = time.time()

(end - start)/60
# Get feature weights

eli5.show_weights(

                  perm_gs,

                  feature_names = colnames

                  )
fw_gs = eli5.explain_weights_df(

                  perm_gs,

                  feature_names = colnames    

                  )
# Print importance

fw_gs
##################### Randomized Search #################



# Tune parameters using randomized search

#  Hyperparameters to tune and their ranges

parameters = {'xg__learning_rate':  uniform(0, 1),

              'xg__n_estimators':   range(50,100),

              'xg__max_depth':      range(3,5),

              'pca__n_components' : range(5,7)}
#  Tune parameters using random search

#     Create the object first

rs = RandomizedSearchCV(pipe_xg,

                        param_distributions=parameters,

                        scoring= ['roc_auc', 'accuracy'],

                        n_iter=10,          

                        verbose = 3,

                        refit = 'roc_auc',

                        n_jobs = 2,          

                        cv = 2               

                        )
# Fitting data

start = time.time()

rs.fit(X_train, y_train)

end = time.time()

(end - start)/60
# Evaluate

f"Best score: {rs.best_score_} "
f"Best parameter set: {rs.best_params_} "
#  Instantiate the importance object

perm_rs = PermutationImportance(

                            rs,

                            random_state=1

                            )
# fit data & learn

start = time.time()

perm_rs.fit(X_test, y_test)

end = time.time()

(end - start)/60
# Get feature weights

eli5.show_weights(

                  perm_rs,

                  feature_names = colnames      

                  )
fw_rs = eli5.explain_weights_df(

                  perm_rs,

                  feature_names = colnames     

                  )
# Print importance

fw_rs 
##################### Bayesian Optimization #################



# Which parameters to consider and what is each one's range

para_set = {

           'learning_rate':  (0, 1),                 

           'n_estimators':   (50,100),               

           'max_depth':      (3,5),                 

           'n_components' :  (5,7)                 

            }
#  Create a function that when passed some parameters

#    evaluates results using cross-validation

#    This function is used by BayesianOptimization() object

def xg_eval(learning_rate,n_estimators, max_depth,n_components):



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



    # fit the pipeline and evaluate

    cv_result = cross_val_score(estimator = pipe_xg1,

                                X= X_train,

                                y = y_train,

                                cv = 5,

                                n_jobs = 2,

                                scoring = 'f1'

                                ).mean()             





    #  return maximum/average value of result

    return cv_result
#      Instantiate BayesianOptimization() object

#      This object  can be considered as performing an internal-loop

#      i)  Given parameters, xg_eval() evaluates performance

#      ii) Based on the performance, set of parameters are selected

#          from para_set and fed back to xg_eval()

#      (i) and (ii) are repeated for given number of iterations

#

xgBO = BayesianOptimization(

                             xg_eval,     

                             para_set

                             )
#  Gaussian process parameters

#     Modulate intelligence of Bayesian Optimization process

#     This parameters controls how much noise the GP can handle,

#     so increase it whenever you think that extra flexibility is needed.

gp_params = {"alpha": 1e-5}      # Initialization parameter for gaussian Process
#  Fit/train (so-to-say) the BayesianOptimization() object

#     Start optimization. 25minutes

#     Our objective is to maximize performance (results)

start = time.time()

xgBO.maximize(init_points=5,    # Number of randomly chosen points to

                                 # sample the target function before

                                 #  fitting the gaussian Process (gp)

                                 #  or gaussian graph

               n_iter=20,        # Total number of times the

               #acq="ucb",       # ucb: upper confidence bound

                                 #   process is to be repeated

                                 # ei: Expected improvement

               # kappa = 1.0     # kappa=1 : prefer exploitation; kappa=10, prefer exploration

              **gp_params

               )

end = time.time()

(end-start)/60
###############  Fitting Best parameters in our model ##############

###############    Model Importance   #################

#  Model with parameters of grid search

model_gs = XGBClassifier(

                    learning_rate = clf.best_params_['xg__learning_rate'],

                    max_depth = clf.best_params_['xg__max_depth'],

                    n_estimators=clf.best_params_['xg__max_depth']

                    )
# Model with parameters of random search

model_rs = XGBClassifier(

                    learning_rate = rs.best_params_['xg__learning_rate'],

                    max_depth = rs.best_params_['xg__max_depth'],

                    n_estimators=rs.best_params_['xg__max_depth']

                    )
model_bo = XGBClassifier(

                    learning_rate = xgBO.max['params']['learning_rate'],

                    max_depth = int(xgBO.max['params']['max_depth']),

                    n_estimators= int(xgBO.max['params']['n_estimators'])

                    )
#  Modeling with all parameters

start = time.time()

model_gs.fit(X_train, y_train)

model_rs.fit(X_train, y_train)

model_bo.fit(X_train, y_train)

end = time.time()

(end - start)/60
#  Predictions with all models

y_pred_gs = model_gs.predict(X_test)

y_pred_rs = model_rs.predict(X_test)

y_pred_bo = model_bo.predict(X_test)
#Confusion Matrix for all models

print("Confusion Matrix for GS is \n",confusion_matrix(y_test,y_pred_gs))

print("Confusion Matrix for RS is \n",confusion_matrix(y_test,y_pred_rs))

print("Confusion Matrix for BO is \n",confusion_matrix(y_test,y_pred_bo))
# Accuracy from all models

accuracy_gs = accuracy_score(y_test, y_pred_gs)

accuracy_rs = accuracy_score(y_test, y_pred_rs)

accuracy_bo = accuracy_score(y_test, y_pred_bo)
# Calculating Precision/Recall/F-score

precision_gs, precision_rs, precision_bo = precision_score(y_test,y_pred_gs), precision_score(y_test,y_pred_rs), precision_score(y_test,y_pred_bo)

recall_gs, recall_rs, recall_bo = recall_score(y_test,y_pred_gs), recall_score(y_test,y_pred_rs), recall_score(y_test,y_pred_bo)

f1_score_gs, f1_score_rs, f1_score_bo = f1_score(y_test,y_pred_gs), f1_score(y_test,y_pred_rs), f1_score(y_test,y_pred_bo)
#  Get feature importances from all 3 models

model_gs.feature_importances_

model_rs.feature_importances_

model_bo.feature_importances_

plot_importance(model_gs)

plot_importance(model_rs)

plot_importance(model_bo)

plt.show()
# Get probability of occurrence of each class

y_pred_prob_gs = model_gs.predict_proba(X_test)

y_pred_prob_rs = model_rs.predict_proba(X_test)

y_pred_prob_bo = model_bo.predict_proba(X_test)
# Draw ROC curve

fpr_gs, tpr_gs, thresholds = roc_curve(y_test,

                                 y_pred_prob_gs[: , 0],

                                 pos_label= 0

                                 )



fpr_rs, tpr_rs, thresholds = roc_curve(y_test,

                                 y_pred_prob_rs[: , 0],

                                 pos_label= 0

                                 )



fpr_bo, tpr_bo, thresholds = roc_curve(y_test,

                                 y_pred_prob_bo[: , 0],

                                 pos_label= 0

                                 )
# AUC

auc_gs = auc(fpr_gs,tpr_gs)

auc_rs = auc(fpr_rs,tpr_rs)

auc_bo = auc(fpr_bo,tpr_bo)
# Plot ROC Curve

fig = plt.figure()  

ax = fig.add_subplot(111)  



#Connect diagonals

ax.plot([0, 1], [0, 1], ls="--")



# Labels 

ax.set_xlabel('False Positive Rate') 

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for models')



# Set graph limits

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])



# Plot each graph now

ax.plot(fpr_gs, tpr_gs, label = "Grid Search")

ax.plot(fpr_rs, tpr_rs, label = "Random Search")

ax.plot(fpr_bo, tpr_bo, label = "Bayesian Optimization")



# Set legend and show plot

ax.legend(loc="lower right")

plt.show()
f"For GS, Accuracy = {round(accuracy_gs,5)}, Precison = {round(precision_gs,5)}, Recall = {round(recall_gs,5)}, f1_score = {round(f1_score_gs,5)}, AUC = {round(auc_gs,5)}"
f"For RS, Accuracy = {round(accuracy_rs,5)}, Precison = {round(precision_rs,5)}, Recall = {round(recall_rs,5)}, f1_score = {round(f1_score_rs,5)}, AUC = {round(auc_rs,5)}"
f"For BO, Accuracy = {round(accuracy_bo,5)}, Precison = {round(precision_bo,5)}, Recall = {round(recall_bo,5)}, f1_score = {round(f1_score_bo,5)}, AUC = {round(auc_bo,5)}"