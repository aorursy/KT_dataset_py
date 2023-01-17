# To support both python 2 and python 3

from __future__ import division, print_function, unicode_literals



# 1.0 Clear ipython memory

%reset -f



# Common imports

import numpy as np

import pandas as pd

import os

import gc

import time

import datetime as dt



# to make this notebook's output stable across runs

#Somehow this is not happening as o/p of models is not consistent

np.random.seed(42)



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
# 1.1 Working with imbalanced data

# http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.SMOTE.html

# Check imblearn version number as:

#   import imblearn;  imblearn.__version__

from imblearn.over_sampling import SMOTE, ADASYN

# 1.2 Class for applying multiple data transformation jobs

from sklearn.compose import ColumnTransformer as ct

# Scale numeric data

from sklearn.preprocessing import StandardScaler as ss

# One hot encode data--Convert to dummy

from sklearn.preprocessing import OneHotEncoder as ohe

# 1.3 Dimensionality reduction

from sklearn.decomposition import PCA



# 1.4 Data splitting and model parameter search

from sklearn.model_selection import train_test_split

#from sklearn.model_selection import GridSearchCV

#from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression





# 1.5 Modeling modules

# conda install -c anaconda py-xgboost

#from xgboost.sklearn import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout







# 1.6 Model pipelining

#from sklearn.pipeline import Pipeline

#from sklearn.pipeline import make_pipeline





# 1.7 Model evaluation metrics

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve



# 1.8

import matplotlib.pyplot as plt

#from xgboost import plot_importance



# 1.9 Needed for Bayes optimization

#from sklearn.model_selection import cross_val_score



# 1.10 Install as: pip install bayesian-optimization

#     Refer: https://github.com/fmfn/BayesianOptimization

#from bayes_opt import BayesianOptimization





# 1.11 Find feature importance of ANY BLACK BOX estimator

#      See note at the end of this code for explanation

#      Refer: https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html

#      Install as:

#      conda install -c conda-forge eli5

#import eli5

#from eli5.sklearn import PermutationImportance

# 1.12 Model building

#     Install h2o as: conda install -c h2oai h2o=3.22.1.2

import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator
transactions = pd.read_csv("../input/online-shopping-fraud/datasetForFinalAssignment.csv")

transactions.info()
validate=pd.read_csv("../input/online-fraud-sample/datasetForFinalTest.csv")

validate.info()
#columns = transactions.columns

#columns

transactions.head()
"""

In summary, we may utilize the following variable going forward into statistical analysis and model building:

numberOfTimesDeviceUsed, 

timeBetween, 

sex, 

browser, 

Source to arrive at estimated values of target variable 'class'."

"""

X = transactions.copy() # independent variables data

y = X['class']

X.info()
X.drop(columns = ['class'], inplace = True)

X.drop(columns = ['Column 1','user_id','device_id','ip_address','signup_time','purchase_time'], inplace = True)

X.info()
X_val = validate.copy()
X_val.drop(columns = ['Column 1','user_id','device_id','ip_address','signup_time','purchase_time'], inplace = True)

X_val.info()

X.head()
#Define the transformation function using columnTransformer, OHE and StandardScaler

def transform(categorical_columns,numerical_columns,df):

    #  Create a tuple of processing tasks:

    #  (taskName, objectToPerformTask, columns-upon-which-to-perform)

    # 9.1 One hot encode categorical columns

    cat = ('categorical', ohe() , categorical_columns  )

    # 9.2 Scale numerical columns

    num = ('numeric', ss(), numerical_columns)

    # 9.3 Instantiate columnTransformer object to perform task

    #     It transforms X separately by each transformer

    #     and then concatenates results.

    col_trans = ct([cat, num])

    # 9.4 Learn data

    col_trans.fit(df)

    # 9.5 Now transform df

    df_transAndScaled = col_trans.transform(df)

    # 9.6 Return transformed data and also transformation object

    return df_transAndScaled, col_trans

#Define the columns for transformations

categorical_columns = ['source', 'browser','sex']

numerical_columns = ['signup_time-purchase_time', 'purchase_value','age','N[device_id]']
X.source.unique() # ['Direct', 'SEO', 'Ads'] 3 values --> columns 0,1,2
X.browser.unique() # ['Chrome', 'FireFox', 'IE', 'Safari', 'Opera'] 5 values  --> columns 3,4,5,6,7
X.sex.unique() # ['M', 'F'] 2 values --> columns 8,9
#Define the columns for post transformation dataframe - makes referencing and understanding easier

columns = ['source_Direct','source_SEO','source_Ads','browser_Chrome','browser_FireFox','browser_IE','browser_Safari','browser_Opera','sex_M','sex_F'] + numerical_columns

columns
# 10.0 Transform X dataset

X_transAndScaled, _  = transform(categorical_columns, numerical_columns, X)



# 10.1

X_transAndScaled.shape # (74691, 14)
X_transAndScaled = pd.DataFrame(X_transAndScaled, index=X.index, columns=columns)

X_transAndScaled.head()
# 11.0 Transform X_val dataset

Xval_transAndScaled, _  = transform(categorical_columns, numerical_columns, X_val)



# 10.1

Xval_transAndScaled.shape           # (13413, 14)
#Provide the column names for additional dummy variables

Xval_transAndScaled = pd.DataFrame(Xval_transAndScaled, index=X_val.index, columns=columns)

Xval_transAndScaled.head()
# 12. Split data into train/test

#     train-test split. save the indices of split set

X_train,X_test, y_train, y_test ,indicies_tr,indicies_test = train_test_split(

                                                                      X_transAndScaled,    # Predictors

                                                                      y,                # Target

                                                                      np.arange(X_transAndScaled.shape[0]),

                                                                      test_size = 0.3,   # split-ratio

                                                                      random_state=1

                                                                     )
X_train.shape
#Generate the image of test dataset pre-split using indicies_test.

#This will be used to capture the unscaled values of purchase_value for computing cost of model

X_cost = X.iloc[indicies_test]

X_cost.purchase_value.head()
#Using Garbage Collect to clear memory

del X_transAndScaled

del indicies_tr

del indicies_test

gc.collect()
type(y_train)

#y_train.info()
#12.0 Checking the extent of 'class' imbalance

np.sum(y_train)/len(y_train)       # 0.09431363923263776
# 12.1  Process X_train data with SMOTE

sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_sample(X_train, y_train)

type(X_res)       # No longer pandas dataframe

                  #  but we will convert to pandas dataframe



# 12.2 Check

X_res.shape                    # (94756, 14)
#Using y_res for most of the classifiers

#Using y_onehot for NeuralNetworks

y_onehot = pd.get_dummies(y_res)

y_onehot.info()
np.sum(y_res)/len(y_res)       # 0.5 ,earlier ratio was 0.09381634565728822
#y_res = y_res.reshape(len(y_res),1)

y_res = pd.DataFrame(y_res)

type(y_res)
X_res = pd.DataFrame(X_res,columns=columns)

type(X_res)
def modelCost(test_y,model_y,df):

    #falsePositive: Cost is $8*count

    #non-fraudulent transactions (test_y '0') predicted as fraudulent by model (model_y '1')

    falsePositiveCost = df.purchase_value[(test_y==0) & (model_y==1)].count()*8

    print("falsePositive {:.0f}".format(df.purchase_value[(test_y==0) & (model_y==1)].count()))

    print("falsePositiveCost ${:.0f}".format(falsePositiveCost))

    #falseNegative: Cost is sum of purchase_value

    #fraudulent transactions (test_y '1') predicted as non-fraudulent by model (model_y '0')

    falseNegativeCost = df.purchase_value[(test_y==1) & (model_y==0)].sum()

    print("falseNegative {:.0f}".format(df.purchase_value[(test_y==1) & (model_y==0)].count()))

    print("falseNegativeCost ${:.0f}".format(falseNegativeCost))

    totalCost = falsePositiveCost + falseNegativeCost

    print("totalCost ${:.0f}".format(totalCost))

    return totalCost

y0s=np.zeros(y_test.size)

totalCost_0s = modelCost(y_test, y0s, X_cost)
y1s=np.ones(y_test.size)

totalCost_1s = modelCost(y_test, y1s, X_cost)
#Running basic regression first to setup all checking and evaluation functions

log_reg = LogisticRegression(random_state=42)

start = time.time()

log_reg.fit(X_res, y_res)

end = time.time()

(end - start) #0.31 seconds
y_logreg = log_reg.predict(X_test)

log_reg.score(X_test, y_test) #0.920653338093538 Not bad for a start!
totalCost_logreg = modelCost(y_test, y_logreg, X_cost)

totalCost_logreg # $35238
########## Logistic Regression With L1 Penalty ##########

# logistic regression with L1 penalty

start = time.time()

logreg_l1 = LogisticRegression(C=0.1, penalty='l1')

logreg_l1.fit(X_res, y_res)

end = time.time()

print("LOGISTIC REGRESSION - L1 penalty took {:.2f}s".format(end - start)) #0.57s
logreg_l1.coef_

y_logregl1 = logreg_l1.predict(X_test)

logreg_l1.score(X_test, y_test) #0.9206087111745805 not much improvement
totalCost_logregl1 = modelCost(y_test, y_logregl1, X_cost)

totalCost_logregl1 # $35262 -- Cost degrades!
########## Logistic Regression With L2 Penalty ##########

# logistic regression with L2 penalty

start = time.time()

logreg_l2 = LogisticRegression(C=0.1, penalty='l2')

logreg_l2.fit(X_res, y_res)

end = time.time()

print("LOGISTIC REGRESSION - L2 penalty took {:.2f}s".format(end - start)) #0.46s
logreg_l2.coef_

y_logregl2 = logreg_l2.predict(X_test)

logreg_l2.score(X_test, y_test) #0.920653338093538 not much improvement
totalCost_logregl2 = modelCost(y_test, y_logregl2, X_cost)

totalCost_logregl2 # $35291 -- Cost degrades significantly!
########## DecisionTreeClassifier ##########

#from sklearn.tree import DecisionTreeClassifier

start = time.time()

treeclf = DecisionTreeClassifier(max_depth=3, random_state=1)

treeclf.fit(X_res, y_res)

end = time.time()

print("DecisionTreeClassifier took {:.2f}s".format(end - start)) #0.27s
y_treeclf = treeclf.predict(X_test)

treeclf.score(X_test, y_test) #0.9201178150660478
totalCost_treeclf = modelCost(y_test, y_treeclf, X_cost)

totalCost_treeclf # $34890 -- Cost improves slightly from $35238 for logisticRegression!
########## RandomForestClassifier ##########

#from sklearn.ensemble import RandomForestClassifier

start = time.time()

#rfclf = RandomForestClassifier(n_estimators=200, max_features=5, oob_score=True, random_state=1)

rfclf = RandomForestClassifier(n_estimators=200, max_features=5, random_state=1)

rfclf.fit(X_res, y_res)

end = time.time()

print("DecisionTreeClassifier took {:.2f}s".format(end - start)) #44.71s
y_rfclf = rfclf.predict(X_test)

rfclf.score(X_test, y_test) #0.9364512674044984
totalCost_rfclf = modelCost(y_test, y_rfclf, X_cost)

totalCost_rfclf # $38259 -- Cost degrades majorly from $34890 for DecisionTreeClassifier!
X_res.info()
y_onehot.info()
start = time.time()

NN_2l = Sequential()

NN_2l.add(Dense(input_dim=14, output_dim=100))

NN_2l.add(Dense(output_dim=2))

NN_2l.add(Activation("softmax"))



NN_2l.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])



NN_2l.fit(X_res, y_onehot)



end = time.time()

print("2layer NeuralNetwork took {:.2f}s".format(end - start)) #6.9s

y_NN2l = NN_2l.predict_classes(X_test)

print("\n\naccuracy", np.sum(y_NN2l == y_test) / float(len(y_test)))
totalCost_NN2l = modelCost(y_test, y_NN2l, X_cost)

totalCost_NN2l # $33575 -- Cost improvement from $34890 for DecisionTreeClassifier!
start = time.time()

NN_3l = Sequential()

NN_3l.add(Dense(input_dim=14, output_dim=100))

NN_3l.add(Dense(output_dim=100))

NN_3l.add(Dense(output_dim=2))

NN_3l.add(Activation("softmax"))



NN_3l.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])



NN_3l.fit(X_res, y_onehot)



end = time.time()

print("3layer NeuralNetwork took {:.2f}s".format(end - start)) #8.4s

y_NN3l = NN_3l.predict_classes(X_test)

print("\n\naccuracy", np.sum(y_NN3l == y_test) / float(len(y_test)))
totalCost_NN3l = modelCost(y_test, y_NN3l, X_cost)

totalCost_NN3l # $33501 -- Cost improvement from $33575 for 2layer NN!
#from keras.layers import Dense, Activation, Dropout
start = time.time()

NN_ReLu = Sequential()

NN_ReLu.add(Dense(100, input_shape=(14,)))

NN_ReLu.add(Activation('relu'))

NN_ReLu.add(Dropout(0.2))

NN_ReLu.add(Dense(100))

NN_ReLu.add(Activation('relu'))

NN_ReLu.add(Dropout(0.2))

NN_ReLu.add(Dense(2))

NN_ReLu.add(Activation("softmax"))



NN_ReLu.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])



NN_ReLu.fit(X_res, y_onehot)



end = time.time()

print("ReLu NeuralNetwork took {:.2f}s".format(end - start)) #10.56s

y_NNReLu = NN_ReLu.predict_classes(X_test)

print("\n\naccuracy", np.sum(y_NNReLu == y_test) / float(len(y_test)))
totalCost_NNReLu = modelCost(y_test, y_NNReLu, X_cost)

totalCost_NNReLu # $35503 -- Cost degrades from $33501 for 3layer NN!
# 13.0 Preparing to model data with deeplearning

#      H2o requires composite data with both predictors

#      and target

df = np.hstack((X_res,y_res))

df.shape            #





# 13.1 Start h2o

h2o.init()



# 13.2 Transform data to h2o dataframe

df = h2o.H2OFrame(df, column_names=columns+['class'])

#df = h2o.H2OFrame(df)

df.columns

len(df.columns)    # 15

df.shape           # (94842, 15)
len(columns) #14 class is extra in df
# 14. Get list of predictor column names and target column names

#     Column names are given by H2O when we converted array to

#     H2o dataframe

X_columns = df.columns[0:14]        # Only column names. No data

X_columns       # C1 to C18

y_columns = df.columns[14]

y_columns



# 14.1 For classification, target column must be factor

#      Required by h2o

df['class'] = df['class'].asfactor()



# 15. Build a deeplearning model on balanced data

#     http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html

dl_model = H2ODeepLearningEstimator(epochs=1000,

                                    distribution = 'bernoulli',                 # Response has two levels

                                    missing_values_handling = "MeanImputation", # Not needed by us

                                    variable_importances=True,

                                    nfolds = 2,                           # CV folds

                                    fold_assignment = "Stratified",       # Each fold must be sampled carefully

                                    keep_cross_validation_predictions = True,  # For analysis

                                    balance_classes=False,                # SMOTE is not provided by h2o

                                    standardize = True,                   # z-score standardization

                                    activation = 'RectifierWithDropout',  # Default dropout is 0.5

                                    hidden = [100,100],                  # ## more hidden layers -> more complex interactions

                                    stopping_metric = 'logloss',

                                    loss = 'CrossEntropy')



# 16.1 Train our model

start = time.time()

dl_model.train(X_columns,

               y_columns,

               training_frame = df)





end = time.time()

(end - start)/60



# 16.2 Get model summary

print(dl_model)

X_test_h2o = h2o.H2OFrame(X_test)

type(X_test_h2o)
y_h20DL = dl_model.predict(X_test_h2o[: , 0:13])

y_h20DL[:, 0]
#y_h20predict = y_h20DL['predict'].as_data_frame().as_matrix()

y_h20predict = y_h20DL['predict'].as_data_frame().values

type(y_h20predict)

#y_h20predict0 = y_h20predict[:,0]
print("\n\naccuracy", np.sum(y_h20predict[:,0] == y_test) / float(len(y_test)))
totalCost_h20predict = modelCost(y_test, y_h20predict[:,0], X_cost)

totalCost_h20predict # $35503 -- Cost degrades from $33501 for 3layer NN!
'''

Need a way to include totalCost in the optimization objective. 

without it modelling results have an elemnt of chance as 

any small increase in falseNegative disproportionately increases 

the costs without the model being aware of the magnitude of cost error!!



For this reason, I am comparing the results obtained on test dataset after running all 9 models.

Models are evaluated basis how well the falseNegative are addressed.



A comparison of all models tried here, sorted by the falseNegative ascendingly:

'''
Xval_transAndScaled.head()
""" PLaceholder code to try out Bayesian Optimization

para_set = {

           'learning_rate':  (0, 1),                 # any value between 0 and 1

           'n_estimators':   (50,300),               # any number between 50 to 300

           'max_depth':      (3,10),                 # any depth between 3 to 10

           'n_components' :  (20,30)                 # any number between 20 to 30

            }



def xg_eval(learning_rate,n_estimators, max_depth,n_components):

    # 12.1 Make pipeline. Pass parameters directly here

    pipe_xg1 = make_pipeline (ss(),                        # Why repeat this here for each evaluation?

                              PCA(n_components=int(round(n_components))),

                              XGBClassifier(

                                           silent = False,

                                           n_jobs=2,

                                           learning_rate=learning_rate,

                                           max_depth=int(round(max_depth)),

                                           n_estimators=int(round(n_estimators))

                                           )

                             )



    # 12.2 Now fit the pipeline and evaluate

    cv_result = cross_val_score(estimator = pipe_xg1,

                                X= X_train,

                                y = y_train,

                                cv = 2,

                                n_jobs = 2,

                                scoring = 'f1'

                                ).mean()             # take the average of all results





    # 12.3 Finally return maximum/average value of result

    return cv_result

    

xgBO = BayesianOptimization(

                             xg_eval,     # Function to evaluate performance.

                             para_set     # Parameter set from where parameters will be selected

                             )



gp_params = {"alpha": 1e-5}      # Initialization parameter for gaussian



start = time.time()

xgBO.maximize(init_points=5,    # Number of randomly chosen points to

                                 # sample the target function before

                                 #  fitting the gaussian Process (gp)

                                 #  or gaussian graph

               n_iter=25,        # Total number of times the

               #acq="ucb",       # ucb: upper confidence bound

                                 #   process is to be repeated

                                 # ei: Expected improvement

               # kappa = 1.0     # kappa=1 : prefer exploitation; kappa=10, prefer exploration

              **gp_params

               )

end = time.time()



xgBO.res



type(xgBO.res) #if list the following line will not work

#xgBO.res['max']

xgBO.max # using the function directly to get the parameters

"""
""" Plceholder code to try out pipeline



clf = Pipeline([

        ("kpca", KernelPCA(n_components=2)),

        ("log_reg", LogisticRegression())

    ])



param_grid = [{

        "kpca__gamma": np.linspace(0.03, 0.05, 10),

        "kpca__kernel": ["rbf", "sigmoid"]

    }]



grid_search = GridSearchCV(clf, param_grid, cv=3)

grid_search.fit(X, y)

print(grid_search.best_params_)

"""
