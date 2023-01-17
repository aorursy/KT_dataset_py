import os
import gc
import time
import numpy as np
import pandas as pd
import glob
import io
import math
import matplotlib


from sklearn.cross_validation import train_test_split
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import seaborn as sns
print(os.listdir("../input"))
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.

path = '../input/'
train_df = pd.read_csv(path+"/train.csv")
test_df = pd.read_csv(path+"/test.csv")
#What size data are we dealing with?
print("Training data shape","\n")
print(train_df.shape,"\n")
print("Testing data shape","\n")
print(test_df.shape,"\n")

#What columns / features do we have?
print("Training columns","\n")
print(train_df.columns,"\n")
print("Testing columns","\n")
print(test_df.columns,"\n")

#What type of data do we have?
#print("Train data types","\n")
#print(train_df.dtypes,"\n")
#print("Test data types","\n")
#print(test_df.dtypes)
#Pull out Targets for later
Targets_df=pd.DataFrame()
Targets_df["bandgap_energy_ev"]=train_df["bandgap_energy_ev"].copy()
Targets_df["formation_energy_ev_natom"]=train_df["formation_energy_ev_natom"].copy()
train_df=train_df.drop(["formation_energy_ev_natom","bandgap_energy_ev"],axis=1)

#Pull out IDs for later, since don't need in the models
train_id_df=pd.DataFrame()
train_id_df["id"]=train_df["id"].copy()
train_df=train_df.drop(["id"],axis=1)
test_id_df=pd.DataFrame()
test_id_df["id"]=test_df["id"].copy()
test_df=test_df.drop(["id"],axis=1)

#Combine both sets of data, so it's possible to do the same transformations on both
combined_df = pd.concat([train_df, test_df], ignore_index=True)
#Count number of NAs
print("Total number of null values in the df","\n")
print(combined_df.isna().sum().sum())
numerical_df=pd.DataFrame.copy(combined_df[['number_of_total_atoms', 'percent_atom_al',
       'percent_atom_ga', 'percent_atom_in', 'lattice_vector_1_ang',
       'lattice_vector_2_ang', 'lattice_vector_3_ang',
       'lattice_angle_alpha_degree', 'lattice_angle_beta_degree',
       'lattice_angle_gamma_degree']])

one_hot_df=pd.DataFrame.copy(combined_df[["spacegroup"]])

one_hot_df=pd.get_dummies(one_hot_df,prefix=["spacegroup"],
                       columns=["spacegroup"])

features_df=pd.concat([numerical_df,one_hot_df],axis=1)

#Manual unskewing / normalizing / standardizing
#Needed for linear methods etc, but don't need to worry about with XGB.

#print("Original skew","\n")
#print(numerical_df.skew())
#Get names of features which I'll class as skewed and unskewed(at least wrt right skewed)
skewed_feats= numerical_df.skew()
skewed_feats = skewed_feats[skewed_feats > 0.1]
skewed_feats = skewed_feats.index

unskewed_feats= numerical_df.skew()
unskewed_feats = unskewed_feats[unskewed_feats < 0.1]
unskewed_feats = unskewed_feats.index

#The X is an arbitrary cut-off & can be fine-tuned to get the best result

#Linearize the unskewed features & log transform the skewed features.
transform_df=pd.DataFrame()
transform_df[unskewed_feats]=(numerical_df[unskewed_feats]
                               - numerical_df[unskewed_feats].mean()) / (numerical_df[unskewed_feats].max() - numerical_df[unskewed_feats].min())
transform_df[skewed_feats] = np.log1p(numerical_df[skewed_feats])

#Check this worked.
#print("Transformed skew","\n")
#print(transform_df.skew())
#_ = transform_df.hist(bins=20, figsize=(18, 18), xlabelsize=10)

features_transform_df=pd.concat([transform_df,one_hot_df],axis=1)
features_df=pd.concat([numerical_df,one_hot_df],axis=1)
#Split back
training_examples=features_df.iloc[0:2400].copy()
test_examples=features_df.iloc[2400:3000].copy()
training_examples_transform=features_transform_df.iloc[0:2400].copy()
test_examples_transform=features_transform_df.iloc[2400:3000].copy()
#rmsle_cv measure
def rmsle_cv(model):
    rmsle= np.sqrt(-cross_val_score(model, training_examples_transform, training_targets, scoring="neg_mean_squared_log_error", cv = 5))
    return(rmsle)
#Linear regression model
training_targets=Targets_df["bandgap_energy_ev"].copy()
model_linear=LinearRegression().fit(training_examples_transform, training_targets)
linear_BG_pred = model_linear.predict(test_examples_transform)
BG_rmsle=rmsle_cv(model_linear).mean()
print("Band gap RMSLE:")
print(BG_rmsle,"\n")
#Get 0.1217

#Linear regression model
training_targets=Targets_df["formation_energy_ev_natom"].copy()
model_linear=LinearRegression().fit(training_examples_transform, training_targets)
linear_EF_pred = model_linear.predict(test_examples_transform)
EF_rmsle=rmsle_cv(model_linear).mean()
print("Formation Energy RMSLE:")
print(EF_rmsle,"\n")
# Get 0.0491 with the >0.1 skew transform

print("Expected combined RMSLE")
combined_rmsle=(EF_rmsle+BG_rmsle)/2
print(combined_rmsle)
#Expect 0.0854 for just linear regression

Predictions_df=pd.DataFrame()
Predictions_df["id"]=test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"]=linear_EF_pred
Predictions_df["bandgap_energy_ev"]=linear_BG_pred
Predictions_df.to_csv("Linear_Nomad.csv",index=False)
#XGB model for BG
training_targets=np.log1p(Targets_df["bandgap_energy_ev"].copy())
dtrain = xgb.DMatrix(training_examples, label = training_targets)
dtest = xgb.DMatrix(test_examples)
#tune wrt params
params = {"max_depth":2,
          "eta":0.1,
         'gamma':0,  
         'subsample':0.8,
         'colsample_bytree':1,
         'min_child_weight':10,
         'reg_alpha':7e-3,
         'reg_lambda':1,
        }
#for i in range(1,11):
#    params["max_depth"]=i
#    for j in range(1,11):
#        params["min_child_weight"]=j
#        model_xgb = xgb.cv(params, dtrain,  num_boost_round=500,early_stopping_rounds=100)
#        model_xgb.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
#        print("max_depth:")
#        print(params["max_depth"])
#        print("min_child_weight:")
#        print(params["min_child_weight"])
#        last=len(model_xgb.loc[:])-1
#        print(model_xgb.loc[last:,["test-rmse-mean"]])
#for i in np.arange(0,1.1,0.1):
#    params["gamma"]=i
#    model_xgb = xgb.cv(params, dtrain,  num_boost_round=500,early_stopping_rounds=100)
#    model_xgb.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
#    print("gamma:")
#    print(params["gamma"])
#    last=len(model_xgb.loc[:])-1
#   print(model_xgb.loc[last:,["test-rmse-mean"]])
#for i in np.arange(0.1,1.1,0.1):
#    params["subsample"]=i
#    for j in np.arange(0.1,1.1,0.1):
#        params["colsample_bytree"]=j
#        model_xgb = xgb.cv(params, dtrain,  num_boost_round=500,early_stopping_rounds=100)
#        model_xgb.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
#        print("subsample:")
#        print(params["subsample"])
#        print("colsample_bytree:")
#        print(params["colsample_bytree"])
#        last=len(model_xgb.loc[:])-1
#        print(model_xgb.loc[last:,["test-rmse-mean"]])

#for i in (1,0.9,0.8,1.1,1.2,1.5):
#    params["reg_lambda"]=i
#    model_xgb = xgb.cv(params, dtrain,  num_boost_round=1000,early_stopping_rounds=100)
#    print("reg_lambda:")
#    print(params["reg_lambda"])
#    last=len(model_xgb.loc[:])-1
#    print(model_xgb.loc[last:,["test-rmse-mean"]])
    
#for i in (0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01):
#    params["eta"]=i
#    model_xgb = xgb.cv(params, dtrain,  num_boost_round=1000,early_stopping_rounds=100)
#   print("eta:")
#    print(params["eta"])
#    last=len(model_xgb.loc[:])-1
#    print(model_xgb.loc[last:,["test-rmse-mean"]])
model_xgb = xgb.cv(params, dtrain,  num_boost_round=500,early_stopping_rounds=100)
model_xgb.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
last=len(model_xgb.loc[:])-1
print(model_xgb.loc[last:,["test-rmse-mean"]])
#Fit a model with optimized parameters
model_xgb = xgb.XGBRegressor(n_estimators=last,max_depth=2, learning_rate=0.1,
                             gamma=0,subsample=0.8,colsample_bytree=1,min_child_weight=10,
                            reg_alpha=7e-3,reg_lambda=1)
model_xgb.fit(training_examples, training_targets)
xgb.plot_importance(model_xgb)

#Predictions
xgb_BG_preds = np.expm1(model_xgb.predict(test_examples))
#XGB model for EF
training_targets=np.log1p(Targets_df["formation_energy_ev_natom"].copy())
dtrain = xgb.DMatrix(training_examples, label = training_targets)
dtest = xgb.DMatrix(test_examples)
#tune wrt params
params = {"max_depth":4,
          "eta":0.08,
          'gamma':0,  
         'subsample':1,
          'colsample_bytree':0.4,
          'min_child_weight':3,
          'reg_alpha':0,
          'reg_lambda':0,
         }
#for i in range(1,11):
#    params["max_depth"]=i
#    for j in range(1,11):
#        params["min_child_weight"]=j
#        model_xgb = xgb.cv(params, dtrain,  num_boost_round=500,early_stopping_rounds=100)
#        model_xgb.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
#        print("max_depth:")
#        print(params["max_depth"])
#        print("min_child_weight:")
#        print(params["min_child_weight"])
#        last=len(model_xgb.loc[:])-1
#        print(model_xgb.loc[last:,["test-rmse-mean"]])
#for i in np.arange(0,1.1,0.1):
#    params["gamma"]=i
#    model_xgb = xgb.cv(params, dtrain,  num_boost_round=500,early_stopping_rounds=100)
#    model_xgb.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
#    print("gamma:")
#    print(params["gamma"])
#    last=len(model_xgb.loc[:])-1
#    print(model_xgb.loc[last:,["test-rmse-mean"]])
#for i in np.arange(0.1,1.1,0.1):
#    params["subsample"]=i
#    for j in np.arange(0.1,1.1,0.1):
#        params["colsample_bytree"]=j
#        model_xgb = xgb.cv(params, dtrain,  num_boost_round=500,early_stopping_rounds=100)
#        model_xgb.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
#        print("subsample:")
#        print(params["subsample"])
#        print("colsample_bytree:")
#        print(params["colsample_bytree"])
#        last=len(model_xgb.loc[:])-1
#        print(model_xgb.loc[last:,["test-rmse-mean"]])
#for i in (0,1,1e-4,1e-3,1e-2,0.1,1.5):
#    params["reg_lambda"]=i
#    model_xgb = xgb.cv(params, dtrain,  num_boost_round=1000,early_stopping_rounds=100)
#    print("reg_lambda:")
#    print(params["reg_lambda"])
#    last=len(model_xgb.loc[:])-1
#    print(model_xgb.loc[last:,["test-rmse-mean"]])
model_xgb = xgb.cv(params, dtrain,  num_boost_round=500,early_stopping_rounds=100)
model_xgb.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
last=len(model_xgb.loc[:])-1
print(model_xgb.loc[last:,["test-rmse-mean"]])
#Fit a model with optimized parameters
model_xgb = xgb.XGBRegressor(n_estimators=last,max_depth=4, learning_rate=0.08,
                            gamma= 0, colsample=1,colsample_bytree=0.4,min_child_weight=3,
                            reg_lambda=0)
model_xgb.fit(training_examples, training_targets)
xgb.plot_importance(model_xgb)

#Predictions
xgb_EF_preds = np.expm1(model_xgb.predict(test_examples))
Predictions_df=pd.DataFrame()
Predictions_df["id"]=test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"]=xgb_EF_preds
Predictions_df["bandgap_energy_ev"]=xgb_BG_preds
Predictions_df.to_csv("XGB_Nomad.csv",index=False)
Predictions_df=pd.DataFrame()
Predictions_df["id"]=test_id_df["id"].copy()
stacked_EF_preds=0.9*xgb_EF_preds+0.1*linear_EF_pred
Predictions_df["formation_energy_ev_natom"]=stacked_EF_preds
stacked_BG_preds=0.9*xgb_BG_preds+0.1*linear_BG_pred
Predictions_df["bandgap_energy_ev"]=stacked_BG_preds
Predictions_df.to_csv("Stacked_Nomad.csv",index=False)