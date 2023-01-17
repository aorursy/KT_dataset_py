# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import random

random.seed(50)
df = pd.read_csv("../input/minor-project-2020/train.csv")
df
df.info()
# so we dont have null values in the dataframe

X = df.drop(['id','target'],axis=1)

y = df['target']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
from sklearn.preprocessing import StandardScaler



scale = StandardScaler()



sc_X_train = scale.fit_transform(X_train)

sc_X_test = scale.transform(X_test)

sc_X= scale.transform(X)
from sklearn.tree import DecisionTreeClassifier
# 0.502 - submission (1) pred1_m.csv

dec = DecisionTreeClassifier()
dec2 = DecisionTreeClassifier(criterion ="entropy", max_depth=200)
# pred5_m : 0.502038451903807

dec2.fit(sc_X_train,y_train)
y_pred = dec2.predict(sc_X_test)



from sklearn.metrics import roc_curve, auc

FPR, TPR, _ = roc_curve(y_test, y_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV



parameters = {'criterion': ("gini", "entropy"), 'max_depth': (50,300)}



dec_gs = DecisionTreeClassifier()



opt = GridSearchCV(dec_gs, parameters, verbose=1)



opt.fit(sc_X_train, y_train)



y_pred = opt.predict(sc_X_test)
from sklearn.metrics import roc_curve, auc

FPR, TPR, _ = roc_curve(y_test, y_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler



scale = StandardScaler()

sc_X_train = scale.fit_transform(X_train)

sc_X_test = scale.transform(X_test)
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold
'''XGB_model = XGBClassifier()



parameters={

    'scale_pos_weight':[1000]

}



cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# GridSearch Evaluates on the basis of roc_auc

XGB_grid = GridSearchCV(estimator=XGB_model, param_grid=parameters, n_jobs=-1, cv=cv, scoring='roc_auc')



XGB_result = XGB_grid.fit(sc_X, y)



print(XGB_grid.best_score_)

print(XGB_grid.best_params_)'''
'''#After Performing Grid Search - set of Best parameters obtained (4,1000)

XGB_model = XGBClassifier()



parameters={

   

    'scale_pos_weight':[1000],

    'max_depth':[4,5]

}



cv_s = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)



XGB_grid = GridSearchCV(estimator=XGB_model, param_grid=parameters, n_jobs=-1, cv=cv_s, scoring='roc_auc')



XGB_result = XGB_grid.fit(sc_X, y)

'''
# Uses parameters from the above grid search



# The above code was run to find the best set of parameters - for pred13p_m.csv

XGB_model = XGBClassifier(scale_pos_weight=1000,max_depth=4)

XGB_result = XGB_model.fit(sc_X, y)
# the parameter values have been obtained by using grid search from the previous command



xgb_model_d = XGBClassifier(scale_pos_weight=1000,max_depth=4,subsample=0.7,colsample_bytree=0.7)

xgb_model_d.fit(sc_X_train,y_train)



y_pred = xgb_model_d.predict(sc_X_test)



from sklearn.metrics import roc_curve, auc

FPR, TPR, _ = roc_curve(y_test, y_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)

#0.5598008517034068
# Using these set of parameters to train next model

XGB_model5= XGBClassifier(scale_pos_weight=1000,max_depth=4,subsample=0.7,colsample_bytree=0.7)

XGB_model5.fit(sc_X,y)



# 2 nd last submission 
# these values are set by testing various parameters and the values obtained from grid search



xgb_model_d1 = XGBClassifier(scale_pos_weight=1000,max_depth=4,subsample=0.7,colsample_bytree=0.7,min_child_weight=10)

xgb_model_d1.fit(sc_X_train,y_train)



y_pred = xgb_model_d1.predict(sc_X_test)



from sklearn.metrics import roc_curve, auc

FPR, TPR, _ = roc_curve(y_test, y_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



#0.5712111723446893 - 10 min_child_weight

# final submission
XGB_model6= XGBClassifier(scale_pos_weight=1000,max_depth=4,subsample=0.7,colsample_bytree=0.7,min_child_weight=10)

XGB_model6.fit(sc_X,y)

# final submission
from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler
# Creating 10% of np. of examples present in training set

over_sm = SMOTE(sampling_strategy=0.1, k_neighbors=2,random_state=0)

# sample such that Majority class to Minority class ratio =0.5

under_sm = RandomUnderSampler(sampling_strategy=1,random_state=0)



t_X_train, t_y_train = over_sm.fit_sample(sc_X_train, y_train)

under_sm.fit(X, y)

X_train_res, y_train_res = under_sm.fit_resample(t_X_train,t_y_train)
xgb_res = XGBClassifier(max_depth=4) 



# testing on validation set

xgb_res.fit(X_train_res, y_train_res.ravel()) 

y_pred = xgb_res.predict(sc_X_test)  



from sklearn.metrics import roc_curve, auc

FPR, TPR, _ = roc_curve(y_test, y_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)
# writing predictions - class wise predictions

df_test = pd.read_csv("../input/minor-project-2020/test.csv")



X_test_set = df_test.drop('id',axis=1)



sc_X_test_set = scale.fit_transform(X_test_set)



y_pred_test = XGB_model6.predict(sc_X_test_set)



subm= {

    'id':df_test['id'],

    'target': y_pred_test.astype(int)

}



df_subm = pd.DataFrame(subm,columns=['id','target'])



filename = 'pred16a_m.csv'

df_subm.to_csv(filename,index=False)
# writing predictions - Probabilities of the class

df_test = pd.read_csv("../input/minor-project-2020/test.csv")



X_test_set = df_test.drop('id',axis=1)



sc_X_test_set = scale.fit_transform(X_test_set)



y_pred_test = XGB_model6.predict_proba(sc_X_test_set)



chk = pd.DataFrame(y_pred_test)



subm= {

    'id':df_test['id'],

    'target': chk[1].astype(float)

}



df_subm = pd.DataFrame(subm,columns=['id','target'])



filename = 'pred17p_m.csv'

df_subm.to_csv(filename,index=False)