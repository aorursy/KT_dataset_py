#Importing Data manipulation and plotting modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import time
#Importing libraries for performance measures

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve
#Importing libraries For data splitting

from sklearn.model_selection import train_test_split

from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
#Import libraries for data balancing

from imblearn.over_sampling import SMOTE, ADASYN
data = pd.read_csv("../input/creditcard.csv")
pd.options.display.max_columns = 200
data.head()
data.shape
sns.countplot(data['Class'])
f, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.distplot(data['Amount'].values,ax=axes[0])

axes[0].set_title("Distribution of Transaction Amount")

sns.distplot(data['Time'].values,ax=axes[1])

axes[1].set_title("Distribution of Transaction Time (in seconds)")

plt.show()
data.drop(['Time'], inplace = True, axis =1)


data['Class'].value_counts()[1]/data.shape[0]
y = data.iloc[:,29]

X = data.iloc[:,0:29]

X.shape
y.shape
X_train, X_test, y_train, y_test =   train_test_split(X,

                                                      y,

                                                      test_size = 0.3,

                                                      stratify = y

                                                      )
X_train.shape
y_train.shape
max_delta_step= [1,2,3,4,5,6,7,8,9,10]

scale_pos_weight= [1,2,3,4,5,6,7,8,9,10]

num_zeros = (data['Class'] == 0).sum()

num_ones = (data['Class'] == 1).sum()

sp_weight = num_zeros / num_ones

for i in max_delta_step:

    print('--------------------')

    print('Iteration ', i)

    print('--------------------')

    print('scale_pos_weight = {} '.format(i))

    print('max_delta_step = {} '.format(i))

    xgb = XGBClassifier(scale_pos_weight = i,max_delta_step=i)

    xgb.fit(X_train,y_train)

    xgb_predict = xgb.predict(X_test)

    xgb_proba = xgb.predict_proba(X_test)

    xgb_cm = confusion_matrix(y_test, xgb_predict)

    p_xg,r_xg,f_xg,_ = precision_recall_fscore_support(y_test,xgb_predict)

    print('Accuracy',accuracy_score(y_test, xgb_predict))

    print('Confusion Matrix: \n', xgb_cm)

    print('Precision: ',p_xg)

    print('Recall: ',r_xg)

    print('F score: ',f_xg)
rf = RandomForestClassifier(n_estimators=100,random_state=0, n_jobs=-1, class_weight="balanced")
rf1 = rf.fit(X_train,y_train)

y_pred_rf = rf1.predict(X_test)

y_pred_rf_prob = rf1.predict_proba(X_test)
accuracy_score(y_test,y_pred_rf)
confusion_matrix(y_test,y_pred_rf)
p_rf,r_rf,f_rf,_  = precision_recall_fscore_support(y_test,y_pred_rf)
print('Precision:',p_rf , '\nRecall',r_rf,'\nFscore',f_rf,_ )
sm = SMOTE(random_state=42)
X_smote, y_smote = sm.fit_sample(X_train, y_train)
X_smote.shape
y_smote.shape
np.sum(y_smote)/len(y_smote)



#We can see now the data is balanced
y_smote = y_smote.reshape(y_smote.size, 1)

y_smote.shape 
xg_smote = XGBClassifier(learning_rate=0.1,

                   reg_alpha= 0,

                   reg_lambda= 1,

                   )

rf_smote = RandomForestClassifier(n_estimators=100,n_jobs=-1)



columns = X_train.columns

X_smote = pd.DataFrame(data = X_smote, columns = columns)



xg_fit = xg_smote.fit(X_smote,y_smote)

rf_fit = rf_smote.fit(X_smote,y_smote)



y_pred_xgb = xg_fit.predict(X_test)

y_pred_rfb = rf_fit.predict(X_test)



y_pred_xgb_prob = xg_fit.predict_proba(X_test)

y_pred_rfb_prob = rf_fit.predict_proba(X_test)



p_rfb,r_rfb,f_rfb,_  = precision_recall_fscore_support(y_test,y_pred_rfb)

p_xgb,r_xgb,f_xgb,_  = precision_recall_fscore_support(y_test,y_pred_xgb)





print('Random Forest:\n')

print('Accuracy - ',accuracy_score(y_test,y_pred_rfb))

print('\nPrecision - ',p_rfb , '\nRecall - ',r_rfb,'\nFscore - ',f_rfb,_ )

print('Confusion Matrix -\n',confusion_matrix(y_test,y_pred_rfb))



print('XGBoost:\n')

print('Accuracy - ',accuracy_score(y_test,y_pred_xgb))

print('\nPrecision - ',p_xgb , '\nRecall - ',r_xgb,'\nFscore - ',f_xgb,_ )

print('Confusion Matrix - \n',confusion_matrix(y_test,y_pred_xgb))



adasyn = ADASYN(random_state=42)

X_ada, y_ada = adasyn.fit_sample(X_train, y_train)
X_ada.shape
y_ada.shape
np.sum(y_ada)/len(y_ada)



y_ada = y_ada.reshape(y_ada.size, 1)

y_ada.shape 



xg_ada = XGBClassifier(learning_rate=0.5,

                   reg_alpha= 0.1,

                   reg_lambda= 1,

                   )





rf_ada = RandomForestClassifier(n_estimators=100,n_jobs=-1)



columns = X_train.columns

X_ada = pd.DataFrame(data = X_ada, columns = columns)



xg_fit = xg_ada.fit(X_ada,y_ada)





rf_fit = rf_ada.fit(X_ada,y_ada)



y_pred_xgb = xg_fit.predict(X_test)





y_pred_rfb = rf_fit.predict(X_test)



y_pred_xgb_prob = xg_fit.predict_proba(X_test)

y_pred_rfb_prob = rf_fit.predict_proba(X_test)



p_rfb,r_rfb,f_rfb,_  = precision_recall_fscore_support(y_test,y_pred_rfb)



p_xgb,r_xgb,f_xgb,_  = precision_recall_fscore_support(y_test,y_pred_xgb)





print('Random Forest:\n')

print('Accuracy - ',accuracy_score(y_test,y_pred_rfb))

print('Precision - \n',p_rfb , 'Recall - \n',r_rfb,'Fscore - \n',f_rfb,_ )

print('Confusion Matrix - \n',confusion_matrix(y_test,y_pred_rfb))



print('XGBoost:\n')

print('Accuracy XGBoost',accuracy_score(y_test,y_pred_xgb))

print('Precision XGBoost:',p_xgb , 'Recall',r_xgb,'Fscore',f_xgb,_ )

print('Confusion Matrix - \n',confusion_matrix(y_test,y_pred_xgb))
