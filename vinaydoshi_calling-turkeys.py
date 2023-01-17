import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import featuretools as ft

%matplotlib inline
train = pd.read_json('../input/train.json')

test = pd.read_json('../input/test.json')

submission = pd.read_csv('../input/sample_submission.csv')
submission['is_turkey'].unique()
train.info(verbose=True)


tgt = train[['is_turkey','vid_id']].copy()

train['set']='train'

test['set']='test'

train.drop(columns='is_turkey',axis=1,inplace=True)

test.info(verbose=True)
data = train.append(test, ignore_index=True, sort=False)

data.tail()
data.info(verbose=True)
data.head()
data['vid_id'].is_unique
data.index = data['vid_id']

data.drop(columns='vid_id',axis=1, inplace=True)



data.head()
sns.countplot(x='is_turkey',data=tgt)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, roc_curve, auc, make_scorer
# got this function idea from refering below kernel:

#https://www.kaggle.com/teemingyi/turkey-competition

#thanks !!

def auto_embed_to_cols(data,i):

    df=pd.DataFrame(data['audio_embedding'].loc[i])

    df['vid_id']=i

    return df
auto_embeds =[]

for i in data.index:

    auto_embeds.append(auto_embed_to_cols(data,i))

    

auto_embed_values = pd.concat(auto_embeds)

auto_embed_values.shape
auto_embed_values.columns=['feature_'+str(x) for x in auto_embed_values.columns[:128]] + ['vid_id']

auto_embed_values.head()
data = pd.merge(data,auto_embed_values, left_index=True, right_on='vid_id',how='inner')

data.drop(columns='audio_embedding',axis=1,inplace=True)

data.shape
data[data['set']=='test'].shape
#A nice way for identifying correlated features by Will Koehrsen in,

#https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough

data_corr = data.corr()

upper = data_corr.where(np.triu(np.ones(data_corr.shape),k=1).astype(np.bool))

to_drop = [col for col in data_corr.columns if any(abs(upper[col])>0.9)]

print(to_drop)
data_updated=data.copy()

data_updated['duration'] = data['start_time_seconds_youtube_clip']-data['end_time_seconds_youtube_clip']

data_updated.drop(columns='start_time_seconds_youtube_clip',axis=1,inplace=True)
data_updated[data_updated['set']=='test'].shape
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler
scorer = make_scorer(roc_auc_score, greater_is_better=True)

final_train = data_updated[data_updated['set']=='train'].merge(tgt, how='inner', on='vid_id')

final_test = data_updated[data_updated['set']=='test'].merge(submission, how='inner', on='vid_id')

y_train = final_train[['is_turkey']]

X_train = final_train.drop(columns=['vid_id','set','is_turkey'],axis=1)

y_test = final_test[['is_turkey']]

X_test = final_test.drop(columns=['vid_id','set','is_turkey'],axis=1)

sc = StandardScaler()

X_train_stdsc = sc.fit_transform(X_train)

X_test_stdsc = sc.transform(X_test)

X_train_stdsc
model_results = pd.DataFrame(columns=['model','cv_mean','cv_std'])

def model_cv(model, X_train, y_train, name, nfolds=10, model_results=None):

    cv_scores = cross_val_score(estimator=model, 

                                X=X_train, y=y_train,

                                scoring=scorer, n_jobs=-1,

                                cv=nfolds

                               )

    print(f'{nfolds} CV score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')

    if model_results is not None:

        model_results = model_results.append({'model':name,'cv_mean':cv_scores.mean(),'cv_std':cv_scores.std()},

                                             ignore_index=True)

    return model_results
#Baseline Logreg score

model_results = model_cv(model = LogisticRegression(),

                         X_train=X_train,

                         y_train = y_train,

                         name = 'LogReg',

                         nfolds=10,

                         model_results=model_results                        

                        )
def gridsearch(model, param_grid, X_train, y_train, nfolds):

    gs = GridSearchCV(estimator=model,param_grid=param_grid, n_jobs=-1, scoring=scorer, cv=nfolds, iid=True

                      ,verbose=10)

    gs.fit(X_train,y_train)

    #scores = cross_val_score(estimator=gs, X=X_train_stdsc, y=y_train, cv=nfolds, scoring=scorer,n_jobs=-1)



    #print(f'10 fold CV score: {round(scores.mean(), 5)} with std: {round(scores.std(), 5)}')

    print(f'best score: {gs.best_score_}\n Best Params: {gs.best_params_}')

    return gs.best_score_, gs.best_params_
submission['is_turkey'].unique()
param_grid = [{'penalty':['l1','l2'],

               'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100] ,

               'class_weight':[None, 'balanced']

              }]

best_score_logreg, best_param_logreg = gridsearch(model = LogisticRegression(solver='liblinear',multi_class='ovr', n_jobs=-1)

                  , param_grid=param_grid, X_train=X_train_stdsc, y_train = y_train.values.ravel() , nfolds=10)
model_results = model_cv(model = LogisticRegression(**best_param_logreg),

                         X_train=X_train_stdsc,

                         y_train = y_train,

                         name = 'LogReg_stdsc_gs',

                         nfolds=10,

                         model_results=model_results                        

                        )
#SVM

from sklearn.svm import SVC

import time

start= time.time()

model_results = model_cv(model = SVC(),

                         X_train=X_train_stdsc,

                         y_train = y_train,

                         name = 'SVC',

                         nfolds=10,

                         model_results=model_results                        

                        )

end=time.time()

print(f'Total TIme: {end-start}')
#reg_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100] 

#param_grid=[{'C':reg_range,

#             'kernel':['linear']

##             'gamma':range(1,20,5)

#            }]

#

#best_score_svc, best_param_svc = gridsearch(model = SVC()

#                  , param_grid=param_grid, X_train=X_train_stdsc, y_train = y_train.values.ravel() , nfolds=2)
best_param_svc= {'C': 0.001, 'kernel': 'linear'}

start= time.time()

model_results = model_cv(model = SVC(**best_param_svc),

                         X_train=X_train_stdsc,

                         y_train = y_train,

                         name = 'SVC_gs',

                         nfolds=10,

                         model_results=model_results                        

                        )

end=time.time()

print(f'Total TIme: {end-start}')
from sklearn.ensemble import RandomForestClassifier

start= time.time()

model_results = model_cv(model = RandomForestClassifier(n_estimators=100),

                         X_train=X_train_stdsc,

                         y_train = y_train,

                         name = 'RFC',

                         nfolds=10,

                         model_results=model_results                        

                        )

end=time.time()

print(f'Total TIme: {end-start}')
#reg_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100] 

#param_grid=[{'n_estimators':range(50,500,50),

#             'max_depth':range(2,150,10),

#             'min_samples_split':[2,5,10],

#             'min_samples_leaf':[1,2,4],

#             'bootstrap':[True,False]

#            }]

#

#best_score_rfc, best_param_rfc = gridsearch(model = RandomForestClassifier()

#                  , param_grid=param_grid, X_train=X_train_stdsc, y_train = y_train.values.ravel() , nfolds=2)
start= time.time()

model_results = model_cv(model = RandomForestClassifier(n_estimators=400, bootstrap=False,

                                                        max_depth=132,min_samples_leaf=1,

                                                        min_samples_split=5

                                                       ),

                         X_train=X_train_stdsc,

                         y_train = y_train,

                         name = 'RFC_gs',

                         nfolds=10,

                         model_results=model_results                        

                        )

end=time.time()

print(f'Total TIme: {end-start}')
from sklearn.neighbors import KNeighborsClassifier

for i in [2,5,10,20]:

    print(f'KNN neighbors: {i}')

    start= time.time()

    model_results = model_cv(model = KNeighborsClassifier(n_neighbors=i),

                             X_train=X_train_stdsc,

                             y_train = y_train,

                             name = f'KNN-{i}',

                             nfolds=10,

                             model_results=model_results                        

                            )

    end=time.time()

    print(f'Total TIme for KNN-{i}: {end-start}')

    
from lightgbm import LGBMClassifier

start= time.time()

model_results = model_cv(model = LGBMClassifier(),

                         X_train=X_train_stdsc,

                         y_train = y_train,

                         name = 'LGBM',

                         nfolds=10,

                         model_results=model_results                        

                        )

end=time.time()

print(f'Total TIme: {end-start}')
plt.figure(figsize=(15,12))

sns.barplot(data=model_results, x='model',y='cv_mean',yerr=list(model_results['cv_std']),color='orange',edgecolor='k',linewidth=2)

plt.xticks(rotation=90)

plt.title('Model F1 Score Results')

plt.ylabel('Mean F1 Score (with error bar)')
#scorer = make_scorer(roc_auc_score, greater_is_better=True)

final_train = data_updated[data_updated['set']=='train'].merge(tgt, how='inner', on='vid_id')

final_test = data_updated[data_updated['set']=='test'].merge(submission, how='inner', on='vid_id')

#y_train = final_train[['is_turkey']]

X_train2 = final_train.drop(columns=['set'],axis=1)

#y_test = final_test[['is_turkey']]

X_test2 = final_test.drop(columns=['set'],axis=1)

#sc = StandardScaler()

#X_train_stdsc = sc.fit_transform(X_train)

#X_test_stdsc = sc.transform(X_test)

#X_train_stdsc
#Adding additional information from data

#https://www.kaggle.com/frtgnn/yam-potatoes-thanksgiving-2018

#Thanks!



X_train2_columns = X_train2.columns

X_test2_columns  = X_test2.columns



X_train2['all_feature_mean'] = X_train2[X_train2_columns[4:131]].mean(axis=1)

X_test2['all_feature_mean']  = X_test2[X_test2_columns[3:130]].mean(axis=1)



X_train2['all_feature_median'] = X_train2[X_train2_columns[4:131]].median(axis=1)

X_test2['all_feature_median']  = X_test2[X_test2_columns[3:130]].median(axis=1)



X_train2['all_feature_min'] = X_train2[X_train2_columns[4:131]].min(axis=1)

X_test2['all_feature_min']  = X_test2[X_test2_columns[3:130]].min(axis=1)



X_train2['all_feature_max'] = X_train2[X_train2_columns[4:131]].max(axis=1)

X_test2['all_feature_max']  = X_test2[X_test2_columns[3:130]].max(axis=1)



X_train2['all_feature_std'] = X_train2[X_train2_columns[4:131]].std(axis=1)

X_test2['all_feature_std']  = X_test2[X_test2_columns[3:130]].std(axis=1)
X_train2_concat = X_train2.groupby('vid_id').mean()

y_train_concat = X_train2_concat['is_turkey']

X_train2_concat.drop(['is_turkey'],axis=1,inplace=True)



X_test2_concat = X_test2.groupby('vid_id').mean()

y_test = X_test2_concat['is_turkey']

X_test2_concat.drop(['is_turkey'],axis=1,inplace=True)
sc = StandardScaler()

X_train2_concat_stdsc = sc.fit_transform(X_train2_concat)

X_test2_concat_stdsc = sc.transform(X_test2_concat)

X_train2_concat_stdsc
model_results = model_cv(model = LogisticRegression(),

                         X_train=X_train2_concat,

                         y_train = y_train_concat,

                         name = 'LogReg_new_data',

                         nfolds=10,

                         model_results=model_results                        

                        )
param_grid = [{'penalty':['l1'],

               'solver':['liblinear','saga'],

               'C':[0.00001,0.0001, 0.001, 0.01, 0.1, 1, 10, 100] ,

               'class_weight':[None, 'balanced']},

              {'solver':['newton-cg', 'lbfgs', 'sag','saga'],

               'penalty':['l2','none'],

               'C':[0.00001,0.0001, 0.001, 0.01, 0.1, 1, 10, 100] ,

               'class_weight':[None, 'balanced']                 

              }]

best_score_logreg, best_param_logreg = gridsearch(model = LogisticRegression(multi_class='ovr', n_jobs=-1,max_iter=100)

                  , param_grid=param_grid, X_train=X_train2_concat, y_train = y_train_concat.values.ravel() , nfolds=10)
model_results = model_cv(model = LogisticRegression(**best_param_logreg),

                         X_train=X_train2_concat,

                         y_train = y_train_concat,

                         name = 'LogReg_new_data_gs',

                         nfolds=10,

                         model_results=model_results                        

                        )
import time

from sklearn.svm import SVC

start= time.time()

model_results = model_cv(model = SVC(),

                         X_train=X_train2_concat,

                         y_train = y_train_concat,

                         name = 'SVC_new_data',

                         nfolds=10,

                         model_results=model_results                        

                        )

end=time.time()

print(f'Total TIme: {end-start}')
#reg_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100] 

#param_grid=[{'C':reg_range,

#             'kernel':['poly'],

#             'gamma':range(1,5,1),

#             'degree':range(1,5,1)

#            }]

#

#best_score_svc, best_param_svc = gridsearch(model = SVC()

#                  , param_grid=param_grid, X_train=X_train2_concat, y_train = y_train_concat.values.ravel() , nfolds=2)
best_param_svc= {'C': 0.0001, 'degree': 4, 'gamma': 1, 'kernel': 'poly'}

start= time.time()

model_results = model_cv(model = SVC(**best_param_svc),

                         X_train=X_train2_concat,

                         y_train = y_train_concat,

                         name = 'SVC_new_data_gs',

                         nfolds=10,

                         model_results=model_results                        

                        )

end=time.time()

print(f'Total TIme: {end-start}')
from sklearn.ensemble import RandomForestClassifier

start= time.time()

model_results = model_cv(model = RandomForestClassifier(n_estimators=100),

                         X_train=X_train2_concat,

                         y_train = y_train_concat,

                         name = 'RFC_new_data',

                         nfolds=10,

                         model_results=model_results                        

                        )

end=time.time()

print(f'Total TIme: {end-start}')
#reg_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100] 

#param_grid=[{'n_estimators':range(400,600,50),

#             'max_depth':range(120,150,10),

#             'min_samples_split':[2,5,10],

#             'min_samples_leaf':[1,2,3],

#             'bootstrap':[True,False]

#            }]



#best_score_rfc, best_param_rfc = gridsearch(model = RandomForestClassifier()

#                  , param_grid=param_grid, X_train=X_train2_concat, y_train = y_train_concat.values.ravel() , nfolds=5)
best_param_rfc= {'bootstrap': True, 'max_depth': 62, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 150}

start= time.time()

model_results = model_cv(model = RandomForestClassifier(**best_param_rfc),

                         X_train=X_train2_concat,

                         y_train = y_train_concat,

                         name = 'RFC_new_data_gs',

                         nfolds=10,

                         model_results=model_results                        

                        )

end=time.time()

print(f'Total TIme: {end-start}')
from sklearn.neighbors import KNeighborsClassifier

for i in [2,5,10,20]:

    print(f'KNN neighbors: {i}')

    start= time.time()

    model_results = model_cv(model = KNeighborsClassifier(n_neighbors=i),

                             X_train=X_train2_concat,

                             y_train = y_train_concat,

                             name = f'KNN_new_data-{i}',

                             nfolds=10,

                             model_results=model_results                        

                            )

    end=time.time()

    print(f'Total TIme for KNN-{i}: {end-start}')
from lightgbm import LGBMClassifier

start= time.time()

model_results = model_cv(model = LGBMClassifier(),

                         X_train=X_train2_concat,

                         y_train = y_train_concat,

                         name = 'LGBM_new_data',

                         nfolds=10,

                         model_results=model_results                        

                        )

end=time.time()

print(f'Total TIme: {end-start}')
plt.figure(figsize=(15,7))

sns.barplot(data=model_results, x='model',y='cv_mean',yerr=list(model_results['cv_std']),color='orange',edgecolor='k',linewidth=2)

plt.xticks(rotation=90)

plt.title('Model ROC Score Results')

plt.ylabel('Mean ROC Score (with error bar)')
# Additional features have massive effect! This indicates initial data was not enough.

# Tuned Logreg leads with 0.95! Regularisation results in increase of 0.03!!

# Power of LGBM!!It reaches 0.94 without tuning.

# Need to re-tune RFC.

# SVC tuning indicates massive effect due to regularisation. Need to re-tune with multiple kernels and verify.
from IPython.display import display

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

def modelgbm(X_train, y_train, X_test, train_features, nfolds=5):

    model = lgb.LGBMClassifier( objective='binary', n_jobs=-1, n_estimators=10000)

    strkfold = StratifiedKFold(n_splits=nfolds,shuffle=True)

    predictions = pd.DataFrame()

    importances = np.zeros(len(train_features))

    tr = np.array(X_train)

    ts = np.array(X_test)

    tgt = np.array(y_train).reshape((-1))

    valid_scores=[]

    

    for i, (train_indices, tgt_indices) in enumerate(strkfold.split(tr,tgt)):

        

        fold_predictions = pd.DataFrame()

        

        X_tr = tr[train_indices]

        X_tgt = tr[tgt_indices]

        y_tr = tgt[train_indices]

        y_tgt = tgt[tgt_indices]

        

        model.fit(X=X_tr, y=y_tr, early_stopping_rounds=100, eval_metric='auc', eval_set=[(X_tr,y_tr), (X_tgt,y_tgt)],

                  eval_names=['train','valid'],

                  verbose=200

                 )

        

        valid_scores.append(model.best_score_['valid']['auc'])

        fold_probabilities = model.predict_proba(X_test)

        for j in range(2):

            fold_predictions[j] = fold_probabilities[:,j]

        fold_predictions['vid_id'] = X_test.index

        fold_predictions['fold'] = i+1

        predictions = predictions.append(fold_predictions)

        importances = model.feature_importances_/nfolds

        print(f'Fold {i+1}: Validation Score = {round(valid_scores[i],5)}, Estimators Trained = {model.best_iteration_}')

    

    feature_importances = pd.DataFrame({'feature':train_features, 'importance':importances})

    #if return_preds:

    #    predictions['Target'] = predictions[[0,1]].idxmax(axis=1)

    #    predictions['Probability_Confidence'] = predictions[[0,1]].max(axis=1)

    predictions['Target'] = predictions[[0,1]].idxmax(axis=1)

    predictions['Probability_Confidence'] = predictions[[0,1]].max(axis=1)

    predictions = predictions.groupby('vid_id',as_index=False).mean()

    submission = predictions[['vid_id','Target']].copy()

    submission.columns = ['vid_id','is_turkey']

    

    return valid_scores, predictions,submission
valid_scores, predictions,submission=modelgbm(X_train=X_train2_concat, y_train = y_train_concat, X_test=X_test2_concat, train_features=X_train2_concat.columns)

submission.to_csv('submission.csv', index=False)