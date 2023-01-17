#  Libraries

import numpy as np 

import pandas as pd 

# Data processing, metrics and modeling

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import StratifiedKFold

from datetime import datetime

from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve

from sklearn import metrics

# Lgbm

import lightgbm as lgb

import catboost

from catboost import Pool

import xgboost as xgb



# Suppr warning

import warnings

warnings.filterwarnings("ignore")



import itertools

from scipy import interp



# Plots

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import rcParams





#Timer

def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('Time taken for Modeling: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
data = pd.read_csv('../input/creditcard.csv')
display(data.head(),data.describe(),data.shape)
f,ax=plt.subplots(15,2,figsize=(12,60))

#f.delaxes(ax)

col = list(data)

col = [e for e in col if e not in ('Class')]



for i,feature in enumerate(col):

    sns.distplot(data[data['Class']==1].dropna()[(feature)], ax=ax[i//2,i%2], kde_kws={"color":"black"}, hist=False )

    sns.distplot(data[data['Class']==0].dropna()[(feature)], ax=ax[i//2,i%2], kde_kws={"color":"black"}, hist=False )



    # Get the two lines from the ax[i//2,i%2]es to generate shading

    l1 = ax[i//2,i%2].lines[0]

    l2 = ax[i//2,i%2].lines[1]



    # Get the xy data from the lines so that we can shade

    x1 = l1.get_xydata()[:,0]

    y1 = l1.get_xydata()[:,1]

    x2 = l2.get_xydata()[:,0]

    y2 = l2.get_xydata()[:,1]

    ax[i//2,i%2].fill_between(x2,y2, color="deeppink", alpha=0.6)

    ax[i//2,i%2].fill_between(x1,y1, color="darkturquoise", alpha=0.6)



    #grid

    ax[i//2,i%2].grid(b=True, which='major', color='grey', linewidth=0.3)

    

    ax[i//2,i%2].set_title('{} by target'.format(feature), fontsize=18)

    ax[i//2,i%2].set_ylabel('count', fontsize=12)

    ax[i//2,i%2].set_xlabel('Modality', fontsize=12)



    #sns.despine(ax[i//2,i%2]=ax[i//2,i%2], left=True)

    ax[i//2,i%2].set_ylabel("frequency", fontsize=12)

    ax[i//2,i%2].set_xlabel(str(feature), fontsize=12)



plt.tight_layout()

plt.show()
#reshape amount

data['nAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

#drop useless 

data = data.drop(columns = ['Time','Amount'])
for i in range(1, 29):

    col = 'V%d'%i

    data['rounded_4_'+str(col)] = round(data[col],4) 
for i in range(1, 29):

    col = 'rounded_4_V%d'%i

    var = data.groupby(col).agg({col:'count'})

    var.columns = ['%s_count'%col]

    data = pd.merge(data,var,on=col,how='left')
train_df = data

features = list(train_df)

features.remove('Class')

target = 'Class'
# Confusion matrix 

def plot_confusion_matrix(cm, classes,

                          normalize = False,

                          title = 'Confusion matrix"',

                          cmap = plt.cm.Blues) :

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation = 0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :

        plt.text(j, i, cm[i, j],

                 horizontalalignment = 'center',

                 color = 'white' if cm[i, j] > thresh else 'black')

 

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
def gradient_boosting_model(params, folds, model='LGB'):

    print(str(model)+' modeling...')

    start_time = timer(None)

    

    plt.rcParams["axes.grid"] = True



    nfold = 5

    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)



    oof = np.zeros(len(train_df))

    mean_fpr = np.linspace(0,1,100)

    cms= []

    tprs = []

    aucs = []

    y_real = []

    y_proba = []

    recalls = []

    roc_aucs = []

    f1_scores = []

    accuracies = []

    precisions = []

    feature_importance_df = pd.DataFrame()



    i = 1

    for train_idx, valid_idx in skf.split(train_df, train_df.Class.values):

        print("\nfold {}".format(i))

        

        if model == 'LGB':

        

            trn_data = lgb.Dataset(train_df.iloc[train_idx][features].values,

                                   label=train_df.iloc[train_idx][target].values

                                   )

            val_data = lgb.Dataset(train_df.iloc[valid_idx][features].values,

                                   label=train_df.iloc[valid_idx][target].values

                                   )   



            clf = lgb.train(param_lgb, trn_data, num_boost_round=10000,  valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)

            oof[valid_idx] = clf.predict(train_df.iloc[valid_idx][features].values) 

            

        if model == 'CB':

            

            trn_data = Pool(train_df.iloc[train_idx][features].values,

                           label=train_df.iloc[train_idx][target].values

                           )

            val_data = Pool(train_df.iloc[valid_idx][features].values,

                            label=train_df.iloc[valid_idx][target].values

                            )   



            clf = catboost.train(trn_data, param_cb, eval_set=val_data, verbose = 100)



            oof[valid_idx]  = clf.predict(train_df.iloc[valid_idx][features].values)   

            oof[valid_idx]  = np.exp(oof[valid_idx]) / (1 + np.exp(oof[valid_idx]))

            

        if model == 'XGB':



            trn_data = xgb.DMatrix(train_df.iloc[train_idx][features], 

                                   label=train_df.iloc[train_idx][target].values)

            val_data = xgb.DMatrix(train_df.iloc[valid_idx][features], 

                                   label=train_df.iloc[valid_idx][target].values)



            watchlist = [(trn_data, 'train'), (val_data, 'valid')]



            clf = xgb.train(params, dtrain = trn_data, evals=watchlist, early_stopping_rounds=100, maximize=True, verbose_eval=100)

            oof[valid_idx] = clf.predict(val_data, ntree_limit=clf.best_ntree_limit)



        # Scores 

        roc_aucs.append(roc_auc_score(train_df.iloc[valid_idx][target].values, oof[valid_idx]))

        accuracies.append(accuracy_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))

        recalls.append(recall_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))

        precisions.append(precision_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))

        f1_scores.append(f1_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))



        # Roc curve by folds

        f = plt.figure(1)

        fpr, tpr, t = roc_curve(train_df.iloc[valid_idx][target].values, oof[valid_idx])

        tprs.append(interp(mean_fpr, fpr, tpr))

        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))



        # Precion recall by folds

        g = plt.figure(2)

        precision, recall, _ = precision_recall_curve(train_df.iloc[valid_idx][target].values, oof[valid_idx])

        y_real.append(train_df.iloc[valid_idx][target].values)

        y_proba.append(oof[valid_idx])

        plt.plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (i))  



        i= i+1

        

        # Confusion matrix by folds

        cms.append(confusion_matrix(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))

        

        # Features imp

        fold_importance_df = pd.DataFrame()

        fold_importance_df["Feature"] = features

        if model == 'LGB':

            fold_importance_df["importance"] = clf.feature_importance()

        if model == 'CB':

            fold_importance_df["importance"] = clf.get_feature_importance()

        fold_importance_df["fold"] = nfold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)    



    # Metrics

    print(

            '\nCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),

            '\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),

            '\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),

            '\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),

            '\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))

    )

    

    # Roc plt

    f = plt.figure(1)

    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')

    mean_tpr = np.mean(tprs, axis=0)

    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, color='blue',

             label=r'Mean ROC (AUC = %0.4f)' % (np.mean(roc_aucs)),lw=2, alpha=1)



    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title(str(model)+' ROC curve by folds')

    plt.legend(loc="lower right")

    

    # PR plt

    g = plt.figure(2)

    plt.plot([0,1],[1,0],linestyle = '--',lw = 2,color = 'grey')

    y_real = np.concatenate(y_real)

    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    plt.plot(recall, precision, color='blue',

             label=r'Mean P|R')

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.title(str(model)+' P|R curve by folds')

    plt.legend(loc="lower left")



    # Confusion maxtrix

    plt.rcParams["axes.grid"] = False

    cm = np.average(cms, axis=0)

    class_names = [0,1]

    plt.figure()

    plot_confusion_matrix(cm, 

                          classes=class_names, 

                          title= str(model).title()+' Confusion matrix [averaged/folds]')

    

    # Feat imp plt

    if model != 'XGB':

        cols = (feature_importance_df[["Feature", "importance"]]

            .groupby("Feature")

            .mean()

            .sort_values(by="importance", ascending=False)[:30].index)

        best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



        plt.figure(figsize=(10,10))

        sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False),

                edgecolor=('white'), linewidth=2, palette="rocket")

        plt.title(str(model)+' Features importance (averaged/folds)', fontsize=18)

        plt.tight_layout()

        

    # Timer end    

    timer(start_time)
param_xgb = {

            'n_jobs' : -1, 

            'n_estimators' : 200,

            'seed' : 1337,

            'random_state':1337,

            'eval_metric':'auc'

    }
gradient_boosting_model(param_xgb, 5, 'XGB')
param_lgb = {

            'bagging_fraction': 0.8082060379239122,

            'colsample_bytree': 0.4236846658378094,

            'feature_fraction': 0.1622850961512378,

            'learning_rate': 0.24617571597038826,

            'max_depth': 10,

            'min_child_samples': 110.4846966877894,

            'min_child_weight': 0.0077240770377460955,

            'min_data_in_leaf': 16,

            'num_leaves': 8,

            'reg_alpha': 0.6051612648874549,

            'reg_lambda': 97.89699721669824,

            'subsample': 0.20955925262252026,

            'objective': 'binary',

            'save_binary': True,

            'seed': 1337,

            'feature_fraction_seed': 1337,

            'bagging_seed': 1337,

            'drop_seed': 1337,

            'data_random_seed': 1337,

            'boosting_type': 'gbdt',

            'verbose': 1,

            'is_unbalance': False,

            'boost_from_average': True,

            'metric':'auc'

    }
gradient_boosting_model(param_lgb, 5, 'LGB')
param_cb = {

            'learning_rate': 0.09445645065743215,

            'colsample_bylevel' : 0.24672360788274705,

            'bagging_temperature': 0.39963209507789, 

            'l2_leaf_reg': int(22.165305913463673),

            'depth': int(7.920859337748043), 

            'iterations' : 500,

            'loss_function' : "Logloss",

            'objective':'CrossEntropy',

            'eval_metric' : "AUC",

            'bootstrap_type' : 'Bayesian',

            'random_seed':1337,

            'early_stopping_rounds' : 100,

            'use_best_model':True 

    }
gradient_boosting_model(param_cb, 5, 'CB')