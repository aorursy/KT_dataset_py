# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import VarianceThreshold



# 导入基本分类器

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, NuSVC

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from tqdm import tqdm_notebook

from sklearn.metrics import roc_auc_score



%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
oof_qda = np.zeros(train.shape[0])

pred_qda = np.zeros(test.shape[0])

qda_params = {

    'reg_param': 0.6

}



cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

valid_features = [column for column in train.columns if column not in ['id', 'target', 'wheezy-copper-turtle-magic']] 



folds = StratifiedKFold(n_splits=25, shuffle=True, random_state=137)



for i in tqdm_notebook(range(512)):

    train_c = train[train['wheezy-copper-turtle-magic']==i]

    test_c = test[test['wheezy-copper-turtle-magic']==i]

    train_index_c = train_c.index

    test_index_c = test_c.index

    train_c.reset_index(drop=True, inplace=True)

    test_c.reset_index(drop=True, inplace=True)

    # 选择40个特征

    sel = VarianceThreshold(threshold=1.5).fit(train_c[valid_features])

    train_s = sel.transform(train_c[valid_features])

    test_s = sel.transform(test_c[valid_features])

    for fold_, (train_idx, val_idx) in enumerate(folds.split(train_s, train_c['target'])):

        model = QuadraticDiscriminantAnalysis(**qda_params)

        model.fit(train_s[train_idx], train_c.loc[train_idx]['target'])

        oof_qda[train_index_c[val_idx]] = model.predict_proba(train_s[val_idx])[:, 1]

        pred_qda[test_index_c] += model.predict_proba(test_s)[:,1] / folds.n_splits
print(roc_auc_score(train['target'], oof_qda))
# INITIALIZE VARIABLES

test['target'] = pred_qda

oof = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for k in tqdm_notebook(range(512)):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==k] 

    train2p = train2.copy(); idx1 = train2.index 

    test2 = test[test['wheezy-copper-turtle-magic']==k]

    

    # ADD PSEUDO LABEL DATA

    test2p = test2[ (test2['target']<=0.01) | (test2['target']>=0.99) ].copy()

    test2p.loc[ test2p['target']>=0.5, 'target' ] = 1

    test2p.loc[ test2p['target']<0.5, 'target' ] = 0 

    train2p = pd.concat([train2p,test2p],axis=0)

    train2p.reset_index(drop=True,inplace=True)

    

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     

    train3p = sel.transform(train2p[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])

        

    # STRATIFIED K FOLD

    skf = StratifiedKFold(n_splits=25, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3p, train2p['target']):

        test_index3 = test_index[ test_index<len(train3) ] # ignore psuedo in oof

        

        # MODEL AND PREDICT WITH QDA

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        clf.fit(train3p[train_index,:],train2p.loc[train_index]['target'])

        oof[idx1[test_index3]] += clf.predict_proba(train3[test_index3,:])[:,1]

        preds[test2.index] += clf.predict_proba(test3)[:,1] / skf.n_splits

       

    #if k%64==0: print(k)

        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)

print('Pseudo Labeled QDA scores CV =',round(auc,5))
saved_targets = train['target'].values.copy()

train.loc[ abs(train['target']-oof)>0.9,'target'] = 1-train.loc[ abs(train['target']-oof)>0.9,'target']

CV = roc_auc_score(saved_targets,oof)

print(CV)
# INITIALIZE VARIABLES

test['target'] = pred_qda

oof = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for k in tqdm_notebook(range(512)):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==k] 

    train2p = train2.copy(); idx1 = train2.index 

    test2 = test[test['wheezy-copper-turtle-magic']==k]

    

    # ADD PSEUDO LABEL DATA

    test2p = test2[ (test2['target']<=0.01) | (test2['target']>=0.99) ].copy()

    test2p.loc[ test2p['target']>=0.5, 'target' ] = 1

    test2p.loc[ test2p['target']<0.5, 'target' ] = 0 

    train2p = pd.concat([train2p,test2p],axis=0)

    train2p.reset_index(drop=True,inplace=True)

    

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     

    train3p = sel.transform(train2p[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])

        

    # STRATIFIED K FOLD

    skf = StratifiedKFold(n_splits=25, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3p, train2p['target']):

        test_index3 = test_index[ test_index<len(train3) ] # ignore psuedo in oof

        

        # MODEL AND PREDICT WITH QDA

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        clf.fit(train3p[train_index,:],train2p.loc[train_index]['target'])

        oof[idx1[test_index3]] += clf.predict_proba(train3[test_index3,:])[:,1]

        preds[test2.index] += clf.predict_proba(test3)[:,1] / skf.n_splits

       

    #if k%64==0: print(k)

        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)

print('Pseudo Labeled QDA scores CV =',round(auc,5))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv', index=False)