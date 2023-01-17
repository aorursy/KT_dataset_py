import numpy as np, pandas as pd, os

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import roc_auc_score



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# INITIALIZE VARIABLES

cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')

oof = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for i in range(512):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])

    

    # STRATIFIED K-FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3, train2['target']):

        

        # MODEL AND PREDICT WITH QDA

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)

print('QDA scores CV =',round(auc,5))
# INITIALIZE VARIABLES

test['target'] = preds

oof = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for k in range(512):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==k] 

    train2p = train2.copy(); idx1 = train2.index 

    test2 = test[test['wheezy-copper-turtle-magic']==k]

    

    # ADD PSEUDO LABELED DATA

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

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3p, train2p['target']):

        test_index3 = test_index[ test_index<len(train3) ] # ignore pseudo in oof

        

        # MODEL AND PREDICT WITH QDA

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        clf.fit(train3p[train_index,:],train2p.loc[train_index]['target'])

        oof[idx1[test_index3]] = clf.predict_proba(train3[test_index3,:])[:,1]

        preds[test2.index] += clf.predict_proba(test3)[:,1] / skf.n_splits

       

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)

print('Pseudo Labeled QDA scores CV =',round(auc,5))
from sklearn import svm, neighbors, linear_model, neural_network

from sklearn.svm import NuSVC

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.semi_supervised import LabelSpreading

from sklearn.covariance import ShrunkCovariance

from sklearn.mixture import GaussianMixture

from tqdm import tqdm_notebook

import warnings

warnings.filterwarnings('ignore')



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



oof_QDA = np.zeros(len(train))

preds_QDA = np.zeros(len(test))

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

for i in tqdm_notebook(range(512)):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])

    # STRATIFIED K-FOLD

    skf = StratifiedKFold(n_splits=25, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3, train2['target']):

        # MODEL AND PREDICT WITH QDA

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof_QDA[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds_QDA[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof_QDA)

print('QDA scores CV =',round(auc,8))

sub_QDA = pd.read_csv('../input/sample_submission.csv')

sub_QDA['target'] = preds_QDA

sub_QDA.to_csv('submission_QDA.csv',index=False)

oof_preds_QDA = train[['id', 'target']].copy()

oof_preds_QDA['target'] = oof_QDA

oof_preds_QDA.to_csv('oof_preds_QDA.csv', index = False)
test['target'] = preds_QDA

oof_QDA_PL = np.zeros(len(train))

preds_QDA_PL = np.zeros(len(test))

preds_QDA_PL_label = np.zeros(len(test))

n = 240

for k in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==k] 

    train2p = train2.copy(); idx1 = train2.index 

    test2 = test[test['wheezy-copper-turtle-magic']==k].sort_values(by = 'target')

    idx2 = test2.index

    # ADD PSEUDO LABELED DATA

    #Use Private as Pseudo Label to see LB

    test2p = pd.concat([test2[: n], test2[-n: ]], axis = 0)

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

    preds_QDA_PL_label_slice = preds_QDA_PL_label[list(idx2[: n]) + list(idx2[-n: ])]

    skf = StratifiedKFold(n_splits=25, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3p, train2p['target']):

        test_index3 = test_index[ test_index<len(train3) ] # ignore pseudo in oof

        test_index4 = test_index[ test_index>=len(train3) ]

        # MODEL AND PREDICT WITH QDA

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        clf.fit(train3p[train_index,:],train2p.loc[train_index]['target'])

        oof_QDA_PL[idx1[test_index3]] = clf.predict_proba(train3p[test_index3])[:,1]

        if(len(test_index4) > 0):

            preds_QDA_PL_label_slice[test_index4 - len(train3)] = clf.predict_proba(train3p[test_index4])[:, 1]

        preds_QDA_PL[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

    preds_QDA_PL_label[list(idx2[: n]) + list(idx2[-n: ])] = preds_QDA_PL_label_slice

preds_QDA_PL[preds_QDA_PL_label != 0] = preds_QDA_PL_label[preds_QDA_PL_label != 0]

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof_QDA_PL)

print('Pseudo Labeled QDA scores CV =',round(auc,8))

sub_QDA_PL = pd.read_csv('../input/sample_submission.csv')

sub_QDA_PL['target'] = preds_QDA_PL

sub_QDA_PL.to_csv('submission_QDA_PL.csv',index=False)

oof_preds_QDA_PL = train[['id', 'target']].copy()

oof_preds_QDA_PL['target'] = oof_QDA_PL

oof_preds_QDA_PL.to_csv('oof_preds_QDA_PL.csv', index = False)
def get_mean_cov(x,y):

    model = ShrunkCovariance()

    ones = (y==1).astype(bool)

    x2 = x[ones]

    model.fit(x2)

    p1 = model.precision_

    m1 = model.location_

    

    onesb = (y==0).astype(bool)

    x2b = x[onesb]

    model.fit(x2b)

    p2 = model.precision_

    m2 = model.location_

    

    ms = np.stack([m1,m2])

    ps = np.stack([p1,p2])

    return ms,ps

oof_GMM = np.zeros(len(train)) 

preds_GMM = np.zeros(len(test))

for i in tqdm_notebook(range(512)):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])

    # STRATIFIED K-FOLD

    skf = StratifiedKFold(n_splits=25, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3, train2['target']):

        # MODEL AND PREDICT WITH QDA

        ms, ps = get_mean_cov(train3[train_index,:],train2.loc[train_index]['target'].values)

        gm = GaussianMixture(n_components=2, init_params='random', covariance_type='full', tol=0.001,reg_covar=0.001, max_iter=100, n_init=1,means_init=ms, precisions_init=ps)

        gm.fit(np.concatenate([train3[train_index,:],test3],axis = 0))

        oof_GMM[idx1[test_index]] = gm.predict_proba(train3[test_index,:])[:,0]

        preds_GMM[idx2] += gm.predict_proba(test3)[:,0] / skf.n_splits

auc = roc_auc_score(train['target'],oof_GMM)

print('GMM scores CV =',round(auc,8))



sub_GMM = pd.read_csv('../input/sample_submission.csv')

sub_GMM['target'] = preds_GMM

sub_GMM.to_csv('submission_GMM.csv', index = False)

oof_preds_GMM = train[['id', 'target']].copy()

oof_preds_GMM['target'] = oof_GMM

oof_preds_GMM.to_csv('oof_preds_GMM.csv', index = False)
test['target'] = preds_QDA

oof_NuSVC = np.zeros(len(train)) 

preds_NuSVC = np.zeros(len(test))

preds_NuSVC_label = np.zeros(len(test))

n = 220

for i in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i].sort_values(by = 'target')

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    pca = PCA(svd_solver='full',n_components='mle')

    scaler = StandardScaler()

    data2 = scaler.fit_transform(pca.fit_transform(data[cols]))

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



    #Pseudo_Label

#Use Private as Pseudo Label to see LB

    test2p = pd.concat([test2[: n], test2[-n: ]], axis = 0)

    test2p.loc[ test2p['target']>=0.5, 'target' ] = 1

    test2p.loc[ test2p['target']<0.5, 'target' ] = 0

    train2p = pd.concat([train2, test2p], axis = 0)

    train2p.reset_index(drop=True,inplace=True)

    test5 = scaler.transform(pca.transform(test2p[cols]))

    train3p = np.concatenate([train3, test5])

    

    # STRATIFIED K FOLD (Using splits=25 scores 0.002 better but is slower)

    preds_NuSVC_label_slice = preds_NuSVC_label[list(idx2[: n]) + list(idx2[-n: ])]

    skf = StratifiedKFold(n_splits=25, random_state=42, shuffle = True)

    for train_index, test_index in skf.split(train3p, train2p['target']):

        test_index3 = test_index[test_index < len(train3)]

        test_index4 = test_index[test_index >= len(train3)]

        clf = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.75, coef0=0.053)

        clf.fit(train3p[train_index],train2p.loc[train_index]['target'])

        

        oof_NuSVC[idx1[test_index3]] = clf.predict_proba(train3p[test_index3])[:,1]

        if(len(test_index4) > 0):

            preds_NuSVC_label_slice[test_index4 - len(train3)] = clf.predict_proba(train3p[test_index4])[:, 1]

        preds_NuSVC[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

    preds_NuSVC_label[list(idx2[: n]) + list(idx2[-n: ])] = preds_NuSVC_label_slice

preds_NuSVC[preds_NuSVC_label != 0] = preds_NuSVC_label[preds_NuSVC_label != 0]



auc = roc_auc_score(train['target'],oof_NuSVC)

print('Pseudo Labeled NuSVC scores CV =',round(auc,8))

sub_NuSVC = pd.read_csv('../input/sample_submission.csv')

sub_NuSVC['target'] = preds_NuSVC

sub_NuSVC.to_csv('submission_NuSVC.csv', index = False)

oof_preds_NuSVC = train[['id', 'target']].copy()

oof_preds_NuSVC['target'] = oof_NuSVC

oof_preds_NuSVC.to_csv('oof_preds_NuSVC.csv', index = False)
test["target"] = preds_QDA

oof_LS = np.zeros(len(train)) 

preds_LS = np.zeros(len(test))

preds_LS_label = np.zeros(len(test))

n = 230

for k in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==k] 

    train2p = train2.copy(); idx1 = train2.index 

    test2 = test[test['wheezy-copper-turtle-magic']==k].sort_values(by = 'target')

    idx2 = test2.index

    # ADD PSEUDO LABELED DATA

    test2p = pd.concat([test2[: n], test2[-n: ]], axis = 0)

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

    preds_LS_label_slice = preds_LS_label[list(idx2[: n]) + list(idx2[-n: ])]

    skf = StratifiedKFold(n_splits = 25, random_state = 42, shuffle = True)

    for train_index, test_index in skf.split(train3p, train2p['target']):

        test_index3 = test_index[ test_index<len(train3) ] # ignore pseudo in oof

        test_index4 = test_index[ test_index>=len(train3) ]

        # MODEL AND PREDICT WITH QDA

        clf = LabelSpreading(gamma = 0.0125, kernel = 'rbf', max_iter = 10,alpha = 0.4,tol = 0.001)

        clf.fit(train3p[train_index,:],train2p.loc[train_index]['target'])

        oof_LS[idx1[test_index3]] = clf.predict_proba(train3p[test_index3])[:,1]

        if(len(test_index4) > 0):

            preds_LS_label_slice[test_index4 - len(train3)] = clf.predict_proba(train3p[test_index4])[:, 1]

        preds_LS[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

    preds_LS_label[list(idx2[: n]) + list(idx2[-n: ])] = preds_LS_label_slice

preds_LS[preds_LS_label != 0] = preds_LS_label[preds_LS_label != 0]



auc = roc_auc_score(train['target'],oof_LS)

print('Pseudo Labeled LS scores CV =',round(auc,8))

sub_LS = pd.read_csv('../input/sample_submission.csv')

sub_LS['target'] = preds_LS

sub_LS.to_csv('submission_LS.csv', index = False)

oof_preds_LS = train[['id', 'target']].copy()

oof_preds_LS['target'] = oof_LS

oof_preds_LS.to_csv('oof_preds_LS.csv', index = False)
oof_preds = pd.concat([oof_preds_QDA_PL['target'], oof_preds_NuSVC['target'], oof_preds_GMM['target'], oof_preds_LS['target']], axis = 1)

sub_preds = pd.concat([sub_QDA_PL['target'], sub_NuSVC['target'], sub_GMM['target'], sub_LS['target']], axis = 1)



oof_stacking = np.zeros(len(train)) 

preds_stacking = np.zeros(len(test))

skf = StratifiedKFold(n_splits=25, random_state=42, shuffle = True)

for train_index, test_index in skf.split(oof_preds, train['target']):

    lrr = linear_model.LogisticRegression()

    lrr.fit(oof_preds.loc[train_index], train.loc[train_index, 'target'])

    oof_stacking[test_index] = lrr.predict_proba(oof_preds.loc[test_index,:])[:,1]

    preds_stacking += lrr.predict_proba(sub_preds)[:,1] / skf.n_splits

auc = roc_auc_score(train['target'],oof_stacking)

print('Stacking scores CV =',round(auc,8))



sub_stacking = pd.read_csv('../input/sample_submission.csv')

sub_stacking['target'] = preds_stacking

sub_stacking.to_csv('submission_stacking.csv', index = False)

oof_preds_stacking = train[['id', 'target']].copy()

oof_preds_stacking['target'] = oof_stacking

oof_preds_stacking.to_csv('oof_preds_stacking.csv', index = False)