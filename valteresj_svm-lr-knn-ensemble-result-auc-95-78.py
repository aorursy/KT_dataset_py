import numpy as np, pandas as pd

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.svm import SVC,NuSVC

from sklearn.feature_selection import VarianceThreshold

from random import sample

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
oof = np.zeros(len(train))

preds = np.zeros(len(test))



oof_lr = np.zeros(len(train))

preds_lr = np.zeros(len(test))



oof_nusvc = np.zeros(len(train))

preds_nusvc = np.zeros(len(test))



oof_knn = np.zeros(len(train))

preds_knn = np.zeros(len(test))

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]





for i in range(512):

    

    

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    

    sel = VarianceThreshold(threshold=1).fit(train2[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])

        

    

    skf = StratifiedKFold(n_splits=11, random_state=42)

    

    for train_index, test_index in skf.split(train3, train2['target']):

        

        u=NuSVC(.6,'poly',4,'auto',.1,1,1)

        u.fit(train3[train_index,:],train2.loc[train_index]['target'])

        

        oof_nusvc[idx1[test_index]] = u.predict_proba(train3[test_index,:])[:,1]

        preds_nusvc[idx2] += u.predict_proba(test3)[:,1] / skf.n_splits

        

        k=KNeighborsClassifier(17,p=2.9)

        k.fit(train3[train_index,:],train2.loc[train_index]['target'])

        

        oof_knn[idx1[test_index]] = k.predict_proba(train3[test_index,:])[:,1]

        preds_knn[idx2] += k.predict_proba(test3)[:,1] / skf.n_splits

        

        

        logi = LogisticRegression('l2',1,.01,.05,1,solver='liblinear',max_iter=500)

        logi.fit(train3[train_index,:],train2.loc[train_index]['target'])

        

        oof_lr[idx1[test_index]] = logi.predict_proba(train3[test_index,:])[:,1]

        preds_lr[idx2] += logi.predict_proba(test3)[:,1] / skf.n_splits

        

        clf = SVC(probability=True,kernel='poly',degree=4,gamma='auto')

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        
data_tr=pd.DataFrame({'svm':oof,'svm_mod':oof_nusvc,'knn':oof_knn,'mlp':oof_lr})

data_ts=pd.DataFrame({'svm':preds,'svm_mod':preds_nusvc,'knn':preds_knn,'mlp':preds_lr})



index_trn=sample(list(data_tr.index),round(len(data_tr)*0.8))



logi1 = LogisticRegression('l2',1,.01,.05,1,solver='liblinear',max_iter=500)

logi1.fit(data_tr.loc[index_trn,:].values,train.loc[index_trn,'target'])

est_train=logi1.predict_proba(data_tr.drop(labels=index_trn,axis=0).values)[:,1]

est_tst=logi1.predict_proba(data_ts.values)[:,1]





auc = roc_auc_score(train['target'],oof_nusvc)

print('CV score =',round(auc,5))



auc = roc_auc_score(train.drop(labels=index_trn,axis=0)['target'],est_train)

print('CV score =',round(auc,5))

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = est_tst

sub.to_csv('submission.csv',index=False)