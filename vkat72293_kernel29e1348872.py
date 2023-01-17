import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.svm import NuSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")    
train.head(5)
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

oof = np.zeros(len(train))

preds = np.zeros(len(test))
for i in range(512):

    new_train = train[train['wheezy-copper-turtle-magic']==i]

    new_test = test[test['wheezy-copper-turtle-magic']==i]

    index_1 = new_train.index; index_2 = new_test.index

    new_train.reset_index(drop = True, inplace = True)

    

    data = pd.concat([pd.DataFrame(new_train[cols]), pd.DataFrame(new_test[cols])])

    data2 = VarianceThreshold(threshold = 2).fit_transform(data[cols])

    train3 = data2[:new_train.shape[0]]

    test3 = data2[new_train.shape[0]:]

    

    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(new_train, new_train['target']):

            clf = QuadraticDiscriminantAnalysis(reg_param = 0.5)

            clf.fit(train3[train_index,:], new_train.loc[train_index]['target'])

            if len(test_index) > 0:

                oof[index_1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

            preds[index_2] += clf.predict_proba(test3)[:,1] / skf.n_splits

            

auc = roc_auc_score(train['target'], oof)

print(f'AUC: {auc:.5}')
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

oof_nusvc = np.zeros(len(train))

preds_nusvc = np.zeros(len(test))



for i in range(512):

    new_train = train[train['wheezy-copper-turtle-magic']==i]

    new_test = test[test['wheezy-copper-turtle-magic']==i]

    index_1 = new_train.index; index_2 = new_test.index

    new_train.reset_index(drop = True, inplace = True)

    

    data = pd.concat([pd.DataFrame(new_train[cols]), pd.DataFrame(new_test[cols])])

    data2 = VarianceThreshold(threshold = 2).fit_transform(data[cols])

    train3 = data2[:new_train.shape[0]]

    test3 = data2[new_train.shape[0]:]

    

    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(new_train, new_train['target']):

            clf = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=1, nu=0.6, coef0=0.75)

            clf.fit(train3[train_index,:], new_train.loc[train_index]['target'])

            if len(test_index) > 0:

                oof_nusvc[index_1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

            preds_nusvc[index_2] += clf.predict_proba(test3)[:,1] / skf.n_splits

            

auc = roc_auc_score(train['target'], oof_nusvc)

print(f'AUC: {auc:.5}')
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

oof_knn = np.zeros(len(train))

preds_knn = np.zeros(len(test))



for i in range(512):

    new_train = train[train['wheezy-copper-turtle-magic']==i]

    new_test = test[test['wheezy-copper-turtle-magic']==i]

    index_1 = new_train.index; index_2 = new_test.index

    new_train.reset_index(drop = True, inplace = True)

    

    data = pd.concat([pd.DataFrame(new_train[cols]), pd.DataFrame(new_test[cols])])

    data2 = StandardScaler().fit_transform(PCA(n_components=40, random_state=4).fit_transform(data[cols]))

    train3 = data2[:new_train.shape[0]]

    test3 = data2[new_train.shape[0]:]

    

    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(new_train, new_train['target']):            

            k=KNeighborsClassifier(17,p=2.9)

            k.fit(train3[train_index,:],new_train.loc[train_index]['target'])

            oof_knn[index_1[test_index]] = k.predict_proba(train3[test_index,:])[:,1]

            preds_knn[index_2] += k.predict_proba(test3)[:,1] / skf.n_splits

            

auc = roc_auc_score(train['target'], oof_knn)

print(f'AUC: {auc:.5}')
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = 0.9*preds+0.08*preds_nusvc+0.02*preds_knn

sub.to_csv('submission_grat.csv', index=False)