import numpy as np

import pandas as pd

from sklearn import svm, neighbors, linear_model, neural_network

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn.neural_network import MLPClassifier

from sklearn.mixture import GaussianMixture

from sklearn.covariance import GraphicalLasso

from sklearn.svm import SVC, NuSVC

from sklearn.pipeline import Pipeline

from tqdm import tqdm_notebook

import warnings

import multiprocessing

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import manifold

from scipy.optimize import minimize  

warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

print(train.shape, test.shape)
def get_mean_cov(x,y):

    ones = (y==1).astype(bool)

    x2 = x[ones]

    gmm_1 = GaussianMixture(n_components=2)

    gmm_1.fit(x2)

    m1 = gmm_1.means_

    p1 = gmm_1.precisions_



    zeros = (y==0).astype(bool)

    x2b = x[zeros]

    gmm_0 = GaussianMixture(n_components=2)

    gmm_0.fit(x2b)

    m2 = gmm_0.means_

    p2 = gmm_0.precisions_



    ms = np.concatenate((m1,m2),axis=0)

    ps = np.concatenate((p1,p2),axis=0)

    

    return ms,ps
np.random.seed(42)

oof_gmm = np.zeros(len(train))

preds_gmm = np.zeros(len(test))



# 512 models

for i in tqdm_notebook(range(512)):



    train2 = train[train['wheezy-copper-turtle-magic'] == i]

    test2 = test[test['wheezy-copper-turtle-magic'] == i]

    idx1 = train2.index

    idx2 = test2.index

    train2.reset_index(drop=True, inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    pipe = Pipeline([('vt', VarianceThreshold(threshold=2)),

                     ('scaler', StandardScaler())])

    data2 = pipe.fit_transform(data[cols])

    train3 = data2[:train2.shape[0]]

    test3 = data2[train2.shape[0]:]

    

    skf = StratifiedKFold(n_splits=11, random_state=2019,shuffle=True)



    for train_index, test_index in skf.split(train2, train2['target']):



        # MODEL AND PREDICT WITH QDA

        ms, ps = get_mean_cov(train3[train_index,:],train2.loc[train_index]['target'].values)



        gmm = GaussianMixture(n_components=4, init_params='kmeans', covariance_type='full', max_iter=100, n_init=1,means_init=ms, precisions_init=ps)

        gmm.fit(np.concatenate([train3,test3],axis = 0))

        

        prob = gmm.predict_proba(train3[test_index,:])

        prob_class = np.zeros(prob.shape[0])



        for j in range(prob.shape[0]):

            if(np.argmax(prob,axis=1)[j] in {0, 1}):

                prob_class[j] = np.max(prob,axis=1)[j]

            else:

                prob_class[j] = 1 - np.max(prob,axis=1)[j]



        oof_gmm[idx1[test_index]] = prob_class

        

        prob_test = gmm.predict_proba(test3)

        prob_test_class = np.zeros(prob_test.shape[0])

        for j in range(prob_test.shape[0]):

            if(np.argmax(prob_test,axis=1)[j] in {0, 1}):

                prob_test_class[j] = np.max(prob_test,axis=1)[j]

            else:

                prob_test_class[j] = 1 - np.max(prob_test,axis=1)[j]



        preds_gmm[idx2] += prob_test_class / skf.n_splits

        

auc = roc_auc_score(train['target'], oof_gmm)

print(f'AUC: {auc:.5}')

#0.96916
for itr in range(3):

    test['target'] = preds_gmm

    test.loc[test['target'] > 0.96, 'target'] = 1

    test.loc[test['target'] < 0.14, 'target'] = 0

    usefull_test = test[(test['target'] == 1) | (test['target'] == 0)]

    new_train = pd.concat([train, usefull_test]).reset_index(drop=True)

    print(usefull_test.shape[0], "Test Records added for iteration : ", itr)

    

    new_train.loc[oof_gmm > 0.98, 'target'] = 1

    new_train.loc[oof_gmm < 0.02, 'target'] = 0

    

    oof_gmm2 = np.zeros(len(train))

    preds_gmm = np.zeros(len(test))



    for i in tqdm_notebook(range(512)):



        train2 = train[train['wheezy-copper-turtle-magic'] == i]

        test2 = test[test['wheezy-copper-turtle-magic'] == i]

        idx1 = train2.index

        idx2 = test2.index

        train2.reset_index(drop=True, inplace=True)



        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

        pipe = Pipeline([('vt', VarianceThreshold(threshold=2)),

                         ('scaler', StandardScaler())])

        data2 = pipe.fit_transform(data[cols])

        train3 = data2[:train2.shape[0]]

        test3 = data2[train2.shape[0]:]



        skf = StratifiedKFold(n_splits=11, random_state=2019,shuffle=True)



        for train_index, test_index in skf.split(train2, train2['target']):

            

            oof_test_index = [t for t in test_index if t < len(idx1)]



            # MODEL AND PREDICT WITH QDA

            ms, ps = get_mean_cov(train3[train_index,:],train2.loc[train_index]['target'].values)



            gmm = GaussianMixture(n_components=4, init_params='kmeans', covariance_type='full', max_iter=100, n_init=1,means_init=ms, precisions_init=ps)

            gmm.fit(np.concatenate([train3,test3],axis = 0))

            

            if len(oof_test_index) > 0:

                prob = gmm.predict_proba(train3[oof_test_index,:])

                prob_class = np.zeros(prob.shape[0])



            for j in range(prob.shape[0]):

                if(np.argmax(prob,axis=1)[j] in {0, 1}):

                    prob_class[j] = np.max(prob,axis=1)[j]

                else:

                    prob_class[j] = 1 - np.max(prob,axis=1)[j]



            oof_gmm2[idx1[test_index]] = prob_class



            prob_test = gmm.predict_proba(test3)

            prob_test_class = np.zeros(prob_test.shape[0])

            for j in range(prob_test.shape[0]):

                if(np.argmax(prob_test,axis=1)[j] in {0, 1}):

                    prob_test_class[j] = np.max(prob_test,axis=1)[j]

                else:

                    prob_test_class[j] = 1 - np.max(prob_test,axis=1)[j]



            preds_gmm[idx2] += prob_test_class / skf.n_splits



    auc = roc_auc_score(train['target'], oof_gmm)

    print(f'AUC: {auc:.5}')

    #0.97256
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds_gmm

sub.to_csv('submission.csv',index=False)