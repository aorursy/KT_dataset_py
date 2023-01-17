import time

import numpy as np

import pandas as pd

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



from sklearn.covariance import ShrunkCovariance, GraphicalLasso, LedoitWolf, OAS, MinCovDet

from sklearn.mixture import GaussianMixture



import warnings

warnings.filterwarnings('ignore')
# Params

n_models = 4



n_splits = 12

seed_split = 123

seed_gm = 987



cov_type = 'GL_0.2'

n_init = 4

init_params = 'random'

n_clusters_per_class = 3
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train = train.reset_index(drop=True)



print('train:', train.shape)

print('test:', test.shape)
def get_cov_estimator(cov_type):

    if cov_type == 'LW':

        model = LedoitWolf()

    elif cov_type == 'OAS':

        model = OAS()

    elif cov_type == 'MCD':

        model = MinCovDet()

    elif cov_type[:2] == 'SC':

        shrinkage = float(cov_type.split('_')[1])

        model = ShrunkCovariance(shrinkage=shrinkage)

    elif cov_type[:2] == 'GL':

        alpha = float(cov_type.split('_')[1])

        model = GraphicalLasso(alpha=alpha)

    return model





def get_mean_cov(x, y, n_clusters_per_class=1, cov_type='LW'):

    model = get_cov_estimator(cov_type)

    ones = (y == 1).astype(bool)

    x2 = x[ones]

    model.fit(x2)

    p1 = model.precision_

    m1 = model.location_



    onesb = (y == 0).astype(bool)

    x2b = x[onesb]

    model.fit(x2b)

    p2 = model.precision_

    m2 = model.location_



    ms = np.stack([m1] * n_clusters_per_class + [m2] * n_clusters_per_class)

    ps = np.stack([p1] * n_clusters_per_class + [p2] * n_clusters_per_class)

    return ms, ps
# INITIALIZE VARIABLES

cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')



oof_avg = np.zeros(len(train))

preds = np.zeros(len(test))



for j in range(n_models):

    t1 = time.time()

    oof = np.zeros(len(train))



    # BUILD 512 SEPARATE MODELS

    for i in range(512):

        train2 = train[train['wheezy-copper-turtle-magic'] == i]

        test2 = test[test['wheezy-copper-turtle-magic'] == i]

        idx1 = train2.index

        idx2 = test2.index

        train2.reset_index(drop=True, inplace=True)



        # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

        sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

        train3 = sel.transform(train2[cols])

        test3 = sel.transform(test2[cols])



        # STRATIFIED K-FOLD

        skf = StratifiedKFold(n_splits=n_splits,

                              random_state=seed_split + j,

                              shuffle=True)

        for train_idx, valid_idx in skf.split(train3, train2['target']):

            ms, ps = get_mean_cov(train3[train_idx, :],

                                  train2.loc[train_idx]['target'].values,

                                  n_clusters_per_class=n_clusters_per_class,

                                  cov_type=cov_type)



            gm = GaussianMixture(n_components=2 * n_clusters_per_class,

                                 init_params=init_params,

                                 covariance_type='full',

                                 max_iter=100,

                                 n_init=n_init,

                                 precisions_init=ps,

                                 random_state=seed_gm + i)

            gm.fit(np.concatenate([train3[train_idx, :], test3]))

            oof[idx1[valid_idx]] += gm.predict_proba(train3[valid_idx, :])[:, :2].sum(1)

            preds[idx2] += gm.predict_proba(test3)[:, :2].sum(1)

    oof_avg += oof

    auc_one = roc_auc_score(train['target'], oof)

    auc_avg = roc_auc_score(train['target'], oof_avg)

    print(f'  model:{j} AUC_avg:{auc_avg:.5f} AUC:{auc_one:.5f} time:{time.time() - t1:.1f}s')
preds /= n_models * n_splits

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv', index=False)