import sys

!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import numpy as np

import pandas as pd

import cudf

import cupy as cp

from cuml.neighbors import KNeighborsClassifier

from cuml.linear_model import LogisticRegression

from cuml import RandomForestClassifier

from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import roc_auc_score, log_loss

import gc
test_InceptionResNetV2 = np.load('../input/melanoma-pretrained-embeddings/Pretrained/test_InceptionResNetV2.npy')

train_InceptionResNetV2 = np.load('../input/melanoma-pretrained-embeddings/Pretrained/train_InceptionResNetV2.npy')
train_0 = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

target = train_0.target.values
train_lr_oof_0 = np.zeros((train_InceptionResNetV2.shape[0], ))

test_lr_preds_0 = 0



n_splits = 8

n_seeds = 5



for ii in range(n_seeds):

    kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



    for jj, (train_index, val_index) in enumerate(kf.split(train_InceptionResNetV2)):

        print("Fitting fold", jj+1)

        train_features = train_InceptionResNetV2[train_index]

        train_target = target[train_index]



        val_features = train_InceptionResNetV2[val_index]

        val_target = target[val_index]



        model = LogisticRegression(C=1.7, max_iter=120)

        model.fit(train_features, train_target)

        



        val_pred = model.predict_proba(val_features)[:,1]

        test_lr_preds_0 += model.predict_proba(test_InceptionResNetV2)[:,1]/(n_splits*n_seeds)

        train_lr_oof_0[val_index] += val_pred/n_seeds

        print("Fold AUC:", roc_auc_score(val_target, val_pred))

        del train_features, train_target, val_features, val_target

        gc.collect()

    

print(roc_auc_score(target, train_lr_oof_0))
train_knn_oof_0 = np.zeros((train_InceptionResNetV2.shape[0], ))

test_knn_preds_0 = 0



n_splits = 8



n_seeds = 5



for ii in range(n_seeds):

    kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



    for jj, (train_index, val_index) in enumerate(kf.split(train_InceptionResNetV2)):

        print("Fitting fold", jj+1)

        train_features = train_InceptionResNetV2[train_index]

        train_target = target[train_index]



        val_features = train_InceptionResNetV2[val_index]

        val_target = target[val_index]



        model = KNeighborsClassifier(n_neighbors=170)

        model.fit(train_features, train_target)

        val_pred = model.predict_proba(val_features)[:,1]

        test_knn_preds_0 += model.predict_proba(test_InceptionResNetV2)[:,1]/(n_splits*n_seeds)

        train_knn_oof_0[val_index] += val_pred/n_seeds

        print("Fold AUC:", roc_auc_score(val_target, val_pred))

        del train_features, train_target, val_features, val_target

        gc.collect()



print(roc_auc_score(target, train_knn_oof_0))
train_rfc_oof_0 = np.zeros((train_InceptionResNetV2.shape[0], ))

test_rfc_preds_0 = 0



cu_rf_params = {'n_estimators': 1000,

    'max_depth': 17,

    'n_bins': 15,

    'n_streams': 8

}



n_splits = 8

kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



for jj, (train_index, val_index) in enumerate(kf.split(train_InceptionResNetV2)):

    print("Fitting fold", jj+1)

    train_features = train_InceptionResNetV2[train_index]

    train_target = target[train_index]

    

    val_features = train_InceptionResNetV2[val_index]

    val_target = target[val_index]

    

    model = RandomForestClassifier(**cu_rf_params)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)[:,1]

    test_rfc_preds_0 += model.predict_proba(test_InceptionResNetV2)[:,1]/n_splits

    train_rfc_oof_0[val_index] = val_pred

    print("Fold AUC:", roc_auc_score(val_target, val_pred))

    del train_features, train_target, val_features, val_target

    gc.collect()

    

print(roc_auc_score(target, train_rfc_oof_0))
0.8370434403843127
sample_submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
sample_submission['target'] = test_lr_preds_0

sample_submission.to_csv('InceptionResNetV2_pretrained_LR.csv', index=False)

sample_submission.head()
sample_submission['target'] = test_knn_preds_0

sample_submission.to_csv('InceptionResNetV2_pretrained_KNN.csv', index=False)

sample_submission.head()
sample_submission['target'] = test_rfc_preds_0

sample_submission.to_csv('InceptionResNetV2_pretrained_RFC.csv', index=False)

sample_submission.head()
print(roc_auc_score(target, 0.9*train_knn_oof_0+0.1*train_lr_oof_0))
print(roc_auc_score(target, 0.5*train_knn_oof_0+0.5*train_lr_oof_0))
print(roc_auc_score(target, (0.45*train_knn_oof_0+0.45*train_lr_oof_0+0.1*train_rfc_oof_0)))
sample_submission['target'] = 0.9*test_knn_preds_0+0.1*test_lr_preds_0

sample_submission.to_csv('InceptionResNetV2_pretrained.csv', index=False)

sample_submission.head()
sample_submission['target'] = 0.5*test_knn_preds_0+0.5*test_lr_preds_0

sample_submission.to_csv('InceptionResNetV2_pretrained_2.csv', index=False)

sample_submission.head()
sample_submission['target'] = (0.45*test_knn_preds_0+0.45*test_lr_preds_0+0.1*test_rfc_preds_0)

sample_submission.to_csv('InceptionResNetV2_pretrained_3.csv', index=False)

sample_submission.head()
sample_submission['target'] = (0.4*test_knn_preds_0+0.4*test_lr_preds_0+0.2*test_rfc_preds_0)

sample_submission.to_csv('InceptionResNetV2_pretrained_4.csv', index=False)

sample_submission.head()