import numpy as np

import pandas as pd

import os

from PIL import Image

from tqdm import tqdm

from sklearn import preprocessing

from sklearn.model_selection import StratifiedKFold,cross_val_score

from sklearn.metrics import roc_auc_score



import lightgbm as lgb
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

sample = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
train['sex'] = train['sex'].fillna('na')

train['age_approx'] = train['age_approx'].fillna(0)

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')



test['sex'] = test['sex'].fillna('na')

test['age_approx'] = test['age_approx'].fillna(0)

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')
trn_images = train['image_name'].values

trn_sizes = np.zeros((trn_images.shape[0],2))

for i, img_path in enumerate(tqdm(trn_images)):

    img = Image.open(os.path.join('../input/siim-isic-melanoma-classification/jpeg/train/', f'{img_path}.jpg'))

    trn_sizes[i] = np.array([img.size[0],img.size[1]])
test_images = test['image_name'].values

test_sizes = np.zeros((test_images.shape[0],2))

for i, img_path in enumerate(tqdm(test_images)):

    img = Image.open(os.path.join('../input/siim-isic-melanoma-classification/jpeg/test/', f'{img_path}.jpg'))

    test_sizes[i] = np.array([img.size[0],img.size[1]])
train['w'] = trn_sizes[:,0]

train['h'] = trn_sizes[:,1]

test['w'] = test_sizes[:,0]

test['h'] = test_sizes[:,1]
train.head()
test.head()
le = preprocessing.LabelEncoder()



train.sex = le.fit_transform(train.sex)

train.anatom_site_general_challenge = le.fit_transform(train.anatom_site_general_challenge)

test.sex = le.fit_transform(test.sex)

test.anatom_site_general_challenge = le.fit_transform(test.anatom_site_general_challenge)
model = lgb.LGBMRegressor(n_estimators=500)
feature_names = ['sex','age_approx','anatom_site_general_challenge','w','h']

ycol = ['target']
test['target'] = 0



kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)



for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train[feature_names], train[ycol])):

    X_train = train.iloc[trn_idx][feature_names]

    Y_train = train.iloc[trn_idx][ycol]



    X_val = train.iloc[val_idx][feature_names]

    Y_val = train.iloc[val_idx][ycol]



    print('\nFold_{} Training ================================\n'.format(fold_id+1))



    lgb_model = model.fit(X_train,

                          Y_train,

                          eval_names=['train', 'valid'],

                          eval_set=[(X_train, Y_train), (X_val, Y_val)],

                          verbose=100,

                          eval_metric='auc',

                          early_stopping_rounds=100)



    pred_test = lgb_model.predict(test[feature_names], num_iteration=lgb_model.best_iteration_)

    

    test['target'] += pred_test / kfold.n_splits
lgb_model.feature_importances_
sample.target = test.target
sample.to_csv('submission.csv',index=False)