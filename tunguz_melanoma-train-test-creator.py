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
le = preprocessing.LabelEncoder()



train.sex = le.fit_transform(train.sex)

train.anatom_site_general_challenge = le.fit_transform(train.anatom_site_general_challenge)

test.sex = le.fit_transform(test.sex)

test.anatom_site_general_challenge = le.fit_transform(test.anatom_site_general_challenge)
feature_names = ['sex','age_approx','anatom_site_general_challenge','w','h']

ycol = ['target']
train[feature_names + ycol].to_csv('train_meta_size.csv', index=False)

test[feature_names ].to_csv('test_meta_size.csv', index=False)
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
test.head()
train.head()
np.unique(train.diagnosis.values, return_counts=True)
cols = ['sex', 'age_approx', 'anatom_site_general_challenge']



train_test = train[cols].append(test[cols])
train_test.shape
train_test['age_approx'].mean()
train_test['age_approx'] = train_test['age_approx'].fillna(train_test['age_approx'].mean())#float

train_test['sex'] = train_test['sex'].fillna(train_test['sex'].value_counts().index[0])

train_test['anatom_site_general_challenge'] = train_test['anatom_site_general_challenge'].fillna(train_test['anatom_site_general_challenge'].value_counts().index[0])
train[cols] = train_test[:train.shape[0]][cols].values

test[cols] = train_test[train.shape[0]:][cols].values
test.head()
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
le = preprocessing.LabelEncoder()



le.fit(train_test.sex)



train.sex = le.transform(train.sex)

test.sex = le.transform(test.sex)



le = preprocessing.LabelEncoder()



le.fit(train_test.anatom_site_general_challenge)



train.anatom_site_general_challenge = le.transform(train.anatom_site_general_challenge)

test.anatom_site_general_challenge = le.transform(test.anatom_site_general_challenge)

train.head()
test.head()
train[feature_names + ycol].to_csv('train_meta_size_2.csv', index=False)

test[feature_names ].to_csv('test_meta_size_2.csv', index=False)
ycol
oof_c = pd.read_csv('../input/triple-stratified-kfold-with-tfrecords/oof.csv')

submission_c = pd.read_csv('../input/triple-stratified-kfold-with-tfrecords/submission.csv')

oof_c.head()
del oof_c['target']

oof_c.head()
oof_c.shape
train.shape
train_2 = train[train['image_name'].isin(oof_c['image_name'].values)]
train_2 = train_2.merge(oof_c, on='image_name')
feature_names.append('pred')
submission_c.head()
test['pred'] = submission_c['target']

test.head()
ycol
train_2[feature_names + ['fold'] + ycol].to_csv('train_meta_size_3.csv', index=False)

test[feature_names ].to_csv('test_meta_size_3.csv', index=False)
train_2[feature_names + ['fold'] + ycol].head()
test[feature_names]
train_32 = np.load('../input/siimisic-melanoma-resized-images/x_train_32.npy')/255

test_32 = np.load('../input/siimisic-melanoma-resized-images/x_test_32.npy')/255
train_32 = train_32.reshape((train_32.shape[0], 32*32*3))

test_32 = test_32.reshape((test_32.shape[0], 32*32*3))
columns = [f'c_{i}' for i in range(3072)]
train_32 = pd.DataFrame(data = train_32, columns=columns)

test_32 = pd.DataFrame(data = test_32, columns=columns)
train_32['target'] = train['target']
train_32.to_csv('train_32.csv', index=False)

test_32.to_csv('test_32.csv', index=False)

np.save('columns_32', columns)