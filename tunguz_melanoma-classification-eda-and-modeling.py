

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import json

import math

import cv2

import PIL

from PIL import Image



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

from sklearn.decomposition import PCA

import os

import imagesize



%matplotlib inline
import os

print(os.listdir("../input/siim-isic-melanoma-classification"))
#Loading Train and Test Data

train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")

test = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")

print("{} images in train set.".format(train.shape[0]))

print("{} images in test set.".format(test.shape[0]))
train.head()
test.head()
np.mean(train.target)
plt.figure(figsize=(12, 5))

plt.hist(train['age_approx'].values, bins=200)

plt.title('Histogram age_approx counts in train')

plt.xlabel('Value')

plt.ylabel('Count')

plt.show()
images = []

for i, image_id in enumerate(tqdm(train['image_name'].head(10))):

    im = Image.open(f'../input/siim-isic-melanoma-classification/jpeg/train/{image_id}.jpg')

    im = im.resize((128, )*2, resample=Image.LANCZOS)

    images.append(im)

    
images[0]
images[1]
images[3]
plt.figure(figsize=(12, 5))

plt.hist(test['age_approx'].values, bins=200)

plt.title('Histogram age_approx counts in test')

plt.xlabel('Value')

plt.ylabel('Count')

plt.show()
x_train_32 = np.load('../input/siimisic-melanoma-resized-images/x_train_32.npy')

x_test_32 = np.load('../input/siimisic-melanoma-resized-images/x_test_32.npy')
x_train_32.shape
x_train_32 = x_train_32.reshape((x_train_32.shape[0], 32*32*3))

x_train_32.shape
x_test_32 = x_test_32.reshape((x_test_32.shape[0], 32*32*3))

x_test_32.shape
y = train.target.values
train_oof = np.zeros((x_train_32.shape[0], ))

test_preds = 0

train_oof.shape
n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):

    print("Fitting fold", jj+1)

    train_features = x_train_32[train_index]

    train_target = y[train_index]

    

    val_features = x_train_32[val_index]

    val_target = y[val_index]

    

    model = LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial', max_iter=60)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)[:,1]

    train_oof[val_index] = val_pred

    print("Fold AUC:", roc_auc_score(val_target, val_pred))

    test_preds += model.predict_proba(x_test_32)[:,1]/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()
print(roc_auc_score(y, train_oof))
train_oof_0_2 = np.zeros((x_train_32.shape[0], ))

test_preds_0_2 = 0



n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):

    print("Fitting fold", jj+1)

    train_features = x_train_32[train_index]

    train_target = y[train_index]

    

    val_features = x_train_32[val_index]

    val_target = y[val_index]

    

    model = LogisticRegression(C=5, solver='lbfgs', multi_class='multinomial', max_iter=80)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)[:,1]

    train_oof_0_2[val_index] = val_pred

    print("Fold AUC:", roc_auc_score(val_target, val_pred))

    test_preds_0_2 += model.predict_proba(x_test_32)[:,1]/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()

    

print(roc_auc_score(y, train_oof_0_2))
print(roc_auc_score(y, 0.95*train_oof+0.05*train_oof_0_2))
train['age_approx'].unique()
train['sex'] = (train['sex'].values == 'male')*1

test['sex'] = (test['sex'].values == 'male')*1

train.head()
test.head()
train['sex'].mean()
test['sex'].mean()
train['age_approx'].mean()
test['age_approx'].mean()
train['age_approx'] = train['age_approx'].fillna(train['age_approx'].mean())

test['age_approx'] = test['age_approx'].fillna(test['age_approx'].mean())
x_train_32 = np.hstack([x_train_32, train['sex'].values.reshape(-1,1), train['age_approx'].values.reshape(-1,1)])

x_test_32 = np.hstack([x_test_32, test['sex'].values.reshape(-1,1), test['age_approx'].values.reshape(-1,1)])
train_oof_2 = np.zeros((x_train_32.shape[0], ))

test_preds_2 = 0





n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):

    print("Fitting fold", jj+1)

    train_features = x_train_32[train_index]

    train_target = y[train_index]

    

    val_features = x_train_32[val_index]

    val_target = y[val_index]

    

    model = LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial', max_iter=50)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)[:,1]

    train_oof_2[val_index] = val_pred

    print("Fold AUC:", roc_auc_score(val_target, val_pred))

    test_preds_2 += model.predict_proba(x_test_32)[:,1]/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()
print(roc_auc_score(y, train_oof_2))
print(roc_auc_score(y, 0.8*train_oof_2+0.2*train_oof))
print(roc_auc_score(y, 0.5*train_oof_2+0.5*train_oof))
test_preds.max()
test_preds_2.max()
train_oof_2_2 = np.zeros((x_train_32.shape[0], ))

test_preds_2_2 = 0





n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):

    print("Fitting fold", jj+1)

    train_features = x_train_32[train_index]

    train_target = y[train_index]

    

    val_features = x_train_32[val_index]

    val_target = y[val_index]

    

    model = LogisticRegression(C=5, solver='lbfgs', multi_class='multinomial', max_iter=80)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)[:,1]

    train_oof_2_2[val_index] = val_pred

    print("Fold AUC:", roc_auc_score(val_target, val_pred))

    test_preds_2_2 += model.predict_proba(x_test_32)[:,1]/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()
print(roc_auc_score(y, train_oof_2_2))
train['anatom_site_general_challenge'].unique()
test['anatom_site_general_challenge'].unique()
train['anatom_site_general_challenge'].mode()
test['anatom_site_general_challenge'].mode()
train['anatom_site_general_challenge'].fillna(train['anatom_site_general_challenge'].mode(), inplace=True)

test['anatom_site_general_challenge'].fillna(test['anatom_site_general_challenge'].mode(), inplace=True)
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype(str)

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype(str)
test['anatom_site_general_challenge'].isnull().sum()
x_train_32 = np.hstack([x_train_32, pd.get_dummies(train['anatom_site_general_challenge']).values])

x_test_32 = np.hstack([x_test_32, pd.get_dummies(test['anatom_site_general_challenge']).values])
train_oof_3 = np.zeros((x_train_32.shape[0], ))

test_preds_3 = 0





n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):

    print("Fitting fold", jj+1)

    train_features = x_train_32[train_index]

    train_target = y[train_index]

    

    val_features = x_train_32[val_index]

    val_target = y[val_index]

    

    model = LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial', max_iter=60)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)[:,1]

    train_oof_3[val_index] = val_pred

    print("Fold AUC:", roc_auc_score(val_target, val_pred))

    test_preds_3 += model.predict_proba(x_test_32)[:,1]/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()
print(roc_auc_score(y, train_oof_3))
train_oof_4 = np.zeros((x_train_32.shape[0], ))

test_preds_4 = 0





n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):

    print("Fitting fold", jj+1)

    train_features = x_train_32[train_index]

    train_target = y[train_index]

    

    val_features = x_train_32[val_index]

    val_target = y[val_index]

    

    model = LogisticRegression(C=5, max_iter=80)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)[:,1]

    train_oof_4[val_index] = val_pred

    print("Fold AUC:", roc_auc_score(val_target, val_pred))

    test_preds_4 += model.predict_proba(x_test_32)[:,1]/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()
print(roc_auc_score(y, train_oof_4))
print(roc_auc_score(y, 0.25*train_oof_2+0.25*train_oof+0.25*train_oof_3+0.25*train_oof_4))
1
im_shape_test = []

im_shape_train = []



for i in range(train.shape[0]):

    im_shape_train.append(imagesize.get('../input/siim-isic-melanoma-classification/jpeg/train/'+train['image_name'][i]+'.jpg'))

for i in range(test.shape[0]):

    im_shape_test.append(imagesize.get('../input/siim-isic-melanoma-classification/jpeg/test/'+test['image_name'][i]+'.jpg'))

    



train['dim'] = im_shape_train

test['dim'] = im_shape_test
train['dim'] == (6000,4000)
train['dim'] == (1872,1053)
(train['dim'] != (6000,4000)) & (train['dim'] != (1872,1053))
train['dim_1'] = (train['dim'] == (6000,4000))

train['dim_1'] = train['dim_1'].values*1

train['dim_2'] = (train['dim'] == (1872,1053))

train['dim_2'] = train['dim_2'].values*1

train['dim_3'] = (train['dim'] != (6000,4000)) & (train['dim'] != (1872,1053))

train['dim_3'] = train['dim_3'].values*1

train['dim_3']
test['dim_1'] = (test['dim'] == (6000,4000))

test['dim_1'] = test['dim_1'].values*1

test['dim_2'] = (test['dim'] == (1872,1053))

test['dim_2'] = test['dim_2'].values*1

test['dim_3'] = (test['dim'] != (6000,4000)) & (test['dim'] != (1872,1053))

test['dim_3'] = test['dim_3'].values*1

test['dim_3']
x_train_32 = np.hstack([x_train_32, train['dim_1'].values.reshape(-1,1), train['dim_2'].values.reshape(-1,1), train['dim_3'].values.reshape(-1,1)])

x_test_32 = np.hstack([x_test_32, test['dim_1'].values.reshape(-1,1), test['dim_2'].values.reshape(-1,1), test['dim_3'].values.reshape(-1,1)])
train_oof_4_2 = np.zeros((x_train_32.shape[0], ))

test_preds_4_2 = 0





n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):

    print("Fitting fold", jj+1)

    train_features = x_train_32[train_index]

    train_target = y[train_index]

    

    val_features = x_train_32[val_index]

    val_target = y[val_index]

    

    model = LogisticRegression(C=0.9, max_iter=50)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)[:,1]

    train_oof_4_2[val_index] = val_pred

    print("Fold AUC:", roc_auc_score(val_target, val_pred))

    test_preds_4_2 += model.predict_proba(x_test_32)[:,1]/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()

    

print(roc_auc_score(y, train_oof_4_2))
0.8262080489449671
%%time

pca = PCA(n_components=0.99)

pca.fit(x_train_32)
pca.n_components_
x_train_32 = pca.transform(x_train_32)

x_test_32 = pca.transform(x_test_32)
train_oof_5 = np.zeros((x_train_32.shape[0], ))

test_preds_5 = 0





n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):

    print("Fitting fold", jj+1)

    train_features = x_train_32[train_index]

    train_target = y[train_index]

    

    val_features = x_train_32[val_index]

    val_target = y[val_index]

    

    model = LogisticRegression(C=0.1, max_iter=6)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)[:,1]

    train_oof_5[val_index] = val_pred

    print("Fold AUC:", roc_auc_score(val_target, val_pred))

    test_preds_5 += model.predict_proba(x_test_32)[:,1]/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()

    

print(roc_auc_score(y, train_oof_5))
0.7910534268464863
print(roc_auc_score(y, 0.988*(0.27*train_oof_2+0.27*train_oof+0.27*train_oof_3+0.19*train_oof_4)+0.012*train_oof_5))
print(roc_auc_score(y, 1.082*(0.99*(0.25*train_oof_2+0.25*train_oof+0.25*train_oof_3+0.25*train_oof_4)+0.01*train_oof_5)-0.082*(train_oof_0_2+train_oof_2_2)/2))
x_train_64 = np.load('../input/siimisic-melanoma-resized-images/x_train_64.npy')

x_test_64 = np.load('../input/siimisic-melanoma-resized-images/x_test_64.npy')
x_train_64 = x_train_64.reshape((x_train_64.shape[0], 64*64*3))

x_train_64.shape
x_test_64 = x_test_64.reshape((x_test_64.shape[0], 64*64*3))

x_test_64.shape
train_oof_6 = np.zeros((x_train_64.shape[0], ))

test_preds_6 = 0





n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



for jj, (train_index, val_index) in enumerate(kf.split(x_train_64)):

    print("Fitting fold", jj+1)

    train_features = x_train_64[train_index]

    train_target = y[train_index]

    

    val_features = x_train_64[val_index]

    val_target = y[val_index]

    

    model = LogisticRegression(C=0.1, max_iter=45)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)[:,1]

    train_oof_6[val_index] = val_pred

    print("Fold AUC:", roc_auc_score(val_target, val_pred))

    test_preds_6 += model.predict_proba(x_test_64)[:,1]/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()

    

print(roc_auc_score(y, train_oof_6))
0.8213209504598062
print(roc_auc_score(y, 0.73*(1.1*(0.99*(0.25*train_oof_2+0.25*train_oof+0.25*train_oof_3+0.25*train_oof_4)+0.01*train_oof_5)-0.1*(train_oof_0_2+train_oof_2_2)/2)+0.27*train_oof_6))
print(roc_auc_score(y, 0.9*(0.73*(1.1*(0.99*(0.25*train_oof_2+0.25*train_oof+0.25*train_oof_3+0.25*train_oof_4)+0.01*train_oof_5)-0.1*(train_oof_0_2+train_oof_2_2)/2)+0.27*train_oof_6)+

                   0.1*train_oof_4_2))
0.8285058171399994
sample_submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

sample_submission.head()
sample_submission['target'] = 0.9*(0.73*(1.1*(0.99*(0.25*test_preds+0.25*test_preds_2+0.25*test_preds_3+0.25*test_preds_4)+0.015*test_preds_5)- 0.1*(0.5*test_preds_0_2+0.5*test_preds_2_2))+0.27*test_preds_6)+0.1*test_preds_4_2

sample_submission.to_csv('submission_32x32_64x64_lr.csv', index=False)
sample_submission['target'].max()
sample_submission['target'].min()