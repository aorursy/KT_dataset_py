import os

import pandas as pd

import numpy as np

import xgboost as xgb



from sklearn.preprocessing import MinMaxScaler
dir_file = '../input/siim-isic-melanoma-classification/jpeg'



train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

sub = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
train['sex'] = train['sex'].fillna('na')

train['age_approx'] = train['age_approx'].fillna(0)

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')



test['sex'] = test['sex'].fillna('na')

test['age_approx'] = test['age_approx'].fillna(0)

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')



L = 15

feat = ['sex','age_approx','anatom_site_general_challenge']



M = train.target.mean()

te = train.groupby(feat)['target'].agg(['mean','count']).reset_index()

te['ll'] = ((te['mean']*te['count'])+(M*L))/(te['count']+L)

del te['mean'], te['count']



test = test.merge( te, on=feat, how='left' )

test['ll'] = test['ll'].fillna(M)



meta = test.ll.values

sub['target'] = meta

sub.to_csv('submission_meta.csv', index=False)

sub.head()
train['sex'] = train['sex'].fillna('na')

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')



test['sex'] = test['sex'].fillna('na')

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')



train['sex'] = train['sex'].astype("category").cat.codes +1

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype("category").cat.codes +1



test['sex'] = test['sex'].astype("category").cat.codes +1

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype("category").cat.codes +1



age_approx = np.nanmean(np.concatenate([np.array(train['age_approx']), np.array(test['age_approx'])]))

train['age_approx'].fillna(age_approx, inplace = True)

test['age_approx'].fillna(age_approx, inplace = True)





x_train = train[['sex', 'age_approx','anatom_site_general_challenge']]

y_train = train['target']



x_test = test[['sex', 'age_approx','anatom_site_general_challenge']]



train_DMatrix = xgb.DMatrix(x_train, label= y_train)

test_DMatrix = xgb.DMatrix(x_test)



param = {

    'booster':'gbtree', 

    'eta': 0.3,

    'num_class': 2,

}



clf = xgb.XGBClassifier(n_estimators=1000, 

                        max_depth=8, 

                        objective='multi:softprob',

                        seed=0,  

                        nthread=-1, 

                        learning_rate=0.1, 

                        num_class = 2)



clf.fit(x_train, y_train)

meta_xgbc = clf.predict_proba(x_test)[:,1]



sub['target'] = meta_xgbc

sub.to_csv('submission_meta_xgbc.csv', index=False)

sub.head()
# The mode of a set of values is the value that appears most often.

train['age_approx'].fillna(train['age_approx'].mode().values[0], inplace = True)

test['age_approx'].fillna(test['age_approx'].mode().values[0], inplace = True)



# age_id

train['age_id_min']  = train['patient_id'].map(train.groupby(['patient_id']).age_approx.min())

train['age_id_max']  = train['patient_id'].map(train.groupby(['patient_id']).age_approx.max())



test['age_id_min']  = test['patient_id'].map(test.groupby(['patient_id']).age_approx.min())

test['age_id_max']  = test['patient_id'].map(test.groupby(['patient_id']).age_approx.max())



# sex_enc

train['sex'] = train['sex'].fillna('unknown')

train['sex'] = train['sex'].astype("category").cat.codes +1

test['sex'] = test['sex'].fillna('unknown')

test['sex'] = test['sex'].astype("category").cat.codes +1



# anatom_enc

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype("category").cat.codes +1

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype("category").cat.codes +1



# n_images

train['n_images'] = train.patient_id.map(train.groupby(['patient_id']).image_name.count())

test['n_images'] = test.patient_id.map(test.groupby(['patient_id']).image_name.count())



# image_size_scaled

train_images = train['image_name'].values

train_sizes = np.zeros(train_images.shape[0])

for i, img_path in enumerate(train_images):

    train_sizes[i] = os.path.getsize(os.path.join(dir_file, 'train', f'{img_path}.jpg'))

train['image_size'] = train_sizes





test_images = test['image_name'].values

test_sizes = np.zeros(test_images.shape[0])

for i, img_path in enumerate(test_images):

    test_sizes[i] = os.path.getsize(os.path.join(dir_file, 'test', f'{img_path}.jpg'))

test['image_size'] = test_sizes



scale = MinMaxScaler()

train['image_size_scaled'] = scale.fit_transform(train['image_size'].values.reshape(-1, 1))

test['image_size_scaled'] = scale.transform(test['image_size'].values.reshape(-1, 1))



# corr = train.corr(method = 'pearson')

# corr = corr.abs()

# corr.style.background_gradient(cmap='inferno')



features = [

    'age_approx',

    'age_id_min',

    'age_id_max',

    'sex',

    'anatom_site_general_challenge',

    'n_images',

    'image_size_scaled'

]



x_train = train[features]

y_train = train['target']



x_test = test[features]



# model = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

#              colsample_bynode=1, colsample_bytree=0.8, gamma=1, gpu_id=-1,

#              importance_type='gain', interaction_constraints=None,

#              learning_rate=0.002, max_delta_step=0, max_depth=10,

#              min_child_weight=1, missing=None, monotone_constraints=None,

#              n_estimators=700, n_jobs=-1, nthread=-1, num_parallel_tree=1,

#              objective='binary:logistic', random_state=0, reg_alpha=0,

#              reg_lambda=1, scale_pos_weight=1, silent=True, subsample=0.8,

#              tree_method=None, validate_parameters=False, verbosity=None)



# kfold = StratifiedKFold(n_splits=5, random_state=1001, shuffle=True)

# cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='roc_auc', verbose = 3)

# print(cv_results.mean())



xgb = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.8, gamma=1, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.002, max_delta_step=0, max_depth=10,

             min_child_weight=1, missing=None, monotone_constraints=None,

             n_estimators=700, n_jobs=-1, nthread=-1, num_parallel_tree=1,

             objective='binary:logistic', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, silent=True, subsample=0.8,

             tree_method=None, validate_parameters=False, verbosity=None)



xgb.fit(x_train, y_train)

meta_xgbr = xgb.predict(x_test)



sub = pd.DataFrame({'image_name': test.image_name.values, 'target': meta_xgbr})

sub.to_csv('submission_meta_xgbr.csv',index = False)