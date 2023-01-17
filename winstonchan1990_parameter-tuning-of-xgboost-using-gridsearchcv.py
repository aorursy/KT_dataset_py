## Load libraries
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.model_selection import GridSearchCV
import os
import xgboost as xgb 
from sklearn import metrics
from sklearn.metrics import r2_score
## Configurable options
REMOVE_ZERO_VAR = True
REMOVE_PERFECT_CORRELATED_VAR = False
USE_TSVD = True
USE_PCA = True
USE_ICA = True
USE_GRP = True
USE_SRP = True 
NCOMP = 10
USE_PROP_BINARY_VARS = True
CVFOLD = 5
SEED = 12345
## Read in and process train and test datasets
datadir = '../input'
trainFile = os.path.join(datadir,'train.csv')
testFile = os.path.join(datadir,'test.csv')
train = pd.read_csv(trainFile)
test = pd.read_csv(testFile) 

## remove non-feature columns from train/test datasets
y_train = train['y'].values
test_ids = test['ID'].values
train.drop(['ID','y'],axis=1,inplace=True)
test.drop(['ID'],axis=1,inplace=True)

## seperate categorical and binary variables
catg_vars = ['X'+str(i) for i in [0,1,2,3,4,5,6,8]]
binary_vars = [var for var in list(train.columns) if not var in catg_vars]

print('Number of variables : {}'.format(len(train.columns)))
print()
print('Categorical variables :\n')
print(', '.join(catg_vars))
print()
print('Binary variables :\n')
print(', '.join(binary_vars))
## LabelEncode categorical variables 
for c in catg_vars:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values) + list(test[c].values)) 
    train[c] = lbl.transform(list(train[c].values))
    test[c] = lbl.transform(list(test[c].values))
## Remove zero variance variables
zero_variance_vars = []
for c in train.columns:
    if len(set(train[c].values))==1:
        zero_variance_vars.append(c)
        
print(' ,'.join(zero_variance_vars))
## Remove perfectly correlated variables
from scipy.stats import pearsonr
vars_perfect_corr = []
var_pairs_perfect_corr = []

for i in range(0,len(train.columns)-1):
    for j in range(i+1,len(train.columns)):
        tmpcorr = pearsonr(train[train.columns[i]],train[train.columns[j]])[0]
        if(abs(tmpcorr)==1):
            vars_perfect_corr.append(train.columns[i])
            var_pairs_perfect_corr.append((train.columns[i],train.columns[j]))

for i,j in var_pairs_perfect_corr:
    print('({},{})'.format(i,j),end='\t')
## Dimension reduction 

# tSVD
tsvd = TruncatedSVD(n_components=NCOMP, random_state=SEED)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)
for i in range(1, NCOMP+1):
    train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]

# PCA
pca = PCA(n_components=NCOMP, random_state=SEED)
pca_results_train = pca.fit_transform(train)
pca_results_test = pca.transform(test)
for i in range(1, NCOMP+1):
    train['pca_' + str(i)] = pca_results_train[:,i-1]
    test['pca_' + str(i)] = pca_results_test[:, i-1]
    
# ICA
ica = FastICA(n_components=NCOMP, random_state=SEED)
ica_results_train = ica.fit_transform(train)
ica_results_test = ica.transform(test)
for i in range(1, NCOMP+1):
    train['ica_' + str(i)] = ica_results_train[:,i-1]
    test['ica_' + str(i)] = ica_results_test[:, i-1]
    
# GRP
grp = GaussianRandomProjection(n_components=NCOMP, eps=0.1, random_state=SEED)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)
for i in range(1, NCOMP+1):
    train['grp_' + str(i)] = grp_results_train[:,i-1]
    test['grp_' + str(i)] = grp_results_test[:, i-1]

# SRP
srp = SparseRandomProjection(n_components=NCOMP, dense_output=True, random_state=SEED)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)
for i in range(1, NCOMP+1):
    train['srp_' + str(i)] = srp_results_train[:,i-1]
    test['srp_' + str(i)] = srp_results_test[:, i-1]
# for each record, find the proportion of binary variables with value = 1
train['prop_binary_vars_equals_1'] = [np.mean(i) for i in train[binary_vars].values]
test['prop_binary_vars_equals_1'] = [np.mean(i) for i in test[binary_vars].values]
## Vars to drop
features = set(train.columns)
if REMOVE_ZERO_VAR:
    features = set(features)-set(zero_variance_vars)
if REMOVE_PERFECT_CORRELATED_VAR:
    features = set(features)-set(vars_perfect_corr)
if not USE_TSVD:
    features = set(features)-set(['tsvd_' + str(i) for i in range(1, NCOMP+1)])
if not USE_PCA:
    features = set(features)-set(['pca_' + str(i) for i in range(1, NCOMP+1)])
if not USE_ICA:
    features = set(features)-set(['ica_' + str(i) for i in range(1, NCOMP+1)])
if not USE_GRP:
    features = set(features)-set(['grp_' + str(i) for i in range(1, NCOMP+1)])
if not USE_SRP:
    features = set(features)-set(['srp_' + str(i) for i in range(1, NCOMP+1)])
if not USE_PROP_BINARY_VARS:
    features = set(features)-set(['prop_binary_vars_1'])
    
features = list(features)
print('Features selected : ')
print()
print(', '.join(features))

train = train[features]
test = test[features]
## Parameter grid (edit if neccessary)
params = {}
params['n_estimators'] = [800]
params['learning_rate'] = [0.0040]
params['max_depth'] = [3]
params['subsample'] = [0.7]
## Parameter tuning using Grid Search

# xgb regressor
xgb_model = xgb.XGBRegressor(
    silent=1,
    objective='reg:linear',
    base_score=np.mean(y_train),
    random_state=SEED
)

# create r2 scorer
r2_scorer = metrics.make_scorer(r2_score, greater_is_better = True)

# grid search
model = GridSearchCV(
    estimator = xgb_model,
    param_grid = params,
    scoring = r2_scorer,
    cv = CVFOLD,
    verbose=100,
    n_jobs=-1
)
## Run grid search
model.fit(train,y_train)
# best param config
print(model.best_score_)
print(model.best_params_)
# generate predictions using best model
final_model = model.best_estimator_

y_pred = final_model.predict(test)
submission = pd.DataFrame({
    'id': test_ids.astype(np.int32), 
    'y': y_pred
})
submission.head()
submission.to_csv('submission.csv',index=False)