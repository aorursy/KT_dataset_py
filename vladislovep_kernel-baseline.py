import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore")



train = pd.read_csv("/kaggle/input/cft-shift-customer-retention/train.csv")

test = pd.read_csv("/kaggle/input/cft-shift-customer-retention/test.csv")
train
X = train.drop(columns=['churn'])

y = train['churn']
X['area'] = X['area'].fillna("UNKNOWN")

X['rev_Mean'] = X['rev_Mean'].fillna(-100000)

X['mou_Mean'] = X['mou_Mean'].fillna(-100000)

X['totmrc_Mean'] = X['totmrc_Mean'].fillna(-100000)

X['da_Mean'] = X['da_Mean'].fillna(-100000)

X['ovrmou_Mean'] = X['ovrmou_Mean'].fillna(-100000)

X['datovr_Mean'] = X['datovr_Mean'].fillna(-100000)

X['ovrrev_Mean'] = X['ovrrev_Mean'].fillna(-100000)

X['vceovr_Mean'] = X['vceovr_Mean'].fillna(-100000)

X['roam_Mean'] = X['roam_Mean'].fillna(-100000)

X['change_mou'] = X['change_mou'].fillna(-100000)

X['change_rev'] = X['change_rev'].fillna(-100000)

X['hnd_price'] = X['hnd_price'].fillna(-100000)

X['phones'] = X['phones'].fillna(-100000)

X['truck'] = X['truck'].fillna(-100000)

X['rv'] = X['rv'].fillna(-100000)

X['lor'] = X['lor'].fillna(-100000)

X['adults'] = X['adults'].fillna(-100000)

X['income'] = X['income'].fillna(-100000)

X['numbcars'] = X['numbcars'].fillna(-100000)

X['forgntvl'] = X['forgntvl'].fillna(-100000)

X['eqpdays'] = X['eqpdays'].fillna(-100000)

X['avg6mou'] = X['avg6mou'].fillna(-100000)

X['avg6qty'] = X['avg6qty'].fillna(-100000)

X['avg6rev'] = X['avg6rev'].fillna(-100000)







X['area'] = X['area'].fillna('UNKW')

X['rev_Mean'] = X['rev_Mean'].fillna(X['rev_Mean'].median())

X['mou_Mean'] = X['mou_Mean'].fillna(X['mou_Mean'].median())

X['totmrc_Mean'] = X['totmrc_Mean'].fillna(X['totmrc_Mean'].median())

X['da_Mean'] = X['da_Mean'].fillna(X['da_Mean'].median())

X['ovrmou_Mean'] = X['ovrmou_Mean'].fillna(X['ovrmou_Mean'].median())

X['datovr_Mean'] = X['datovr_Mean'].fillna(X['datovr_Mean'].median())

X['ovrrev_Mean'] = X['ovrrev_Mean'].fillna(X['ovrrev_Mean'].median())

X['vceovr_Mean'] = X['vceovr_Mean'].fillna(X['vceovr_Mean'].median())

X['roam_Mean'] = X['roam_Mean'].fillna(X['roam_Mean'].median())

X['change_mou'] = X['change_mou'].fillna(X['change_mou'].median())

X['change_rev'] = X['change_rev'].fillna(X['change_rev'].median())

X['hnd_price'] = X['hnd_price'].fillna(X['hnd_price'].median())

X['phones'] = X['phones'].fillna(X['phones'].median())

X['truck'] = X['truck'].fillna(X['truck'].median())

X['rv'] = X['rv'].fillna(X['rv'].median())

X['lor'] = X['lor'].fillna(X['lor'].median())

X['adults'] = X['adults'].fillna(X['adults'].median())

X['income'] = X['income'].fillna(X['income'].median())

X['numbcars'] = X['numbcars'].fillna(X['numbcars'].median())

X['forgntvl'] = X['forgntvl'].fillna(X['forgntvl'].median())

X['eqpdays'] = X['eqpdays'].fillna(X['eqpdays'].median())

X['avg6mou'] = X['avg6mou'].fillna(X['avg6mou'].median())

X['avg6qty'] = X['avg6qty'].fillna(X['avg6qty'].median())

X['avg6rev'] = X['avg6rev'].fillna(X['avg6rev'].median())





X['prizm_social_one'] = X['prizm_social_one'].fillna('U')

X['dualband'] = X['dualband'].fillna('U')

X['refurb_new'] = X['refurb_new'].fillna('N')

X['models'] = X['models'].fillna(1.0)

X['hnd_webcap'] = X['hnd_webcap'].fillna('UNKW')

X['ownrent'] = X['ownrent'].fillna('UNKW')

X['dwlltype'] = X['dwlltype'].fillna('UNKW')

X['marital'] = X['marital'].fillna('U')

X['infobase'] = X['infobase'].fillna('UNKW')

X['HHstatin'] = X['HHstatin'].fillna('UNKW')

X['dwllsize'] = X['dwllsize'].fillna('UNKW')

X['ethnic'] = X['ethnic'].fillna('U')

X['kid0_2'] = X['kid0_2'].fillna('U')

X['kid3_5'] = X['kid3_5'].fillna('U')

X['kid6_10'] = X['kid6_10'].fillna('U')

X['kid11_15'] = X['kid11_15'].fillna('U')

X['kid16_17'] = X['kid16_17'].fillna('U')

X['creditcd'] = X['creditcd'].fillna('U')





test['area'] = test['area'].fillna("UNKNOWN")

test['rev_Mean'] = test['rev_Mean'].fillna(-100000)

test['mou_Mean'] = test['mou_Mean'].fillna(-100000)

test['totmrc_Mean'] = test['totmrc_Mean'].fillna(-100000)

test['da_Mean'] = test['da_Mean'].fillna(-100000)

test['ovrmou_Mean'] = test['ovrmou_Mean'].fillna(-100000)

test['datovr_Mean'] = test['datovr_Mean'].fillna(-100000)

test['ovrrev_Mean'] = test['ovrrev_Mean'].fillna(-100000)

test['vceovr_Mean'] = test['vceovr_Mean'].fillna(-100000)

test['roam_Mean'] = test['roam_Mean'].fillna(-100000)

test['change_mou'] = test['change_mou'].fillna(-100000)

test['change_rev'] = test['change_rev'].fillna(-100000)

test['hnd_price'] = test['hnd_price'].fillna(-100000)

test['phones'] = test['phones'].fillna(-100000)

test['truck'] = test['truck'].fillna(-100000)

test['rv'] = test['rv'].fillna(-100000)

test['lor'] = test['lor'].fillna(-100000)

test['adults'] = test['adults'].fillna(-100000)

test['income'] = test['income'].fillna(-100000)

test['numbcars'] = test['numbcars'].fillna(-100000)

test['forgntvl'] = test['forgntvl'].fillna(-100000)

test['eqpdays'] = test['eqpdays'].fillna(-100000)

test['avg6mou'] = test['avg6mou'].fillna(-100000)

test['avg6qty'] = test['avg6qty'].fillna(-100000)

test['avg6rev'] = test['avg6rev'].fillna(-100000)







test['area'] = test['area'].fillna('UNKW')

test['rev_Mean'] = test['rev_Mean'].fillna(test['rev_Mean'].median())

test['mou_Mean'] = test['mou_Mean'].fillna(test['mou_Mean'].median())

test['totmrc_Mean'] = test['totmrc_Mean'].fillna(test['totmrc_Mean'].median())

test['da_Mean'] = test['da_Mean'].fillna(test['da_Mean'].median())

test['ovrmou_Mean'] = test['ovrmou_Mean'].fillna(test['ovrmou_Mean'].median())

test['datovr_Mean'] = test['datovr_Mean'].fillna(test['datovr_Mean'].median())

test['ovrrev_Mean'] = test['ovrrev_Mean'].fillna(test['ovrrev_Mean'].median())

test['vceovr_Mean'] = test['vceovr_Mean'].fillna(test['vceovr_Mean'].median())

test['roam_Mean'] = test['roam_Mean'].fillna(test['roam_Mean'].median())

test['change_mou'] = test['change_mou'].fillna(test['change_mou'].median())

test['change_rev'] = test['change_rev'].fillna(test['change_rev'].median())

test['hnd_price'] = test['hnd_price'].fillna(test['hnd_price'].median())

test['phones'] = test['phones'].fillna(test['phones'].median())

test['truck'] = test['truck'].fillna(test['truck'].median())

test['rv'] = test['rv'].fillna(test['rv'].median())

test['lor'] = test['lor'].fillna(test['lor'].median())

test['adults'] = test['adults'].fillna(test['adults'].median())

test['income'] = test['income'].fillna(test['income'].median())

test['numbcars'] = test['numbcars'].fillna(test['numbcars'].median())

test['forgntvl'] = test['forgntvl'].fillna(test['forgntvl'].median())

test['eqpdays'] = test['eqpdays'].fillna(test['eqpdays'].median())

test['avg6mou'] = test['avg6mou'].fillna(test['avg6mou'].median())

test['avg6qty'] = test['avg6qty'].fillna(test['avg6qty'].median())

test['avg6rev'] = test['avg6rev'].fillna(test['avg6rev'].median())





test['prizm_social_one'] = test['prizm_social_one'].fillna('U')

test['dualband'] = test['dualband'].fillna('U')

test['refurb_new'] = test['refurb_new'].fillna('N')

test['models'] = test['models'].fillna(1.0)

test['hnd_webcap'] = test['hnd_webcap'].fillna('UNKW')

test['ownrent'] = test['ownrent'].fillna('UNKW')

test['dwlltype'] = test['dwlltype'].fillna('UNKW')

test['marital'] = test['marital'].fillna('U')

test['infobase'] = test['infobase'].fillna('UNKW')

test['HHstatin'] = test['HHstatin'].fillna('UNKW')

test['dwllsize'] = test['dwllsize'].fillna('UNKW')

test['ethnic'] = test['ethnic'].fillna('U')

test['kid0_2'] = test['kid0_2'].fillna('U')

test['kid3_5'] = test['kid3_5'].fillna('U')

test['kid6_10'] = test['kid6_10'].fillna('U')

test['kid11_15'] = test['kid11_15'].fillna('U')

test['kid16_17'] = test['kid16_17'].fillna('U')

test['creditcd'] = test['creditcd'].fillna('U')



X = pd.get_dummies(X)

test = pd.get_dummies(test)
X = X.dropna()
test.shape
X.shape
to_drop = set(test.columns) - set(X.columns)

test = test.drop(columns = to_drop)

to_drop = set(X.columns) - set(test.columns)

X = X.drop(columns = to_drop)
test.shape
X = X.fillna(0)

from lightgbm import LGBMClassifier



clf = LGBMClassifier(

    n_estimators=900,

    learning_rate=0.03,

    num_leaves=30,

    colsample_bytree=.8,

    subsample=.9,

    max_depth=8,

    reg_alpha=.1,

    reg_lambda=.1,

    min_split_gain=.01,

    min_child_weight=2,

    silent=-1,

    verbose=-1,

)
from catboost import CatBoostClassifier



best_model = CatBoostClassifier(

   bagging_temperature=4,

   random_strength=1,

   thread_count=8,

   iterations=5000,

   l2_leaf_reg = 10.0,

   bootstrap_type='MVS',

   #learning_rate = 0.07521709965938336,

   save_snapshot=False,

   snapshot_file='snapshot_best.bkp',

   random_seed=42,

   od_type='IncToDec',

   od_wait=20,

   silent=True,

   custom_loss=['AUC', 'Accuracy'],

   #use_best_model=True   

)

from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[

       ('lgbm', clf), ('catboost', best_model)], voting='soft')
eclf1 = eclf1.fit(X, y)
from sklearn.model_selection import KFold, cross_val_score



y_pred = eclf1.predict_proba(test)

kf = KFold(n_splits=5, random_state=42) # cоздаем генератор разбиений

scores = cross_val_score(eclf1, X, y,scoring='roc_auc', cv=kf) # передаем генератор как пар

score = scores.mean()

score
y_pred = eclf1.predict_proba(test)

probability = pd.DataFrame(y_pred[:,1], columns=["churn"])

saveFrame = pd.concat([test["Customer_ID"],probability], axis = 1, sort = False)

saveFrame.to_csv('EnsebleTEST.csv', encoding = "utf-8-sig", index = None, header = True)
result = pd.read_csv('EnsebleTEST.csv')

train = pd.read_csv('/kaggle/input/cft-shift-customer-retention/train.csv')

test = pd.read_csv('/kaggle/input/cft-shift-customer-retention/test.csv')

to_test1_one = result[result['churn']>0.90]

to_test1_zero = result[result['churn'] < 0.13]



to_test1_one['churn'] = 1

to_test1_zero['churn'] = 0
new_t0 = test.merge(to_test1_one, on='Customer_ID', how='inner')

new_t1 = test.merge(to_test1_zero, on='Customer_ID', how='inner')

new_t01 = new_t0.append(new_t1, ignore_index=True,sort =False)

result_new = train.append(new_t01, ignore_index=True,sort =False)

result_new
result_new.to_csv('GAINED.csv', encoding = "utf-8-sig", index = None, header = True)