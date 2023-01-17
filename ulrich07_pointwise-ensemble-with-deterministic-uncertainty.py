import numpy as np

import pandas as pd

import pydicom

import os

import random

import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold
import tensorflow as tf

#import tensorflow.keras.backend as K

#import tensorflow.keras.layers as L

#import tensorflow.keras.models as M
def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    

seed_everything(42)
ROOT = "../input/osic-pulmonary-fibrosis-progression"

DESIRED_SIZE = 256

BATCH_SIZE=128
tr = pd.read_csv(f"{ROOT}/train.csv")

tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])

chunk = pd.read_csv(f"{ROOT}/test.csv")



print("add infos")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")

sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]

sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")
tr['WHERE'] = 'train'

chunk['WHERE'] = 'val'

sub['WHERE'] = 'test'

data = tr.append([chunk, sub])
data['min_week'] = data['Weeks']

data.loc[data.WHERE=='test','min_week'] = np.nan

data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
base = data.loc[data.Weeks == data.min_week]

base = base[['Patient','FVC']].copy()

base.columns = ['Patient','min_FVC']

base['nb'] = 1

base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

base = base[base.nb==1]

base.drop('nb', axis=1, inplace=True)
data = data.merge(base, on='Patient', how='left')

data['base_week'] = data['Weeks'] - data['min_week']

del base
COLS = ['Sex','SmokingStatus'] #,'Age'

FE = []

for col in COLS:

    for mod in data[col].unique():

        FE.append(mod)

        data[mod] = (data[col] == mod).astype(int)

#=================
#

data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )

data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )

data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )

data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )

FE += ['age','week','BASE'] #,'percent'
"""

DEGS = [2,3,4,5] #,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20

for deg in DEGS:

    col = f"age_{deg}"

    data[col] = data["age"]**deg

    data[col] = (data[col] - data[col].min() ) / ( data[col].max() - data[col].min() )

    FE.append(col)

#=============================================#

for deg in DEGS:

    col = f"week_{deg}"

    data[col] = data["week"]**deg

    data[col] = (data[col] - data[col].min() ) / ( data[col].max() - data[col].min() )

    FE.append(col)

#=============================================#

"""
tr = data.loc[data.WHERE=='train']

chunk = data.loc[data.WHERE=='val']

sub = data.loc[data.WHERE=='test']

del data
def metric( trueFVC, predFVC, predSTD ):

    

    clipSTD = np.clip( predSTD, 70 , 9e9 )  

    

    deltaFVC = np.clip( np.abs(trueFVC-predFVC), 0 , 1000 )  



    return np.mean( -1*(np.sqrt(2)*deltaFVC/clipSTD) - np.log( np.sqrt(2)*clipSTD ) )

#
y = tr['FVC'].values

z = tr[FE].values

ze = sub[FE].values
z.shape
from sklearn.linear_model import Ridge, ElasticNet

from sklearn.ensemble import RandomForestClassifier
NFOLD = 10

kf = KFold(n_splits=NFOLD)
#%%time

cnt = 0

#clf = ElasticNet(alpha=0.3, l1_ratio = 0.7)

clf = Ridge(alpha=0.05)

#clf  = RandomForestClassifier(max_depth=4, random_state=777, n_estimators=50)



pe = np.zeros((ze.shape[0], 2))

pred = np.zeros((z.shape[0], 2))



for tr_idx, val_idx in kf.split(z):

    cnt += 1

    print(f"FOLD {cnt}")

    clf.fit(z[tr_idx], y[tr_idx]) #

    #print("predict val...")

    pred[val_idx, 0] = clf.predict(z[val_idx])

    pred_std = np.mean(np.abs(y[val_idx] - pred[val_idx, 0])) * np.sqrt(2)

    pred[val_idx, 1] = pred_std

    print("val", metric(y[val_idx], pred[val_idx, 0], pred[val_idx, 1]))

    #print("predict test...")

    pe[:, 0] += clf.predict(ze) / NFOLD

    pe[:, 1] += pred_std / NFOLD

#==============

print("oof", metric(y, pred[:, 0], pred[:, 1]))
print("OOF uncertainty", np.unique(pred[:, 1]))

print("TEST uncertainty", np.unique(pe[:, 1]))
idxs = np.random.randint(0, y.shape[0], 80)

plt.plot(y[idxs], label="ground truth")

plt.plot(pred[idxs, 0], label="prediction")

plt.legend(loc="best")

plt.show()
sub.head()
sub['FVC1'] = pe[:, 0]

sub['Confidence1'] = pe[:, 1]
subm = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()
subm.loc[~subm.FVC1.isnull()].head(10)
subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']

subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']
subm.head()
subm.describe().T
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(otest)):

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1
subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)