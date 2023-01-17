# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold,GridSearchCV

from sklearn.metrics import log_loss, make_scorer, accuracy_score





from tqdm.notebook import tqdm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir = '../input/lish-moa/'



train_set = pd.read_csv('../input/lish-moa/train_features.csv')

train_target = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test_set = pd.read_csv('../input/lish-moa/test_features.csv')

#train_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')



ss = pd.read_csv('../input/lish-moa/sample_submission.csv')

ss.loc[:, train_target.columns[1:]] = 0



df1 = pd.DataFrame(train_set)

df2=pd.DataFrame(train_target)

df3 = pd.DataFrame(test_set)
#preprocessing



def transform_features(df):

    one_hot = pd.get_dummies(df['cp_dose'])    #one-hot encoding

    df = df.join(one_hot)

    df = df.drop(df[df.cp_type == 'ctl_vehicle'].index)  # drop where cp_type==ctl_vehicle

    df = df.drop(['sig_id','cp_type', 'cp_dose'], axis=1)

    return df.values





train_matrix = transform_features(df1)

test_matrix = transform_features(df3)





#training labels

drop_index = df1[df1.cp_type == 'ctl_vehicle'].index

df2 = df2.drop(drop_index, axis = 0)

train_target_matrix = np.delete(df2.values, 0, axis=1).astype(int)
#scaler

def scaling(D):

    min_max_scaler = preprocessing.MinMaxScaler()

    D = min_max_scaler.fit_transform(D)

    return D



train_matrix = scaling(train_matrix)

test_matrix = scaling(test_matrix)
#k-fold CV ~15min



N_folds = 5

SEED=40



LR=LogisticRegression(penalty = 'l2', C=0.5, random_state=SEED, tol = 0.1, verbose=0, max_iter = 1000)

kf = KFold(n_splits = N_folds, random_state = SEED, shuffle = True)



test_pred = np.zeros((test_matrix.shape[0], train_target_matrix.shape[1]))

oof_pred = np.zeros((train_target_matrix.shape[0], train_target_matrix.shape[1]))



for tar in tqdm(range(train_target_matrix.shape[1])):

    target = train_target_matrix[:, tar]

    

    if target.sum() >= N_folds: 



        for fold_idx, (train_idx, validate_idx) in enumerate(kf.split(train_matrix, target)):

            X_tr, X_val = train_matrix[train_idx], train_matrix[validate_idx]

            y_tr, y_val = target[train_idx], target[validate_idx]

            

            clf = LR.fit(X_tr, y_tr)

            test_pred[:,tar] += clf.predict_proba(test_matrix)[:,1]/ N_folds

            oof_pred[validate_idx,tar] += clf.predict_proba(X_val)[:,1]

print(f'LR OOF log loss: {log_loss(np.ravel(train_target_matrix), np.ravel(oof_pred))}')
predictions = test_pred.copy()

add_index = df3[df3.cp_type == 'ctl_vehicle'].index

for pos in add_index:

    predictions = np.insert(predictions, pos, values=np.zeros(206), axis=0)
#submission

ss = pd.DataFrame(predictions, columns=train_target.columns[1:])

ss.insert(0,'sig_id', df3['sig_id'].values)

ss.to_csv('submission.csv',index = False)

ss.describe()