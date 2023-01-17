# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import log_loss

from sklearn.model_selection import train_test_split, KFold



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from copy import deepcopy

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
np.random.seed(2020) #Setting the randomness to be deterministic
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
train.head()
targets.iloc[:,:].head()
test.head()
X_treino = train.drop('sig_id',axis='columns')

y_treino = targets.drop('sig_id',axis='columns')

X_teste = test.drop('sig_id',axis='columns')
enc = LabelEncoder()

category_cols = [cols for cols in X_treino.columns if X_treino[cols].dtype == 'object']

print(category_cols)

X_train_enc = deepcopy(X_treino)

X_test_enc = deepcopy(X_teste)

for cols in category_cols:

    X_train_enc[cols] = enc.fit_transform(X_train_enc[cols])

    X_test_enc[cols] = enc.transform(X_test_enc[cols])
X_train_enc.head() #Just to ensure we tranform right!
X_test_enc.head()
et = ExtraTreesClassifier(n_estimators=170,max_depth=10,n_jobs=-1,verbose=1)
X_train, X_test, y_train, y_test = train_test_split(X_train_enc,

                                                    y_treino,

                                                    test_size = 0.2)



et.fit(X_train,y_train)

y_pred_proba_t = et.predict_proba(X_test)



print("Log Loss Score: ",log_loss(y_test.values.ravel(), np.array(y_pred_proba_t)[:,:,1].T.ravel()))
et.fit(X_train_enc,y_treino)

y_pred_proba = et.predict_proba(X_test_enc) 
y_sub = np.array(y_pred_proba)[:,:,1].T

print(y_sub.shape)

print(test.sig_id.values.shape)

submission = pd.DataFrame(np.concatenate([test.sig_id.values[:,None],y_sub],axis=1),columns=targets.columns)

submission.head()

submission.to_csv('submission.csv',index=False)
"""kf = KFold(n_splits=10)

for i,(tr_idx,te_idx) in enumerate(kf.split(X_train_enc,y_treino)):

    print(f"Starting fold: {i}")

    X_train, X_test = X_train_enc.values[tr_idx],X_train_enc.values[te_idx]

    y_train, y_test = y_treino.values[tr_idx], y_treino.values[te_idx]

    et.fit(X_train,y_train)

    y_pred = et.predict_proba(X_test)

    y_pred = np.array(y_pred)[:,:,1].T

    log_loss_preds[te_idx] = y_pred

    

    predi = et.predict_proba(X_test_enc)

    predi = np.array(predi)[:,:,1].T

    teste_preds += predi / 10"""
"""print(log_loss(y_treino.values.ravel(),log_loss_preds.ravel()))"""