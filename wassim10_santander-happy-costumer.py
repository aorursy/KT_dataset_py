# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/train.csv")
data.head(5)
data.info()
pd.value_counts(data.TARGET, normalize = True)

cols = data.columns[1:-1]
data.head()
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from catboost import Pool, CatBoostClassifier
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=5168)

for fold, (trn_idx, val_idx) in enumerate(skf.split(data, data['TARGET'])):

    X_train, y_train = data.iloc[trn_idx][cols], data.iloc[trn_idx]['TARGET']

    X_valid, y_valid = data.iloc[val_idx][cols], data.iloc[val_idx]['TARGET']

    break

        

clf = CatBoostClassifier(loss_function = "Logloss", eval_metric = "AUC",random_seed=123,use_best_model=True,

                          learning_rate=0.1,  iterations=15000,verbose=100,

                           bootstrap_type= "Poisson", 

                           task_type="GPU",

                              l2_leaf_reg = 16.5056753964314982, depth= 3.0,

                              scale_pos_weight = 3.542962442406767, 

                              fold_permutation_block_size=16.0, subsample= 0.46893530376570957,

                              fold_len_multiplier=3.2685541035861747)

print("Model training")

clf.fit(X_train, y_train,  eval_set=(X_valid, y_valid), early_stopping_rounds=2000,verbose=100)
test = pd.read_csv("../input/test.csv")

sample = pd.read_csv('../input/sample_submission.csv')
predict = clf.predict_proba(test[cols])

sample.TARGET = predict[:,1]
sample.head()

from IPython.display import FileLink

def create_submission(submission_file, submission_name):

    submission_file.to_csv(submission_name+".csv",index=False)

    return FileLink(submission_name+".csv")
create_submission(sample, "submission")