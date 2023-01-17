# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# import os

# print(os.listdir("../input"))



# # Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import lightgbm as lgb

from scipy.special import logit



train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

features = [x for x in train_df.columns if x.startswith("var")]

hist_df = pd.DataFrame()

for var in features:

    var_stats = train_df[var].append(test_df[var]).value_counts()

    hist_df[var] = pd.Series(test_df[var]).map(var_stats)

    hist_df[var] = hist_df[var] > 1 # 对于每个变量var， 第i个subject是否拥有非独一无二的值（False即第i个subject的var的值在所有subject的该var中是独一无二的）
# hist_df
ind = hist_df.sum(axis=1) != 200 #第i个subject的所有var的值是否都在别的subject中出现过（若False即该subject拥有别的subject没有的独一无二的var的值）

# print(ind)

# 排除掉那些test set中有独一无二值的subject， 重新统计剩下的subject中各个var的值出现次数，并加入训练集

var_stats = {var:train_df[var].append(test_df[ind][var]).value_counts() for var in features}

# print(var_stats)



pred = 0

for var in features:

    """

    model = lgb.LGBMClassifier(**{ 'learning_rate': 0.04, 'num_leaves': 31, 'max_bin': 1023, 'min_child_samples': 1000, 'reg_alpha': 0.1, 'reg_lambda': 0.2,

     'feature_fraction': 1.0, 'bagging_freq': 1, 'bagging_fraction': 0.85, 'objective': 'binary', 'n_jobs': -1, 'n_estimators':200,})

    """

    

    model = lgb.LGBMClassifier(**{ 'learning_rate':0.05, 'max_bin': 165, 'max_depth': 5, 'min_child_samples': 150,

        'min_child_weight': 0.1, 'min_split_gain': 0.0018, 'n_estimators': 41, 'num_leaves': 6, 'reg_alpha': 2.0,

        'reg_lambda': 2.54, 'objective': 'binary', 'n_jobs': -1})

        

    model = model.fit(np.hstack([train_df[var].values.reshape(-1,1),

                                 train_df[var].map(var_stats[var]).values.reshape(-1,1)]),

                               train_df["target"].values)

    pred += logit(model.predict_proba(np.hstack([test_df[var].values.reshape(-1,1),

                                 test_df[var].map(var_stats[var]).values.reshape(-1,1)]))[:,1])

    

pd.DataFrame({"ID_code":test_df["ID_code"], "target":pred}).to_csv("submission.csv", index=False)
