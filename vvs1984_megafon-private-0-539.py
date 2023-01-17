# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
train_df = pd.read_csv('/kaggle/input/mf-accelerator/contest_train.csv')
blok_list = ['TARGET', 'ID']
train = [data for data in train_df.columns if data not in blok_list ]
nan_cols = []
train_cleared = [data for data in train if data not in nan_cols ]
corr_m=train_df[train_cleared].corr()
dict_corr = {}
corr_counter = {}



for f1 in train_cleared:
    hi_cor_list = list(corr_m[f1][(abs(corr_m[f1]) >0.5) & (abs(corr_m[f1]) <1)].index)
    if len(hi_cor_list) > 0:
        dict_corr[f1] = hi_cor_list
        for feat in hi_cor_list:
            if feat in corr_counter.keys():
                corr_counter[feat] += 1
            else:
                corr_counter[feat] = 1

                
corr_counter = {k: v for k, v in sorted(corr_counter.items(), key=lambda item: item[1], reverse = True)}
for key_counter in list(corr_counter.keys()):
    if key_counter in dict_corr.keys():
        for key in dict_corr[key_counter]:
            dict_corr.pop(key, None) 

index_new_feat = 0 
train_copy = train_df[train_cleared].copy()
for feat in train_copy.columns:
    if feat in dict_corr.keys():
        train_copy.drop(dict_corr[feat], axis=1, inplace = True, errors = 'ignore')  

X = train_copy
y = train_df['TARGET']
cat_list = ['FEATURE_1','FEATURE_2','FEATURE_5','FEATURE_6',
            'FEATURE_10','FEATURE_13','FEATURE_14','FEATURE_15',
            'FEATURE_16','FEATURE_17','FEATURE_18','FEATURE_19',
            'FEATURE_27', 'FEATURE_74', 'FEATURE_75', 'FEATURE_118',
            'FEATURE_123','FEATURE_133', 'FEATURE_142', 
            'FEATURE_144','FEATURE_145', 'FEATURE_146', 'FEATURE_150',
            'FEATURE_151', 'FEATURE_155', 'FEATURE_156',
            'FEATURE_157', 'FEATURE_159', 'FEATURE_172', 
            'FEATURE_175','FEATURE_176', 'FEATURE_178', 'FEATURE_197',
            'FEATURE_199', 'FEATURE_201', 'FEATURE_202', 'FEATURE_206', 
            'FEATURE_213','FEATURE_224', 'FEATURE_246','FEATURE_249',
            'FEATURE_257', 'FEATURE_258','FEATURE_259'
           ]
X[cat_list] = X[cat_list].astype(str)
ignored_feat2 =  [ 'FEATURE_10', 'FEATURE_257', 'FEATURE_245', 'FEATURE_201',
                  'FEATURE_246', 'FEATURE_247', 'FEATURE_145', 'FEATURE_202',
                  'FEATURE_27','FEATURE_157', 'FEATURE_146', 'FEATURE_156',
                  'FEATURE_234', 'FEATURE_242', 'FEATURE_75', 'FEATURE_231',
                  'FEATURE_29', 'FEATURE_2', 'FEATURE_3', 'FEATURE_5', 'FEATURE_6',
                  'FEATURE_15', 'FEATURE_16', 'FEATURE_17', 'FEATURE_18', 'FEATURE_19',
                  'FEATURE_20', 'FEATURE_22', 'FEATURE_25', 'FEATURE_28', 'FEATURE_39',
                   'FEATURE_40', 'FEATURE_41', 'FEATURE_139', 'FEATURE_141',
                  'FEATURE_144', 'FEATURE_159', 'FEATURE_229', 'FEATURE_249','FEATURE_256'
                 ]
it = 200
depth_t = 10
l2_reg = 9
lr = 0.15
weight_class = 'Balanced'


subm_name = f'./subm_f{X.shape[1]}_it{it}_depth{depth_t}_l2_{l2_reg}_lr{lr}_{weight_class}_ignefat_traincleared.csv '

model_train = CatBoostClassifier(
    iterations= it,
    custom_loss=['TotalF1'],
    loss_function = 'MultiClass',
    random_seed=42,
    task_type="GPU",
    cat_features = cat_list,
    ignored_features = ignored_feat2[-31:],
    depth = depth_t,
    l2_leaf_reg = l2_reg,
    learning_rate = lr,
    auto_class_weights = weight_class,
    logging_level='Silent'
)


model_train.fit(
    X, 
    y,
    plot=True
)
subm_df = pd.read_csv('/kaggle/input/mf-accelerator/sample_subm.csv')


test_df = pd.read_csv('/kaggle/input/mf-accelerator/contest_test.csv')
# test_df
list_columns = list(train_copy.columns) 

# test_df = test_df[list_columns]
test_df[cat_list] = test_df[cat_list].astype(str)


predictions = model_train.predict(test_df[list_columns])
# predictions = model_train.predict(test_df[train_cleared])

test_df['TARGET'] = predictions
subm_df.loc[subm_df.ID.isin(test_df.ID),'Predicted'] = test_df['TARGET']

subm_df.to_csv(subm_name, index=False)
