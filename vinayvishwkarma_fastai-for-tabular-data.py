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
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

sample_submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
from fastai.tabular import *
'''from sklearn.decomposition import PCA

pc = PCA(n_components = 500)

pc.fit(train_features[cont_columns])'''
'''result_train = pd.DataFrame(pc.transform(train_features[cont_columns]))

result_test = pd.DataFrame(pc.transform(test_features[cont_columns]))

result_train[cat_names] = train_features[cat_names]

result_test[cat_names] = test_features[cat_names]



result_test['sig_id'] = test_features['sig_id']

result_train['sig_id'] = train_features['sig_id']

result_train.shape,result_test.shape'''
print(train_features.shape)

cat_names = ['cp_type','cp_time','cp_dose']

cont_columns = [i for i in train_features.columns if i not in ['cp_type','cp_time','cp_dose','sig_id']]

dep_var = [i for i in train_targets_scored.columns if i !='sig_id']

train_features[dep_var] = train_targets_scored[dep_var]

print(train_features.shape)
procs = [FillMissing,Categorify,Normalize]

data = (TabularList.from_df(train_features, procs=procs, cont_names=cont_columns, cat_names=cat_names)

        .split_by_rand_pct(valid_pct=0.10,seed=42)

        .label_from_df(cols=dep_var)

        .add_test(TabularList.from_df(test_features, cat_names=cat_names, cont_names=cont_columns, procs=procs))

        .databunch())
learn = tabular_learner(data,layers=[300,200])
'''learn.lr_find()

learn.recorder.plot(suggestion=True)'''
learn.fit_one_cycle(5, 3.31E-02)
learn.recorder.plot_losses()
learn.unfreeze()

learn.fit_one_cycle(5,slice(1e-04),wd=0.2)
learn.recorder.plot_losses()
learn.unfreeze()

learn.fit_one_cycle(5, slice(1e-5),wd=0.25)
learn.recorder.plot_losses()
learn.unfreeze()

learn.fit_one_cycle(5, slice(1e-6),wd=0.4)
learn.recorder.plot_losses()
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
submission = pd.DataFrame({'sig_id':sample_submission['sig_id']})

submission[dep_var] = pd.DataFrame(preds.detach().numpy())
submission.loc[submission['sig_id'].isin(test_features.loc[test_features['cp_type'] =='ctl_vehicle', 'sig_id']), train_targets_scored.columns[1:]] = 0
submission.head()
submission.to_csv('submission.csv',index=False)