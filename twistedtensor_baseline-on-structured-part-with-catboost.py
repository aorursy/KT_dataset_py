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
import pandas as pd
import numpy as np
import catboost as cb
!head ../input/sample_submission.csv
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.columns.values
# restrict to numerical features and categorical features that aren't overly specific (i.e. title)
features = [f for f in train.columns.values if not f in ['item_id','user_id','title','description','activation_date','image','deal_probability']]
# Treat everything but 'price' and 'item_seq_number' as categorical
numerical = ['price','item_seq_number','image_top_1']
cat_ix = [i for i,f in enumerate(features) if not f in numerical]
train.loc[:,'item_seq_number'] = train.loc[:,'item_seq_number'].astype(float)
test.loc[:,'item_seq_number'] = test.loc[:,'item_seq_number'].astype(float)
# Fill missing features (use mean for numerical features)
for f in features:
    if f in numerical:
        mean = train[f].mean()
        train.loc[:,f] = train[f].fillna(mean)
        test.loc[:,f] = test[f].fillna(mean)
    else:
        train.loc[:,f] = train[f].fillna('NULL')
        test.loc[:,f] = test[f].fillna('NULL')
cbr = cb.CatBoostRegressor()
cbr.fit(train[features],train['deal_probability'],cat_features=cat_ix)
cbr.score(train[features],train['deal_probability'])
test.loc[:,'deal_probability'] = cbr.predict(test[features])
test.loc[:,'deal_probability'] = test['deal_probability'].clip(lower=0,upper=1)
test[['item_id','deal_probability']].to_csv('submission1.csv',index=False)
