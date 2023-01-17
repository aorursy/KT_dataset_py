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
train = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/train.csv')

test = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/test.csv')

train['type'] = train['type'].map({'normal':0,  'ependymoma':1,  'glioblastoma':2, 'medulloblastoma':3, 'pilocytic_astrocytoma':4})
from sklearn.feature_selection import SelectFromModel,RFE,SelectPercentile, f_regression

import lightgbm as lgb



Y = train['type'].values

X = train.drop('type',axis=1).values
model = lgb.LGBMClassifier()
from sklearn.metrics import f1_score

def met_f(y_test,y_pred):

    return f1_score(y_test,y_pred,average='micro')
selector = SelectFromModel(model, threshold="mean")    

selector.fit(X, Y)

X_selected = selector.transform(X)

test_selected = selector.transform(test.values)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_validate



#model = RandomForestClassifier()#0.6757141114329516

#model = xgb.XGBClassifier(scale_pos_weight = w)#0.942

#model = SVC(gamma='auto')#0.55

#model = DecisionTreeClassifier()#0.57

#model = GradientBoostingClassifier()#0.51

model = lgb.LGBMClassifier()#0.55



stratifiedkfold = StratifiedKFold(n_splits=3)



score_func = {'auc': make_scorer(met_f)}



scores = cross_validate(model, X_selected, Y, cv = stratifiedkfold, scoring=score_func)

print('auc:', scores['test_auc'])

print('auc:', scores['test_auc'].mean())
model = lgb.LGBMClassifier()

model.fit(X_selected,Y)

p = model.predict(test_selected)

sample = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/sampleSubmission.csv',index_col = 0)

sample['type'] = p

sample.to_csv('predict_lgbm_sfm.csv',header = True)
selector = SelectPercentile(score_func=f_regression, percentile=100) 

selector.fit(X, Y)

X_selected = selector.transform(X)

test_selected = selector.transform(test.values)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_validate



#model = RandomForestClassifier()#0.6757141114329516

#model = xgb.XGBClassifier(scale_pos_weight = w)#0.942

#model = SVC(gamma='auto')#0.55

#model = DecisionTreeClassifier()#0.57

#model = GradientBoostingClassifier()#0.51

model = lgb.LGBMClassifier()#0.55



stratifiedkfold = StratifiedKFold(n_splits=3)



score_func = {'auc': make_scorer(met_f)}



scores = cross_validate(model, X_selected, Y, cv = stratifiedkfold, scoring=score_func)

print('auc:', scores['test_auc'])

print('auc:', scores['test_auc'].mean())
model = lgb.LGBMClassifier()

model.fit(X_selected,Y)

p = model.predict(test_selected)

sample = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/sampleSubmission.csv',index_col = 0)

sample['type'] = p

sample.to_csv('predict_lgbm_100.csv',header = True)