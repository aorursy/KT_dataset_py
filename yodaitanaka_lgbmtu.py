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
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")



data = pd.concat([train, test], sort=False)
from sklearn.preprocessing import LabelEncoder



data = pd.concat([train, test], sort=False)



data['Sex'].replace(['male','female'], [0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(data['Age'].median(), inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
data.head()
delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)



train = data[:len(train)]

test = data[len(train):]



y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)
X_train.head()
y_train.head()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)
from optuna.integration import lightgbm as lgb

import time



ts = time.time()



dtrain = lgb.Dataset(X_train, y_train)

eval_data = lgb.Dataset(X_valid, y_valid)



param = {

        'objective': 'binary'

    }



best_params, history = {}, []

model = lgb.train(param, dtrain, valid_sets=eval_data,

                    verbose_eval=20,

                    num_boost_round=1000,

                    early_stopping_rounds=20,

                    best_params=best_params,

                    tuning_history=history)



time.time() - ts
best_params
history
params = {

    'objective': 'binary',

    'metric':'auc',

    'learning_rate':0.05,

'lambda_l1':best_params['lambda_l1'],

 'lambda_l2':best_params['lambda_l2'],

 'num_leaves':best_params['num_leaves'],

 'feature_fraction':best_params['feature_fraction'],

 'bagging_fraction':best_params['bagging_fraction'],

 'bagging_freq':best_params['bagging_freq'],

 'min_child_samples':best_params['min_child_samples']

}



cv_result = lgb.cv(params, dtrain,

                    verbose_eval=20,

                    num_boost_round=1000,

                    early_stopping_rounds=100,

                    nfold = 10)
params = {

    'objective': 'binary',

    'metric':'auc',

    'learning_rate':0.05,

'lambda_l1': best_params['lambda_l1'],

 'lambda_l2':best_params['lambda_l2'],

 'num_leaves':best_params['num_leaves'],

 'feature_fraction':best_params['feature_fraction'],

 'bagging_fraction':best_params['bagging_fraction'],

 'bagging_freq':best_params['bagging_freq'],

 'min_child_samples':best_params['min_child_samples']

}



lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)



model = lgb.train(

    params, lgb_train,

    valid_sets=[lgb_train, lgb_eval],

    verbose_eval=10,

    num_boost_round=1000,

    early_stopping_rounds=10

)



y_pred = model.predict(X_test, num_iteration=model.best_iteration)
lgb.plot_importance(model)
sub = gender_submission

y_pred = (y_pred > 0.5).astype(int)

sub['Survived'] = y_pred

sub.to_csv("submission_lightgbm_optuna.csv", index=False)



sub.head()
sub = pd.DataFrame(pd.read_csv('../input/titanic/test.csv')['PassengerId'])

sub['Survived'] = list(map(int, y_pred))

sub.to_csv('submission.csv', index=False)