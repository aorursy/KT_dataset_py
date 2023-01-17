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
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)
import optuna

from sklearn.metrics import log_loss

import lightgbm as lgb



def objective(trial):

    #探索するパラメータの設定

    params = {

        'objective': 'binary',

        'max_bin': trial.suggest_int('max_bin', 255, 500),

        'learning_rate': 0.01,

        'num_leaves': trial.suggest_int('num_leaves', 32, 128),

        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

        'verbosity': -1,

        'random_state':0

    }



    #lightGBM用にデータセット

    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_eval = lgb.Dataset(X_valid, y_valid)

    

    #LightGBMのモデル構築

    model = lgb.train(

        params, lgb_train,

        valid_sets=[lgb_train, lgb_eval],

        verbose_eval=10,

        num_boost_round=1000,

        early_stopping_rounds=10

    )

    #予測してスコア出す

    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

    score = log_loss(y_valid, y_pred_valid)

    return score
#指定方法

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0),

                             study_name='foo',

                             storage='sqlite:///example.db',

                             direction='minimize')

study.optimize(objective, n_trials=40)
study = optuna.load_study(study_name='foo',storage='sqlite:///example.db')

study.trials_dataframe()
optuna.visualization.plot_contour(study)
optuna.visualization.plot_edf(study)
optuna.visualization.plot_slice(study)
optuna.visualization.plot_param_importances(study)