!pip install --quiet mlflow
!pip install --quiet nyaggle
import nyaggle



nyaggle.__version__
import numpy as np

import pandas as pd



train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
target_col = 'Survived'

target = train[target_col]

train.drop(columns=[target_col], inplace=True)
is_train = 'is_train'

train[is_train] = 1

test[is_train] = 0
data = pd.concat([train, test], sort=False).reset_index(drop=True)
data.head()
import nyaggle.feature_store as fs



@fs.cached_feature("all_feature")

def all_feature(df: pd.DataFrame) -> pd.DataFrame:

    df['Sex'] = df['Sex'].replace(['male','female'], [0, 1])

    df['Embarked'] = df['Embarked'].fillna(('S'))

    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    df['Fare'] = df['Fare'].fillna(np.mean(df['Fare']))

    df['Age'] = df['Age'].fillna(df['Age'].median())

    delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

    df = df.drop(delete_columns, axis=1)

    return df
_all_feature = all_feature(data)

# _embarked_feature = embarked_feature(data)
%ls features/
_all_feature.head()
del _all_feature
_all_feature = all_feature(data)
train, test = _all_feature[_all_feature[is_train] == 1], _all_feature[_all_feature[is_train] == 0]

train.drop(columns=[is_train], inplace=True)

test.drop(columns=[is_train], inplace=True)
from nyaggle.hyper_parameters import list_hyperparams



list_hyperparams()
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from nyaggle.experiment import run_experiment



skf = StratifiedKFold(4)



lgb_params = {'num_leaves': 8,

  'min_data_in_leaf': 42,

  'objective': 'binary',

  'max_depth': 16,

  'learning_rate': 0.03,

  'boosting': 'gbdt',

  'bagging_freq': 5,

  'bagging_fraction': 0.8,

  'feature_fraction': 0.8201,

  'reg_alpha': 1.7289,

  'reg_lambda': 4.984,

  'metric': 'auc',

  'subsample': 0.81,

  'min_gain_to_split': 0.01,

  'min_child_weight': 19.428}



fit_params = {

    "early_stopping_rounds": 100,

    "verbose": 100

}



result = run_experiment(lgb_params,

                        train, target, test, fit_params=fit_params,

                        cv=skf, eval_func=roc_auc_score, sample_submission=gender_submission,

                        with_mlflow=True)
# !mlflow ui