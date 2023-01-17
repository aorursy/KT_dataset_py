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
path = '/kaggle/input/titanic/'

train = pd.read_csv(path + 'train.csv')

test = pd.read_csv(path + 'test.csv')
train["is_train"] = 1

test['is_train'] = 0
df = pd.concat([train.drop(columns=['Survived']), test])
df["Embarked"] = df["Embarked"].fillna("S")

df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1).drop(columns=['Embarked'])

df = pd.concat([df, pd.get_dummies(df['Sex'], prefix='Sex')], axis=1).drop(columns=['Sex'])

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df.drop(columns=['SibSp', 'Parch'])
df.head()
df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False).value_counts()
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
def title_convert(title):

    if title == 'Master' or title == 'Miss' or title == 'Mr' or title == 'Mrs':

        return title

    else:

        return 'Others'
df['Title'] = [title_convert(i) for i in df['Title']]
df.groupby('Title').mean()['Age']
df['Age'].fillna(df.groupby('Title')['Age'].transform("mean"), inplace=True)
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df = pd.concat([df, pd.get_dummies(df['Title'], prefix='Title')], axis=1).drop(columns=['Title'])
df.head()
df['Ticket'].value_counts()

Ticket_Count = dict(df['Ticket'].value_counts())

df['TicketFreq'] = df['Ticket'].map(Ticket_Count)
df.head()
df['IsAlone'] = 1

df['IsAlone'].loc[df['FamilySize'] > 1] = 0 
df.head()
feature_columns = ['Pclass', 

                   'Sex_male',

                   'FamilySize',

                   'TicketFreq',

                   'Title_Mr',

                   'Title_Mrs',

                   'Title_Miss',

                   'Embarked_C',

                   'Embarked_Q'

                  ]

x_train = df[df['is_train']==1][feature_columns]

x_test = df[df['is_train']==0][feature_columns]

y_train = train['Survived']
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
#optunaでlightgbmのハイパーパラメータチューニング

def objective(trial):

    gamma = trial.suggest_discrete_uniform('gamma', 0.1, 1.0, 0.1)

    reg_lambda = trial.suggest_discrete_uniform('reg_lambda', 0, 1, 0.1)

    max_depth = trial.suggest_int('max_depth', 1, 20)

    min_child_weight = trial.suggest_int('min_child_weight', 1, 20)

    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)

    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)

    xgb = XGBClassifier(

        gamma=gamma,

        reg_lambda=reg_lambda,

        max_depth = max_depth,

        min_child_weight = min_child_weight,

        subsample = subsample,

        colsample_bytree = colsample_bytree,

    )

    xgb.fit(x_train,y_train)

    pred_test = xgb.predict(x_train)

    accuracy = accuracy_score(y_train, pred_test)

    return 1.0 - accuracy
!pip install optuna
import optuna



study = optuna.create_study()

study.optimize(objective, n_trials=50)
#xgboostで学習

kf = KFold(n_splits=3, shuffle=True)

score_list = []

models = []

importance = []



for fold_, (train_index, valid_index) in enumerate(kf.split(x_train, y_train)):

    print(f'fold{fold_+1}start')

    train_x = x_train.iloc[train_index]

    valid_x = x_train.iloc[valid_index]

    train_y = y_train[train_index]

    valid_y = y_train[valid_index]

    

    xgb = XGBClassifier(

        n_estimators = 300,

        gamma=0.8,

        reg_lambda=0.3,

        max_depth = 8,

        min_child_weight = 4,

        subsample = 0.7,

        colsample_bytree = 0.7,

    )

  

    xgb.fit(train_x,train_y)

    oof = xgb.predict(valid_x)

    score_list.append(round(accuracy_score(valid_y, oof)*100,2))

    models.append(xgb)

    print(f'fold{fold_ +1}end\n')



    print(score_list, '平均score', round(np.mean(score_list),2))
test_pred_matrix = np.zeros((len(test),4))

test_pred_matrix.shape



for fold_, xgb in enumerate(models):

    pred_ = xgb.predict(x_test)

    test_pred_matrix[:,fold_] = pred_

    



pred = (np.mean(test_pred_matrix, axis=1)>0.5).astype(int)
pred
submission = pd.read_csv(path + 'gender_submission.csv')

submission["Survived"] = pred

submission.to_csv('sub.csv', index=False)
import xgboost as xgb

from matplotlib import pyplot as plt



_, ax = plt.subplots(figsize=(12, 4))

xgb.plot_importance(models[1],

                    ax=ax,

                    importance_type='gain',

                    show_values=True)