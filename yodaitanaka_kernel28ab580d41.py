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

#data = pd.concat([train, test], sort=False)

train['train_or_test'] = 'train'

test['train_or_test'] = 'test'

data = pd.concat(

    [

        train,

        test

    ],

    sort=False

).reset_index(drop=True)





#data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

data['Embarked'].fillna(data.Embarked.mode()[0], inplace=True)

#data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2}).astype(int)

data['Fare'].fillna(data.Fare.median(), inplace=True)

#data['Age'].fillna(data['Age'].median(), inplace=True)

#死んでるから欠損してる可能性もあるので比較的に体力的に生き残りにくそうな最年長の年齢にしてみた。

#data['Age'].fillna(data['Age'].max(), inplace=True)

#というか、Ageを埋めること自体主観でノイズを入れるからAgeを使わない。



data['FamilySize'] = data['Parch'] + data['SibSp'] + 1



#data['IsAlone'] = 0



#'FamilySize'が、1のものの"行"に対して、の'IsAlone'"列"に対して、1を代入している。

#ちなみに「loc」は行と列のラベルに対しての操作が行える、

#「iloc」というものもあり、これは行列の番号に対して、操作を行える。

#「ix」はどっちもできる



#おまけ「at」「iat」は単独要素に対して

#data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1





data['FamilySize_bin'] = 'big'

data.loc[data['FamilySize'] == 1, 'FamilySize_bin'] = 'alone'

data.loc[(data['FamilySize'] >= 2) & (data['FamilySize']<=4), 'FamilySize_bin'] = 'small'

data.loc[(data['FamilySize'] >= 5) & (data['FamilySize']<=7), 'FamilySize_bin'] = 'mediam'











data.loc[:, 'TicketFreq'] = data.groupby(['Ticket'])['PassengerId'].transform('count')



data['Cabin_ini'] = data['Cabin'].map(lambda x:str(x)[0])

data['Cabin_ini'].replace(['G', 'T'], 'Rare', inplace=True)



data.Pclass = data.Pclass.astype('str')



data['honorific'] = data['Name'].map(lambda x: x.split(',') [1].split('.')[0])

data['honorific'].replace(['Col', 'Dr', 'Rev'], 'Rare', inplace=True)

data['honorific'].replace('Mlle', 'Miss', inplace=True)

data['honorific'].replace('Ms', 'Miss', inplace=True)





data.loc[:, 'Fare_bin'] = pd.qcut(data.Fare, 14)



le_target_col = ['Sex','Fare_bin']

le = LabelEncoder()

for col in le_target_col:

    data.loc[:, col] = le.fit_transform(data[col])



cat_col = ['Embarked','FamilySize_bin', 'Pclass','Cabin_ini', 'honorific', 'Fare_bin']

data=pd.get_dummies(data, drop_first=True, columns=cat_col)
data.head()
#delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']

#ここがめっちゃ大事、ここでカラムを使うために使ったParch,SibSpとかのそのカラム単体で役に立たないものを捨てる。

#まるでロジカルプレゼンテーションのMECEのようだ…！もれなくダブりなくの大切さ…

#今後はこの前処理を使って、残りの手法も使いアンサンブル学習する。

delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin','Age', 'Fare','Parch','SibSp','FamilySize']

data.drop(delete_columns, axis=1, inplace=True)



#train = data[:len(train)]

#test = data[len(train):]



train = data.query('train_or_test == "train"')

test = data.query('train_or_test == "test"')



train.drop('train_or_test', axis=1, inplace=True)

test.drop('train_or_test', axis=1, inplace=True)





y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)
X_train.head()
y_train.head()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score



gs1 = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),

                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],

                  scoring='accuracy',

                  cv=2)

scores = cross_val_score(gs1, X_train, y_train, 

                         scoring='accuracy', cv=5)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), 

                                      np.std(scores)))
'''

from sklearn.ensemble import RandomForestClassifier



gs2 = GridSearchCV(estimator=RandomForestClassifier(random_state=0),

                  param_grid=[{ "max_depth": [6,7,8,9,10,11,12,13,14],

                               "min_samples_leaf": [1,2,3,4,5],

                               "max_features": [4,5,6,7,8,9]}],

                  scoring='accuracy',

                  cv=2)

scores = cross_val_score(gs2, X_train, y_train, 

                         scoring='accuracy', cv=5)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), 

                                      np.std(scores)))

'''
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



pipe_svc  = make_pipeline(StandardScaler(),LogisticRegression(random_state=0))

gs3 = GridSearchCV(estimator=pipe_svc,

                  param_grid=[{ "logisticregression__penalty": ['l2'],

                               "logisticregression__C": [0.001,0.01,0.1,1,10]},

                              {"logisticregression__penalty":['l1'],

                               "logisticregression__C":[0.001,0.01,0.1,1,10],

                               "logisticregression__solver":['liblinear']}],

                  scoring='accuracy',

                  cv=2)

scores = cross_val_score(gs3, X_train, y_train, 

                         scoring='accuracy', cv=5)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), 

                                      np.std(scores)))
'''

gs = gs2.fit(X_train,y_train)

print(gs.best_params_)

'''
'''

best_clf = RandomForestClassifier(max_depth=7,max_features=6,min_samples_leaf=1,random_state=0)

best_clf.fit(X_train, y_train)

best_y_pred = best_clf.predict(X_test)

'''
'''

best_y_pred[:10]

'''
'''

sub = pd.DataFrame(pd.read_csv('../input/titanic/test.csv')['PassengerId'])

sub['Survived'] = list(map(int, best_y_pred))

sub.to_csv('sub_randomforest_10.csv', index=False)

'''
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)
import optuna

import lightgbm as lgb

from sklearn.metrics import log_loss



'''

def objective(trial):

    params = {

        'objective': 'binary',

        'max_bin': trial.suggest_int('max_bin', 255, 500),

        'learning_rate': 0.01,

        'num_leaves': trial.suggest_int('num_leaves', 32, 128)

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



    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

    score = log_loss(y_valid, y_pred_valid)

    return score

'''
'''

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials=100)

'''
'''

params = {

    'objective': 'binary',

    'max_bin': study.best_params['max_bin'],

    'learning_rate': 0.01,

    'num_leaves': study.best_params['num_leaves']

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

'''
'''

sub = gender_submission

y_pred = (y_pred > 0.5).astype(int)

sub['Survived'] = y_pred

sub.to_csv("submission_lightgbm_optuna.csv", index=False)



sub.head()

'''
'''

sub = pd.DataFrame(pd.read_csv('../input/titanic/test.csv')['PassengerId'])

sub['Survived'] = list(map(int, y_pred))

sub.to_csv('submission.csv', index=False)

'''
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score, accuracy_score

import xgboost as xgb



#Objective関数の設定

def objective(trial):

    params = {

        'objective': 'binary:logistic',

        'max_depth': trial.suggest_int('max_depth', 1, 9),

        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),

        'learning_rate': 0.01



    }

    model = xgb.XGBClassifier(**params)

    model.fit(X_train, y_train)



    pred = model.predict(X_valid)



    accuracy = accuracy_score(y_valid, pred)

    return (1-accuracy)



if __name__ == '__main__':



    study = optuna.create_study()

    study.optimize(objective, n_trials=100)



    print(study.best_params)

    print(study.best_value)

    print(study.best_trial)
    params = {

        'objective': 'binary:logistic',

        'max_depth': 2,

        'n_estimators': 706,

        'learning_rate': 0.01



    }

    model = xgb.XGBClassifier(**params)

    model.fit(X_train, y_train)



    y_pred = model.predict(X_test)
sub = gender_submission

y_pred = (y_pred > 0.5).astype(int)

sub['Survived'] = y_pred

sub.to_csv("submission_xgboost_optuna.csv", index=False)



sub.head()
sub = pd.DataFrame(pd.read_csv('../input/titanic/test.csv')['PassengerId'])

sub['Survived'] = list(map(int, y_pred))

sub.to_csv('submission.csv', index=False)