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
'''
■ハイパーパラメタのチューニング方法
・手動でパラメタ調整：パラメタの変動、スコアの動きからデータの理解を深める
・グリッドサーチ/ランダムサーチ
　グリッドサーチは各パラメタの候補を定めてそれらの組み合わせをすべて試す、探索店の数が膨大になり計算量増える
　ランダムサーチは各パラメタの候補を定めてパラメタごとにランダムに選んだ組み合わせを繰り返す
・ベイズ最適化：計算したパラメタの履歴に基づいて次に探索すべきパラメタをベイズ確率の枠組みを用いて選択する方法

■パラメタチューニングの設定
1．ベースラインとなるパラメタ(初期値の重要性)：過去のコンペからパラメタ取得
2．探索する対象となるパラメタとその範囲の設定
3．手動調整/自動調整
4．評価の枠組み(クロスバリデーションなどのfoldの分け方)
　 パラメタチューニングを行うときのfoldの分割と実際にモデルを作成・予測するときのfoldの分割の乱数シードは変える
  
■パラメタチューニングのポイント
・重要なパラメタとそうでないパラメタがある
・パラメタの値を増加させたときにモデルの複雑性を増すパラメタと逆にモデルを単純にするパラメタがある
・パラメタのある範囲を探索したときにその上限または下限に良いスコアが集中するときは範囲を広げて探索するべき
・モデルの乱数シードやfoldの分割の乱数シードを変えた時のスコアの変化を見ることで単なるランダム性か変更によるパラメタ変更による改善なのか判断できる
'''
'''
■グリッドサーチを使ったハイパパラメタのチューニング
しらみつぶしの網羅的探索手法
チューニングしたいパラメタを指定するにはGridSearchCVのparam_gridパラメタに対して引数としてディクショナリのリストを指定する
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

#データの前処理----------------------------------------------------------------------------
train = pd.read_csv('../input/titanic-data/train.csv')
#NameはすべてユニークでPassengerIDと被るため削除
train_x = train.drop(["Name", "Survived"], axis=1)
train_y = train['Survived']
test_x = pd.read_csv('../input/titanic-data/test.csv')
test_x = test_x.drop(["Name"], axis=1)

#今回はひとまずlabel型のnullデータは適当な値で埋める→nullデータの決め方はFeature Engineering へ
train_x = train_x.fillna({'Cabin': 'A00', 'Embarked': "D"})
test_x = test_x.fillna({'Cabin': 'A00'})

#同様に数値型のnullデータはひとまず平均値で埋める
train_x['Age'] = train_x['Age'].fillna(train_x['Age'].mean()) 
test_x['Age'] = test_x['Age'].fillna(test_x['Age'].mean()) 
test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean()) 

#train_x.info()
#test_x.info()

#データの前処理-カテゴリ変数の数値化----------------------------------------------------------
label_cols = ["Sex", "Ticket", "Cabin", "Embarked"]

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', categories='auto')
ohe.fit(train_x[label_cols])

#print(ohe.categories_)

# ダミー変数の列名の作成
columns = []
for i, c in enumerate(label_cols):
    columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# 生成されたダミー変数をデータフレームに変換
dummy_vals_train = pd.DataFrame(ohe.transform(train_x[label_cols]), columns=columns)
dummy_vals_test = pd.DataFrame(ohe.transform(test_x[label_cols]), columns=columns)

#print(dummy_vals_train)
#print(dummy_vals_test)

# 残りの変数と結合元のデータフレームに結合
train_x = pd.concat([train_x.drop(label_cols, axis=1), dummy_vals_train], axis=1)
test_x = pd.concat([test_x.drop(label_cols, axis=1), dummy_vals_test], axis=1)

# 学習データを学習データとバリデーションデータに分ける------------------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

#pipelineは変換器と推定器を結合する---------------------------------------------------------
#変換器はfitメソッドとtransformメソッドをサポートするオブジェクト
#推定器はfitメソッドとpredictメソッドを実装しているオブジェクト
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

#パラメタの範囲指定
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

#網羅的探索のパラメタ設定
param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
gs = gs.fit(va_x, va_y)
print(gs.best_score_)
print(gs.best_params_)

'''
■ランダムサーチを使ったハイパパラメタのチューニング
(調整中)
'''
%time
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import scipy.stats
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#データの前処理----------------------------------------------------------------------------
train = pd.read_csv('../input/titanic-data/train.csv')
#NameはすべてユニークでPassengerIDと被るため削除
train_x = train.drop(["Name", "Survived"], axis=1)
train_y = train['Survived']
test_x = pd.read_csv('../input/titanic-data/test.csv')
test_x = test_x.drop(["Name"], axis=1)

#今回はひとまずlabel型のnullデータは適当な値で埋める→nullデータの決め方はFeature Engineering へ
train_x = train_x.fillna({'Cabin': 'A00', 'Embarked': "D"})
test_x = test_x.fillna({'Cabin': 'A00'})

#同様に数値型のnullデータはひとまず平均値で埋める
train_x['Age'] = train_x['Age'].fillna(train_x['Age'].mean()) 
test_x['Age'] = test_x['Age'].fillna(test_x['Age'].mean()) 
test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean()) 

#train_x.info()
#test_x.info()

#データの前処理-カテゴリ変数の数値化----------------------------------------------------------
label_cols = ["Sex", "Ticket", "Cabin", "Embarked"]

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', categories='auto')
ohe.fit(train_x[label_cols])

#print(ohe.categories_)

# ダミー変数の列名の作成
columns = []
for i, c in enumerate(label_cols):
    columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# 生成されたダミー変数をデータフレームに変換
dummy_vals_train = pd.DataFrame(ohe.transform(train_x[label_cols]), columns=columns)
dummy_vals_test = pd.DataFrame(ohe.transform(test_x[label_cols]), columns=columns)

#print(dummy_vals_train)
#print(dummy_vals_test)

# 残りの変数と結合元のデータフレームに結合
train_x = pd.concat([train_x.drop(label_cols, axis=1), dummy_vals_train], axis=1)
test_x = pd.concat([test_x.drop(label_cols, axis=1), dummy_vals_test], axis=1)

# 学習データを学習データとバリデーションデータに分ける------------------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]


# パラメーターの値の候補を設定
model_param_set_random =  {
    SVC(): {
        #"class_weight":["None","balanced"],
        "coef0":[i for i in range(0, 10)],
        "gamma":["scale","auto"],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "max_iter":[200,500,1000],
        "probability":["True","False"],
        "shrinking":["True","False"],
        "tol":[0.00001,0.0001,0.001,0.1,1],
        "verbose":["True","False"],
        "C": scipy.stats.uniform(0.00001, 1000),
        "decision_function_shape": ["ovr", "ovo"],
        "random_state": scipy.stats.randint(0, 100)
    },
    DecisionTreeClassifier():{
        #"class_weight":["None","balanced"],
        "criterion":["gini","entropy"],
        "max_features":["None","auto","log2"],
        "min_samples_leaf":[i for i in range(1, 10)],   #
        "min_weight_fraction_leaf":[0,0.1],
        "presort":["True","False"],
        "splitter":["best","random"],
        "max_depth":[i for i in range(1, 10)],     #
        "random_state": scipy.stats.randint(0, 100)
    },
    RandomForestClassifier():{
        "bootstrap":["True","False"],
        "class_weight":["None","balanced"],
        "criterion":["gini","entropy"],
        "max_features":["None","auto","log2"],
        "n_estimators":[i for i in range(1, 10)],
        "max_depth":[i for i in range(1, 10)],  
        "random_state": scipy.stats.randint(0, 100),
        "warm_start":["True","False"]
    },
    LogisticRegression():{
        "C":[10 ** i for i in range(-5, 5)],
        #"class_weight":["None","balanced"],
        "dual":["None","balanced"],
        "fit_intercept":["True","False"],
        "max_iter":[200,500,1000],
        "multi_class":["ovr","multinominal"],
        "penalty":["l1","l2"],
        "random_state": scipy.stats.randint(0, 100),
        "solver":["newton-cg","lbfgs","liblinear","sag"],
        "tol":[0.00001,0.0001,0.001,0.1,1],  
        "warm_start":["True","False"]
    },
    KNeighborsClassifier():{
        "algorithm":["ball_tree","kd_tree","brute","auto"],
        "leaf_size":[30],
        "n_jobs":[1],
        "n_neighbors":[i for i in range(1, 11)],
        "weights":["uniform","distance"]
    } 
    }

max_score = 0
best_param = None

# ランダムサーチでパラメーターサーチ
for model, param in model_param_set_random.items():

    clf = RandomizedSearchCV(model, param)
    clf.fit(va_x, va_y)
    pred_y = clf.predict(va_x)
    score = f1_score(va_y, pred_y, average="micro")
    if max_score < score:
        max_score = score
        best_param = clf.best_params_
        best_model = clf.__class__.__name__
    
  

# 最も成績のいいスコアを出力してください。
print("ベストモデル:{}".format(best_model))
print("パラメーター:{}".format(best_param))
print("ベストスコア:{:.5f}".format(max_score))
'''
■入れ子式交差検証アルゴリズム
外側のループでk分割交差検証を使用しデータをトレーニングサブセットとテストサブセットに分割する(最適なパラメタでトレーニング)
内側のループでトレーニングサブセットに対してk分割交差検証を行うことでモデルを選択(パラメタをチューニング)

-交差検証する理由
1.データセットを全部学習に使ってしまうと、その汎化性能が測定できない。
2.学習・検証するデータ交差させないと、訓練された学習器が偏ってないとは言えない。
3.十分に大きいデータセットなら、テスト用のデータと訓練用のデータで分けちゃってもいいかもしれませんが、
  数万単位だったら交差検証やっといたほうがいい
https://qiita.com/LicaOka/items/c6725aa8961df9332cc7
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

#データの前処理----------------------------------------------------------------------------
train = pd.read_csv('../input/titanic-data/train.csv')
#NameはすべてユニークでPassengerIDと被るため削除
train_x = train.drop(["Name", "Survived"], axis=1)
train_y = train['Survived']
test_x = pd.read_csv('../input/titanic-data/test.csv')
test_x = test_x.drop(["Name"], axis=1)

#今回はひとまずlabel型のnullデータは適当な値で埋める→nullデータの決め方はFeature Engineering へ
train_x = train_x.fillna({'Cabin': 'A00', 'Embarked': "D"})
test_x = test_x.fillna({'Cabin': 'A00'})

#同様に数値型のnullデータはひとまず平均値で埋める
train_x['Age'] = train_x['Age'].fillna(train_x['Age'].mean()) 
test_x['Age'] = test_x['Age'].fillna(test_x['Age'].mean()) 
test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean()) 

#train_x.info()
#test_x.info()

#データの前処理-カテゴリ変数の数値化----------------------------------------------------------
label_cols = ["Sex", "Ticket", "Cabin", "Embarked"]

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', categories='auto')
ohe.fit(train_x[label_cols])

#print(ohe.categories_)

# ダミー変数の列名の作成
columns = []
for i, c in enumerate(label_cols):
    columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# 生成されたダミー変数をデータフレームに変換
dummy_vals_train = pd.DataFrame(ohe.transform(train_x[label_cols]), columns=columns)
dummy_vals_test = pd.DataFrame(ohe.transform(test_x[label_cols]), columns=columns)

#print(dummy_vals_train)
#print(dummy_vals_test)

# 残りの変数と結合元のデータフレームに結合
train_x = pd.concat([train_x.drop(label_cols, axis=1), dummy_vals_train], axis=1)
test_x = pd.concat([test_x.drop(label_cols, axis=1), dummy_vals_test], axis=1)

# 学習データを学習データとバリデーションデータに分ける------------------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

#pipelineは変換器と推定器を結合する---------------------------------------------------------
#変換器はfitメソッドとtransformメソッドをサポートするオブジェクト
#推定器はfitメソッドとpredictメソッドを実装しているオブジェクト
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

#パラメタの範囲指定
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

#網羅的探索のパラメタ設定
param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

#SVCのパラメタのチューニング
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=2, n_jobs=-1)
#SVCパラメタのチューニングの結果、最適なパラメタを使用してトレーニング
scores = cross_val_score(gs, va_x, va_y, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#決定木のパラメタのチューニング
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], scoring='accuracy', cv=2)
#決定木パラメタのチューニングの結果、最適なパラメタを使用してトレーニング
scores = cross_val_score(gs, va_x, va_y, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
'''
■ベイズ最適化でのパラメタ探索(optuna)

以下の設定を行う
・最小化したい評価指標の設定
・探索するパラメタの範囲の定義
・探索回数の設定
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import lightgbm as lgb
import optuna
from sklearn.metrics import log_loss

#データの前処理----------------------------------------------------------------------------
train = pd.read_csv('../input/titanic-data/train.csv')
#NameはすべてユニークでPassengerIDと被るため削除
train_x = train.drop(["Name", "Survived"], axis=1)
train_y = train['Survived']
test_x = pd.read_csv('../input/titanic-data/test.csv')
test_x = test_x.drop(["Name"], axis=1)

#今回はひとまずlabel型のnullデータは適当な値で埋める→nullデータの決め方はFeature Engineering へ
train_x = train_x.fillna({'Cabin': 'A00', 'Embarked': "D"})
test_x = test_x.fillna({'Cabin': 'A00'})

#同様に数値型のnullデータはひとまず平均値で埋める
train_x['Age'] = train_x['Age'].fillna(train_x['Age'].mean()) 
test_x['Age'] = test_x['Age'].fillna(test_x['Age'].mean()) 
test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean()) 

#train_x.info()
#test_x.info()

#データの前処理-カテゴリ変数の数値化----------------------------------------------------------
label_cols = ["Sex", "Ticket", "Cabin", "Embarked"]

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', categories='auto')
ohe.fit(train_x[label_cols])

#print(ohe.categories_)

# ダミー変数の列名の作成
columns = []
for i, c in enumerate(label_cols):
    columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# 生成されたダミー変数をデータフレームに変換
dummy_vals_train = pd.DataFrame(ohe.transform(train_x[label_cols]), columns=columns)
dummy_vals_test = pd.DataFrame(ohe.transform(test_x[label_cols]), columns=columns)

#print(dummy_vals_train)
#print(dummy_vals_test)

# 残りの変数と結合元のデータフレームに結合
train_x = pd.concat([train_x.drop(label_cols, axis=1), dummy_vals_train], axis=1)
test_x = pd.concat([test_x.drop(label_cols, axis=1), dummy_vals_test], axis=1)

# 学習データを学習データとバリデーションデータに分ける------------------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

print(tr_x)
print(tr_y)
print(va_x)
print(va_y)

# optunaを使ったパラメータ探索-------------------------------------------------------------
def objective(trial):
    params = {
        'objective': 'binary',
        'max_bin': trial.suggest_int('max_bin', 255, 500),
        'learning_rate': 0.05,
        'num_leaves': trial.suggest_int('num_leaves', 32, 128),
    }

    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(va_x, va_y, reference=lgb_train)

    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000, early_stopping_rounds=10)

    y_pred_valid = model.predict(va_x, num_iteration=model.best_iteration)
    score = log_loss(va_y, y_pred_valid)
    return score

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=40)
study.best_params

'''
ベイズ最適化でのパラメタ探索(for xgboost)

手動でのチューニング方法は「Complete Guide to Parameter Tuning in XGBoost」
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import log_loss

#データの前処理----------------------------------------------------------------------------
train_df = pd.read_csv('../input/titanic-data/train.csv')
test_df = pd.read_csv('../input/titanic-data/test.csv')
train_y = train_df['Survived']
#test_dfから、TicketとCabinの特徴量を削除
train_df = train_df.drop(["Ticket", "Cabin"], axis=1)
test_df = test_df.drop(["Ticket", "Cabin"], axis=1)

combine = [train_df, test_df]
#train_dfとtest_dfそれぞれに対して、Titleという新しい特徴量の中に、ドット(.)より前の敬称を格納してください。
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#SexとTitleでクロス集計をしてください。クロス集計はcrosstabを利用するとできます。
df_list = pd.crosstab(train_df['Title'], train_df['Sex'])
#print(df_list)

#Master, Mr, Miss, Mrsなどの敬称が存在することが分かりました。今度は頻出な値以外をRareという値に置き換え
for dataset in combine:
    # 1. train_df, test_dfのTitleで、'Lady', 'Countess','Capt', 'Col',　'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'以外の項目に関しては'Rare'に書き換え
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    #同様に、MileはMissに書き換えてください。
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    #MmeはMrsに書き換えてください。
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#TitleとSurvivedで、Titleで集計してSurvivedの平均値を算出    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#これらの敬称を、予測モデルにしやすいように順序データに変換
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:
    dataset["Title"] = dataset["Title"] .map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    dataset['Title'] = dataset['Title'].fillna(0)

#NameとPassangerIdをtrain_dfから削除----------------------------------------------------
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

#Nameをtest_dfから削除
test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]
#train_dfとtest_dfの行数と列数を確認。
#print(train_df.shape, test_df.shape)    
    
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#先頭行を抽出
#print(train_df.head())

#連続値であるAgeを任意の個数で分割し離散値に変換----------------------------------------------
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

#AgeBandとSurvivedのピボットテーブルを作成
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
    
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]
#ParchとSibSpを足し合わせた、FamilySizeという特徴量を新規に作成
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
#FamilySizeとSurvivedの平均値をグループで集計してください。--------------------------------
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#isAloneという新しい特徴量を作成します。この特徴量には独身者なら1、家族持ちなら0が格納---------
for dataset in combine:
    # 1. IsAloneという特徴量を全て0として作成
    dataset['IsAlone'] = 0
    # 2. FamilySizeが1の時、isAloneを1。
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
#IsAloneとSurvivedをグループ集計して、Survivedの平均値を出力
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

#train_dfとtest_dfからParch、SibSp、およびFamilySizeを削除------------------------------
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

#Embarkedでnaを削除し、最頻値を新たな特徴量freq_portに格納="S"---------------------------
freq_port = train_df.Embarked.dropna().mode()[0]
#train_dfとtest_dfの欠損値を、freq_portの値に置き換え
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

#EmbarkedとSurvivedをグループ集計して、Survivedの平均値を出力
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Embarkedを{'S': 0, 'C': 1, 'Q': 2}に置き換えて下さい。
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#test_dfのFareの欠損値を、Fareのmedianで埋める----------------------------------------
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

#Fareを4階層に分けたFareBandという特徴量を作成-----------------------------------------

#連続値であるFareを4個に分割し離散値へ変換
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

#FareBandとSurvivedのピボットテーブルを作成
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
#Fareの値が7.91以下なら0、7.91超え14.454以下なら1、14.454超え31以下なら2、31超え3へ変換
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand','Survived'], axis=1)
test_df = test_df.drop(['PassengerId'], axis=1)
combine = [train_df, test_df]

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean()) 
train_df['Age*Class'] = train_df['Age*Class'].fillna(train_df['Age*Class'].mean()) 
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean()) 
test_df['Age*Class'] = test_df['Age*Class'].fillna(test_df['Age*Class'].mean()) 
    
train_x = train_df.copy()
test_x = test_df.copy()

# 学習データを学習データとバリデーションデータに分ける------------------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# xgboostによる学習・予測を行うクラス-------------------------------------------------------
class Model:

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, tr_x, tr_y, va_x, va_y):
        params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
        params.update(self.params)
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred

def score(params):
    # パラメータを与えたときに最小化する評価指標を指定する
    # 具体的には、モデルにパラメータを指定して学習・予測させた場合のスコアを返すようにする
    # max_depthの型を整数型に修正する
    params['max_depth'] = int(params['max_depth'])
    # Modelクラスを定義しているものとする
    # Modelクラスは、fitで学習し、predictで予測値の確率を出力する
    model = Model(params)
    model.fit(tr_x, tr_y, va_x, va_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    print(f'params: {params}, logloss: {score:.4f}')
    # 情報を記録しておく
    history.append((params, score))
    return {'loss': score, 'status': STATUS_OK}

# ベースラインのパラメータ(xgboostのパラメータ空間の例)------------------------------------- 
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': 0.1,
    'gamma': 0.0,
    'alpha': 0.0,
    'lambda': 1.0,
    'min_child_weight': 1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 71,
}

# パラメータの探索範囲------------------------------------------------------------------
param_space = {
    'min_child_weight': hp.loguniform('min_child_weight', np.log(0.1), np.log(10)),
    'max_depth': hp.quniform('max_depth', 3, 9, 1),
    'subsample': hp.quniform('subsample', 0.6, 0.95, 0.05),
    #'colsample_bytree': hp.quniform('subsample', 0.6, 0.95, 0.05),
    'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),
    # 余裕があればalpha, lambdaも調整する
    'alpha' : hp.loguniform('alpha', np.log(1e-8), np.log(1.0)),
    'lambda' : hp.loguniform('lambda', np.log(1e-6), np.log(10.0)),
}

# hyperoptによるパラメータ探索の実行----------------------------------------------------
max_evals = 10
trials = Trials()
history = []
fmin(score, param_space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

# 記録した情報からパラメータとスコアを出力する
# （trialsからも情報が取得できるが、パラメータの取得がやや行いづらいため）
history = sorted(history, key=lambda tpl: tpl[1])
best = history[0]
print(f'best params:{best[0]}, score:{best[1]:.4f}')

'''
ベイズ最適化でのパラメタ探索(for ニューラルネット)
多層パーセプトロンにおいてはネットワークの構成、オプティマイザなどが調整の対象になる
・ネットワークンの構成：中間層の活性化関数、中間層の層数、各層のユニット数、ドロップアウトの率、Batch Normalization層を適用するか
・オプティマイザの選択：SGD,Adamなど
・バッチサイズ
・Waight Dcayなどの正則化の導入
・オプティマイザの学習率以外の調整
#ニューラルネットのパラメタの調整は過去のkaggleのソリューションが参考になる

**最適化パラメタを見つけた後に実際に予測値を出すためにプログラムを組み上げる**

'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
# tensorflowの警告抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from hyperopt import hp
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import ReLU, PReLU
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import log_loss

#データの前処理----------------------------------------------------------------------------
train_df = pd.read_csv('../input/titanic-data/train.csv')
test_df = pd.read_csv('../input/titanic-data/test.csv')
train_y = train_df['Survived']
#test_dfから、TicketとCabinの特徴量を削除
train_df = train_df.drop(["Ticket", "Cabin"], axis=1)
test_df = test_df.drop(["Ticket", "Cabin"], axis=1)

combine = [train_df, test_df]
#train_dfとtest_dfそれぞれに対して、Titleという新しい特徴量の中に、ドット(.)より前の敬称を格納してください。
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#SexとTitleでクロス集計をしてください。クロス集計はcrosstabを利用するとできます。
df_list = pd.crosstab(train_df['Title'], train_df['Sex'])
#print(df_list)

#Master, Mr, Miss, Mrsなどの敬称が存在することが分かりました。今度は頻出な値以外をRareという値に置き換え
for dataset in combine:
    # 1. train_df, test_dfのTitleで、'Lady', 'Countess','Capt', 'Col',　'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'以外の項目に関しては'Rare'に書き換え
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    #同様に、MileはMissに書き換えてください。
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    #MmeはMrsに書き換えてください。
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#TitleとSurvivedで、Titleで集計してSurvivedの平均値を算出    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#これらの敬称を、予測モデルにしやすいように順序データに変換
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:
    dataset["Title"] = dataset["Title"] .map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    dataset['Title'] = dataset['Title'].fillna(0)

#NameとPassangerIdをtrain_dfから削除----------------------------------------------------
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

#Nameをtest_dfから削除
test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]
#train_dfとtest_dfの行数と列数を確認。
#print(train_df.shape, test_df.shape)    
    
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#先頭行を抽出
#print(train_df.head())

#連続値であるAgeを任意の個数で分割し離散値に変換----------------------------------------------
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

#AgeBandとSurvivedのピボットテーブルを作成
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
    
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]
#ParchとSibSpを足し合わせた、FamilySizeという特徴量を新規に作成
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
#FamilySizeとSurvivedの平均値をグループで集計してください。--------------------------------
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#isAloneという新しい特徴量を作成します。この特徴量には独身者なら1、家族持ちなら0が格納---------
for dataset in combine:
    # 1. IsAloneという特徴量を全て0として作成
    dataset['IsAlone'] = 0
    # 2. FamilySizeが1の時、isAloneを1。
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
#IsAloneとSurvivedをグループ集計して、Survivedの平均値を出力
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

#train_dfとtest_dfからParch、SibSp、およびFamilySizeを削除------------------------------
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

#Embarkedでnaを削除し、最頻値を新たな特徴量freq_portに格納="S"---------------------------
freq_port = train_df.Embarked.dropna().mode()[0]
#train_dfとtest_dfの欠損値を、freq_portの値に置き換え
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

#EmbarkedとSurvivedをグループ集計して、Survivedの平均値を出力
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Embarkedを{'S': 0, 'C': 1, 'Q': 2}に置き換えて下さい。
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#test_dfのFareの欠損値を、Fareのmedianで埋める----------------------------------------
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

#Fareを4階層に分けたFareBandという特徴量を作成-----------------------------------------

#連続値であるFareを4個に分割し離散値へ変換
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

#FareBandとSurvivedのピボットテーブルを作成
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
#Fareの値が7.91以下なら0、7.91超え14.454以下なら1、14.454超え31以下なら2、31超え3へ変換
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand','Survived'], axis=1)
test_df = test_df.drop(['PassengerId'], axis=1)
combine = [train_df, test_df]

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean()) 
train_df['Age*Class'] = train_df['Age*Class'].fillna(train_df['Age*Class'].mean()) 
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean()) 
test_df['Age*Class'] = test_df['Age*Class'].fillna(test_df['Age*Class'].mean()) 
    
train_x = train_df.copy()
test_x = test_df.copy()


# 学習データを学習データとバリデーションデータに分ける------------------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# ニューラルネットのパラメータチューニングの例------------------------------------------------
# 基本となるパラメータ
base_param = {
    'input_dropout': 0.0,
    'hidden_layers': 3,
    'hidden_units': 96,
    'hidden_activation': 'relu',
    'hidden_dropout': 0.2,
    'batch_norm': 'before_act',
    'optimizer': {'type': 'adam', 'lr': 0.001},
    'batch_size': 64,
}

# 探索するパラメータの空間を指定する
param_space = {
    'input_dropout': hp.quniform('input_dropout', 0, 0.2, 0.05),
    'hidden_layers': hp.quniform('hidden_layers', 2, 4, 1),
    'hidden_units': hp.quniform('hidden_units', 32, 256, 32),
    'hidden_activation': hp.choice('hidden_activation', ['prelu', 'relu']),
    'hidden_dropout': hp.quniform('hidden_dropout', 0, 0.3, 0.05),
    'batch_norm': hp.choice('batch_norm', ['before_act', 'no']),
    'optimizer': hp.choice('optimizer',
                           [{'type': 'adam',
                             'lr': hp.loguniform('adam_lr', np.log(0.00001), np.log(0.01))},
                            {'type': 'sgd',
                             'lr': hp.loguniform('sgd_lr', np.log(0.00001), np.log(0.01))}]),
    'batch_size': hp.quniform('batch_size', 32, 128, 32),
}


class MLP:

    def __init__(self, params):
        self.params = params
        self.scaler = None
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        # パラメータ
        input_dropout = self.params['input_dropout']
        hidden_layers = int(self.params['hidden_layers'])
        hidden_units = int(self.params['hidden_units'])
        hidden_activation = self.params['hidden_activation']
        hidden_dropout = self.params['hidden_dropout']
        batch_norm = self.params['batch_norm']
        optimizer_type = self.params['optimizer']['type']
        optimizer_lr = self.params['optimizer']['lr']
        batch_size = int(self.params['batch_size'])

        # 標準化
        self.scaler = StandardScaler()
        tr_x = self.scaler.fit_transform(tr_x)
        va_x = self.scaler.transform(va_x)

        self.model = Sequential()

        # 入力層
        self.model.add(Dropout(input_dropout, input_shape=(tr_x.shape[1],)))

        # 中間層
        for i in range(hidden_layers):
            self.model.add(Dense(hidden_units))
            if batch_norm == 'before_act':
                self.model.add(BatchNormalization())
            if hidden_activation == 'prelu':
                self.model.add(PReLU())
            elif hidden_activation == 'relu':
                self.model.add(ReLU())
            else:
                raise NotImplementedError
            self.model.add(Dropout(hidden_dropout))

        # 出力層
        self.model.add(Dense(1, activation='sigmoid'))

        # オプティマイザ
        if optimizer_type == 'sgd':
            optimizer = SGD(lr=optimizer_lr, decay=1e-6, momentum=0.9, nesterov=True)
        elif optimizer_type == 'adam':
            optimizer = Adam(lr=optimizer_lr, beta_1=0.9, beta_2=0.999, decay=0.)
        else:
            raise NotImplementedError

        # 目的関数、評価指標などの設定
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer, metrics=['accuracy'])

        # エポック数、アーリーストッピング
        # あまりepochを大きくすると、小さい学習率のときに終わらないことがあるので注意
        nb_epoch = 15
        patience = 20
        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

        # 学習の実行
        history = self.model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=batch_size, verbose=1, validation_data=(va_x, va_y), callbacks=[early_stopping])

    def predict(self, x):
        # 予測
        x = self.scaler.transform(x)
        y_pred = self.model.predict(x)
        y_pred = y_pred.flatten()
        return y_pred

# ----------------------------------------------------------------------------------------
# パラメータチューニングの実行
def score(params):
    # パラメータセットを指定したときに最小化すべき関数を指定する
    # モデルのパラメータ探索においては、モデルにパラメータを指定して学習・予測させた場合のスコアとする
    model = MLP(params)
    model.fit(tr_x, tr_y, va_x, va_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    print(f'params: {params}, logloss: {score:.4f}')

    # 情報を記録しておく
    history.append((params, score))

    return {'loss': score, 'status': STATUS_OK}

# hyperoptによるパラメータ探索の実行
max_evals = 10
trials = Trials()
history = []
fmin(score, param_space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

# 記録した情報からパラメータとスコアを出力する
# trialsからも情報が取得できるが、パラメータを取得しにくい
history = sorted(history, key=lambda tpl: tpl[1])
best = history[0]
print(f'best params:{best[0]}, score:{best[1]:.4f}')
'''
■線形モデルのパラメタのチューニング
主に正則化のパラメタがチューニングの対象になる
・Lasso,Ridge:alphaが正則化の強さを表すパラメタ。LassoではL1正則化、RidgeではL2正則化が行われる
・ElasticNet:alphaが正則化の強さを表すパラメタ・l1_ratioがL1正則化とL2正則化の割合を表すパラメタ
・LogisticRegression:Cが正則化の強さを表すパラメタ(デフォルトではL2正則化)

https://qiita.com/FujiedaTaro/items/5784eda386146f1fd6e7

'''

%time
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score

#データの前処理----------------------------------------------------------------------------
train_df = pd.read_csv('../input/titanic-data/train.csv')
test_df = pd.read_csv('../input/titanic-data/test.csv')
train_y = train_df['Survived']
#test_dfから、TicketとCabinの特徴量を削除
train_df = train_df.drop(["Ticket", "Cabin"], axis=1)
test_df = test_df.drop(["Ticket", "Cabin"], axis=1)

combine = [train_df, test_df]
#train_dfとtest_dfそれぞれに対して、Titleという新しい特徴量の中に、ドット(.)より前の敬称を格納してください。
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#SexとTitleでクロス集計をしてください。クロス集計はcrosstabを利用するとできます。
df_list = pd.crosstab(train_df['Title'], train_df['Sex'])
#print(df_list)

#Master, Mr, Miss, Mrsなどの敬称が存在することが分かりました。今度は頻出な値以外をRareという値に置き換え
for dataset in combine:
    # 1. train_df, test_dfのTitleで、'Lady', 'Countess','Capt', 'Col',　'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'以外の項目に関しては'Rare'に書き換え
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    #同様に、MileはMissに書き換えてください。
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    #MmeはMrsに書き換えてください。
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#TitleとSurvivedで、Titleで集計してSurvivedの平均値を算出    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#これらの敬称を、予測モデルにしやすいように順序データに変換
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:
    dataset["Title"] = dataset["Title"] .map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    dataset['Title'] = dataset['Title'].fillna(0)

#NameとPassangerIdをtrain_dfから削除----------------------------------------------------
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

#Nameをtest_dfから削除
test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]
#train_dfとtest_dfの行数と列数を確認。
#print(train_df.shape, test_df.shape)    
    
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#先頭行を抽出
#print(train_df.head())

#連続値であるAgeを任意の個数で分割し離散値に変換----------------------------------------------
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

#AgeBandとSurvivedのピボットテーブルを作成
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
    
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]
#ParchとSibSpを足し合わせた、FamilySizeという特徴量を新規に作成
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
#FamilySizeとSurvivedの平均値をグループで集計してください。--------------------------------
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#isAloneという新しい特徴量を作成します。この特徴量には独身者なら1、家族持ちなら0が格納---------
for dataset in combine:
    # 1. IsAloneという特徴量を全て0として作成
    dataset['IsAlone'] = 0
    # 2. FamilySizeが1の時、isAloneを1。
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
#IsAloneとSurvivedをグループ集計して、Survivedの平均値を出力
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

#train_dfとtest_dfからParch、SibSp、およびFamilySizeを削除------------------------------
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

#Embarkedでnaを削除し、最頻値を新たな特徴量freq_portに格納="S"---------------------------
freq_port = train_df.Embarked.dropna().mode()[0]
#train_dfとtest_dfの欠損値を、freq_portの値に置き換え
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

#EmbarkedとSurvivedをグループ集計して、Survivedの平均値を出力
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Embarkedを{'S': 0, 'C': 1, 'Q': 2}に置き換えて下さい。
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#test_dfのFareの欠損値を、Fareのmedianで埋める----------------------------------------
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

#Fareを4階層に分けたFareBandという特徴量を作成-----------------------------------------

#連続値であるFareを4個に分割し離散値へ変換
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

#FareBandとSurvivedのピボットテーブルを作成
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
#Fareの値が7.91以下なら0、7.91超え14.454以下なら1、14.454超え31以下なら2、31超え3へ変換
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand','Survived'], axis=1)
test_df = test_df.drop(['PassengerId'], axis=1)
combine = [train_df, test_df]

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean()) 
train_df['Age*Class'] = train_df['Age*Class'].fillna(train_df['Age*Class'].mean()) 
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean()) 
test_df['Age*Class'] = test_df['Age*Class'].fillna(test_df['Age*Class'].mean()) 
    
train_x = train_df.copy()
test_x = test_df.copy()

# 学習データを学習データとバリデーションデータに分ける------------------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

#条件設定
max_score = 0
SearchMethod = 0
LR_grid = {LogisticRegression(): {"C": [10 ** i for i in range(-5, 6)],
                                  "random_state": [i for i in range(0, 101)]}}
LR_random = {LogisticRegression(): {"C": scipy.stats.uniform(0.00001, 1000),
                                    "random_state": scipy.stats.randint(0, 100)}}


#ランダムサーチ
for model, param in LR_random.items():
    clf =RandomizedSearchCV(model, param)
    clf.fit(tr_x, tr_y)
    pred_y = clf.predict(va_x)
    score = f1_score(va_y, pred_y, average="micro")

    if max_score < score:
        SearchMethod = 1
        max_score = score
        best_param = clf.best_params_
        best_model = model.__class__.__name__

if SearchMethod == 0:
    print("サーチ方法:グリッドサーチ")
else:
    print("サーチ方法:ランダムサーチ")
print("ベストスコア:{}".format(max_score))
print("モデル:{}".format(best_model))
print("パラメーター:{}".format(best_param))

#ハイパーパラメータを調整しない場合との比較
model = LogisticRegression()
model.fit(tr_x, tr_y)
score = model.score(va_x, va_y)
print("")
print("デフォルトスコア:", score)
'''
■GBDTの特徴量の重要度
xgboostは以下の特徴量の重要度を抽出する
・ゲイン：その特徴量の分岐により得た目的関数の減少
・カバー：その特徴量により分岐させられたデータの数(正確には目的関数の二階微分値が使われる)
・頻度：その特徴量が分岐に現れた回数

デフォルトでは頻度が出力されますがゲインの方が特徴量が重要かどうかを表現している

'''

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

#xgboostモデルの実装-----------------------------------------------------------
class Model:

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, tr_x, tr_y):
        #params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
        params.update(self.params)
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        self.model = xgb.train(params, dtrain, num_round)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred

#データの準備-------------------------------------------------------------------
train = pd.read_csv('../input/titanic-data/train.csv')
#NameはすべてユニークでPassengerIDと被るため削除,TicketとCabinも同様
train_x = train.drop(["Name","Ticket", "Cabin", "Survived"], axis=1)
train_y = train['Survived']
test_x = pd.read_csv('../input/titanic-data/test.csv')
test_x = test_x.drop(["Name", "Ticket", "Cabin"], axis=1)

#今回はひとまずlabel型のnullデータは適当な値で埋める→nullデータの決め方はFeature Engineering へ
#train_x = train_x.fillna({'Cabin': 'A00', 'Embarked': "D"})
train_x = train_x.fillna({'Embarked': "D"})
#test_x = test_x.fillna({'Cabin': 'A00'})

#同様に数値型のnullデータはひとまず平均値で埋める
train_x['Age'] = train_x['Age'].fillna(train_x['Age'].mean()) 
test_x['Age'] = test_x['Age'].fillna(test_x['Age'].mean()) 
test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean()) 

#train_x.info()
#test_x.info()

#変換するlabel変数のリスト
label_cols = ["Sex", "Embarked"]

#カテゴリ変数をループしてlabel encoding---------------------------------------------
#テストデータにはトレーニングデータセットに含まれていない値があるとうまく変換できない
for c in label_cols:
    # 学習データに基づいて定義する
    le = LabelEncoder()
    le.fit(train_x[c])
    train_x[c] = le.transform(train_x[c])
    test_x[c] = le.transform(test_x[c])
    
#train_x.info()
#test_x.info()

print(train_x)
print(test_x)

# クロスバリデーション-------------------------------------------------------------
#モデルの学習・評価のため、予測値を出すのは別
# 学習データを4つに分け、うち1つをバリデーションデータとする
# どれをバリデーションデータとするかを変えて学習・評価を4回行う
scores_ll = []
params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71, 'eval_metric': 'logloss'}

kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    model = Model(params)
    model.fit(tr_x, tr_y)
    va_pred = model.predict(va_x)
    score_ll = log_loss(va_y, va_pred)
    scores_ll.append(score_ll) 

print(scores_ll)
# クロスバリデーションの平均のスコアを出力する
print(f'logloss: {np.mean(scores_ll):.4f}')
print()
#print(list(kf.split(train_x)))
#この時点でtr_x,va_xはなにがはいっているか
#print(tr_x)
#print(va_pred.shape)

# 学習データとバリデーションデータのスコアのモニタリング----------------------------------
# モニタリングをloglossで行い、アーリーストッピングの観察するroundを20とする
dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)
dtest = xgb.DMatrix(test_x)

num_round = 500
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist, early_stopping_rounds=20)

# 重要度の上位を出力する-------------------------------------------------------------
fscore = model.get_score(importance_type='total_gain')
fscore = sorted([(k, v) for k, v in fscore.items()], key=lambda tpl: tpl[1], reverse=True)
print('xgboost importance')
print(fscore[:5])

%%timeit
'''
■反復して特徴量を探索する方法
# Greedy Forward Selectionを単純化した手法
1．使用する特徴量の集合を空から始める(この集合をMとする)
2．候補となる特徴量を有望な順番もしくはランダムな順番に並べる
3．次の特徴量を加えることでスコアがよくなればMに加える。そうでなければ加えない
4．3をすべての候補について繰り返す

'''

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

#データの前処理----------------------------------------------------------------------------
train = pd.read_csv('../input/titanic-data/train.csv')
#NameはすべてユニークでPassengerIDと被るため削除
train_x = train.drop(["Name", "Survived"], axis=1)
train_y = train['Survived']
test_x = pd.read_csv('../input/titanic-data/test.csv')
test_x = test_x.drop(["Name"], axis=1)

#今回はひとまずlabel型のnullデータは適当な値で埋める→nullデータの決め方はFeature Engineering へ
train_x = train_x.fillna({'Cabin': 'A00', 'Embarked': "D"})
test_x = test_x.fillna({'Cabin': 'A00'})

#同様に数値型のnullデータはひとまず平均値で埋める
train_x['Age'] = train_x['Age'].fillna(train_x['Age'].mean()) 
test_x['Age'] = test_x['Age'].fillna(test_x['Age'].mean()) 
test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean()) 

#train_x.info()
#test_x.info()

#データの前処理-カテゴリ変数の数値化----------------------------------------------------------
label_cols = ["Sex", "Ticket", "Cabin", "Embarked"]

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', categories='auto')
ohe.fit(train_x[label_cols])

#print(ohe.categories_)

# ダミー変数の列名の作成
columns = []
for i, c in enumerate(label_cols):
    columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# 生成されたダミー変数をデータフレームに変換
dummy_vals_train = pd.DataFrame(ohe.transform(train_x[label_cols]), columns=columns)
dummy_vals_test = pd.DataFrame(ohe.transform(test_x[label_cols]), columns=columns)

#print(dummy_vals_train)
#print(dummy_vals_test)

# 残りの変数と結合元のデータフレームに結合
train_x = pd.concat([train_x.drop(label_cols, axis=1), dummy_vals_train], axis=1)
test_x = pd.concat([test_x.drop(label_cols, axis=1), dummy_vals_test], axis=1)

# 学習データを学習データとバリデーションデータに分ける------------------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# 特徴量のリストに対して精度を評価するevaluate関数の定義
def evaluate(features):
    dtrain = xgb.DMatrix(tr_x[features], label=tr_y)
    dvalid = xgb.DMatrix(va_x[features], label=va_y)
    params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
    num_round = 10  # 実際にはもっと多いround数が必要
    early_stopping_rounds = 3
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    model = xgb.train(params, dtrain, num_round,
                      evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                      verbose_eval=0)
    va_pred = model.predict(dvalid)
    score = log_loss(va_y, va_pred)

    return score

best_score = 9999.0
candidates = np.random.RandomState(71).permutation(train_x.columns)
selected = set([])

print('start simple selection')
for feature in candidates:
    # 特徴量のリストに対して精度を評価するevaluate関数があるものとする
    fs = list(selected) + [feature]
    score = evaluate(fs)

    # スコアは低い方が良いとする
    if score < best_score:
        selected.add(feature)
        best_score = score
        print(f'selected:{feature}')
        print(f'score:{score}')

print(f'selected features: {selected}')
'''
■反復して特徴量を探索する方法
# Greedy Forward Selection(精度重視)
1．使用する特徴量の集合を空から始める(この集合をMとする)
2．候補となる特徴量を有望な順番もしくはランダムな順番に並べる
3．次の特徴量を加えることでスコアがよくなればMに加える。そうでなければ加えない
4．3をすべての候補について繰り返す

'''
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

#データの前処理----------------------------------------------------------------------------
train = pd.read_csv('../input/titanic-data/train.csv')
#NameはすべてユニークでPassengerIDと被るため削除
train_x = train.drop(["Name", "Survived"], axis=1)
train_y = train['Survived']
test_x = pd.read_csv('../input/titanic-data/test.csv')
test_x = test_x.drop(["Name"], axis=1)

#今回はひとまずlabel型のnullデータは適当な値で埋める→nullデータの決め方はFeature Engineering へ
train_x = train_x.fillna({'Cabin': 'A00', 'Embarked': "D"})
test_x = test_x.fillna({'Cabin': 'A00'})

#同様に数値型のnullデータはひとまず平均値で埋める
train_x['Age'] = train_x['Age'].fillna(train_x['Age'].mean()) 
test_x['Age'] = test_x['Age'].fillna(test_x['Age'].mean()) 
test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean()) 

#train_x.info()
#test_x.info()

#データの前処理-カテゴリ変数の数値化----------------------------------------------------------
label_cols = ["Sex", "Ticket", "Cabin", "Embarked"]

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', categories='auto')
ohe.fit(train_x[label_cols])

#print(ohe.categories_)

# ダミー変数の列名の作成
columns = []
for i, c in enumerate(label_cols):
    columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# 生成されたダミー変数をデータフレームに変換
dummy_vals_train = pd.DataFrame(ohe.transform(train_x[label_cols]), columns=columns)
dummy_vals_test = pd.DataFrame(ohe.transform(test_x[label_cols]), columns=columns)

#print(dummy_vals_train)
#print(dummy_vals_test)

# 残りの変数と結合元のデータフレームに結合
train_x = pd.concat([train_x.drop(label_cols, axis=1), dummy_vals_train], axis=1)
test_x = pd.concat([test_x.drop(label_cols, axis=1), dummy_vals_test], axis=1)

# 学習データを学習データとバリデーションデータに分ける------------------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# 特徴量のリストに対して精度を評価するevaluate関数の定義
def evaluate(features):
    dtrain = xgb.DMatrix(tr_x[features], label=tr_y)
    dvalid = xgb.DMatrix(va_x[features], label=va_y)
    params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
    num_round = 10  # 実際にはもっと多いround数が必要
    early_stopping_rounds = 3
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    model = xgb.train(params, dtrain, num_round,
                      evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                      verbose_eval=0)
    va_pred = model.predict(dvalid)
    score = log_loss(va_y, va_pred)

    return score

best_score = 9999.0
selected = set([])

print('start greedy forward selection')

while True:

    if len(selected) == len(train_x.columns):
        # すべての特徴が選ばれて終了
        break

    scores = []
    for feature in train_x.columns:
        if feature not in selected:
            # 特徴量のリストに対して精度を評価するevaluate関数があるものとする
            fs = list(selected) + [feature]
            score = evaluate(fs)
            scores.append((feature, score))

    # スコアは低い方が良いとする
    b_feature, b_score = sorted(scores, key=lambda tpl: tpl[1])[0]
    if b_score < best_score:
        selected.add(b_feature)
        best_score = b_score
        print(f'selected:{b_feature}')
        print(f'score:{b_score}')
    else:
        # どの特徴を追加してもスコアが上がらないので終了
        break

print(f'selected features: {selected}')
