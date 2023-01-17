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
# data wrangling
import numpy as np
import pandas as pd
import pandas_profiling as pdp
from collections import Counter

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import display

# modeling
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate

# evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#1.5.1 タスクと評価指標
# Load
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")

# fundamental statistics
#display(train.describe(include='all'))

# generate detailed report(train)
#pdp.ProfileReport(train)

#学習データを特徴量と目的変数に分ける
train_x = train.drop(["Survived"],axis=1)
train_y = train["Survived"]
#1.5.2 特徴量の作成
from sklearn.preprocessing import LabelEncoder

#変数PassengerIDを削除する
train_x = train_x.drop(["PassengerId"],axis=1)
test_x = test.drop(["PassengerId"],axis=1)

#変数Name, Ticket, Cabinを除外する
train_x = train_x.drop(["Name","Ticket","Cabin"],axis=1)
test_x = test_x.drop(["Name","Ticket","Cabin"],axis=1)

#それぞれのカテゴリ変数にlabel encodingを適用する
#LabelEncoderを使って、名称を数値に変換する
for c in ["Sex", "Embarked"]:
    #学習データに基づいて、どう変換するか定める
    le = LabelEncoder()
    le.fit(train_x[c].fillna("NA"))
    
    #学習データ、テストデータを変換する
    train_x[c]=le.transform(train_x[c].fillna("NA"))
    test_x[c]=le.transform(test_x[c].fillna("NA"))

#display(train_x)
#1.5.3 モデルの作成
from xgboost import XGBClassifier

#モデルを与えての作成および学習データを与えての学習
model = XGBClassifier(n_estimators=20,random_state=71)
model.fit(train_x,train_y)

#テストデータの予測値を確率で出力する
pred = model.predict_proba(test_x)[:,1]
#テストデータの予測値を二値に変換する
pred_label = np.where(pred>0.5,1,0)

#提出用ファイルの作成
submission = pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":pred_label})
submission.to_csv("submission_first.csv", index=False)
#1.5.4モデルの評価
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

#各foldのスコアを保存するリスト
scores_accuracy = []
scores_logloss = []

#クロスバリデーションを行う
#学習データを4つに分割し、うち1つをバリデーションデータとすることをバリデーションデータを変えて繰り返す
kf = KFold(n_splits=4, shuffle =True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    #学習データを学習データとバリデーションデータに分ける
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    
    #モデルの学習を行う
    model = XGBClassifier(n_estimators=20, random_state=71)
    model.fit(tr_x,tr_y)
    
    #バリデーションデータの予測値を確率で出力する
    va_pred = model.predict_proba(va_x)[:,1]
    
    #バリデーションデータでスコアを計算する
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred>0.5)
    
    #そのfoldスコアを保持する
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)
print(np.mean(scores_accuracy),np.mean(scores_logloss))
#1.5.5モデルのチューニング
#グリッドサーチにより、ハイパーパラメータをチューニングする

import itertools
#チューニング候補とするパラメータを準備する
param_space = {
    "max_depth":[3,5,7],
    "min_child_weight":[1.0,2.0,4.0]
}

#探索するハイパーパラメータの組み合わせ
param_combinations = itertools.product(param_space["max_depth"],param_space["min_child_weight"])
#各パラメータの組み合わせとそれに対するスコアを保存するリスト
params = []
scores = []

#各パラメータの組み合わせごとにクロスバリデーションで評価を行う
for max_depth, min_child_weight in param_combinations:
    score_folds = []
    #クロスバリデーションを行う
    #学習データを4つに分割し、うち1つをバリデーションデータとすることをバリデーションデータを変えて繰り返す
    kf = KFold(n_splits=4, shuffle =True, random_state=71)
    
    for tr_idx, va_idx in kf.split(train_x):
        #学習データを学習データとバリデーションデータに分ける
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        
        #モデルの学習を行う
        model = XGBClassifier(n_estimators=20, random_state=71, max_depth=max_depth, min_child_weight=min_child_weight)
        model.fit(tr_x,tr_y) 
        #バリデーションデータの予測値を確率で出力する
        va_pred = model.predict_proba(va_x)[:,1]
    
        #バリデーションデータでスコアを計算する
        logloss = log_loss(va_y, va_pred)
    
        #そのfoldスコアを保持する
        score_folds.append(logloss)
            
    score_mean = np.mean(score_folds)
    
    #パラメータの組み合わせと対応するスコアを保存する
    params.append((max_depth,min_child_weight))
    scores.append(score_mean)

#最もスコアがよいものをベストなパラメータとする
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'max_depth:{best_param[0]},min_child_weight:{best_param[1]}')
# xgboostモデル #1.5.5 で見つけた最適化パラメータを代入した
model_xgb = XGBClassifier(n_estimators=20, random_state=71,max_depth=3, min_child_weight=2)
model_xgb.fit(train_x,train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:,1]

pred_label2 = np.where(pred_xgb>0.5,1,0)

#提出用ファイルの作成
submission3 = pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":pred_label2})
submission3.to_csv("submission_second_2.csv", index=False)
train = pd.read_csv('../input/titanic/train.csv')
test  = pd.read_csv('../input/titanic/test.csv')

print(pd.crosstab([train['Survived']],[train['Sex'],train['Pclass']]))
y_train = train['Survived'].copy()
train.drop('Survived', axis=1, inplace=True)

# 処理しやすいようにまとめておく
train_test = (train, test)
# 不要列の削除
# test データの PassengerId は Submit に必要になるため取っておく
passengerid = train_test[1]['PassengerId']
for df in train_test:
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
#欠損値の補完　→　代表値（平均値）で補完する
for df in train_test:
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)

#欠損値の補完　→　代表値（最頻値）で補完する
for df in train_test:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

#欠損値から新たな特徴量を生成するcabinのデータを削除せずに0か1かで残す、「欠損していること」自体が情報かも知れないので
for df in train_test:
    df['Cabin'] = df['Cabin'].isnull().astype(int)
#モデルが受け付けるデータ型に変換する
#数値化(Sex)
for df in train_test:
    df['Sex'] = (df['Sex']=='female').astype(int) # female -> 1, male -> 0
    
#数値化（Embarked）出現頻度順に 0, 1, 2 とする
for df in train_test:
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
from sklearn.svm import SVC
#標準化（標準化していないデータも裏でsubmitした）したうえで、SVMを用いて予測を行う。
X_train = train_test[0].copy()
X_test  = train_test[1].copy()

scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

model = SVC(kernel='linear', random_state=123, gamma=0.05, C=1.0)
model.fit(X_train_std, y_train)
y_pred = model.predict(X_test_std)

submission_scaled = pd.DataFrame({
        "PassengerId": passengerid,
        "Survived": y_pred
    })
display(submission_scaled.head())
submission_scaled.to_csv('./submission_scaled.csv', index=False)
#主な称号は次の4つ["Mr.","Miss.","Mrs.","Master."]。※["Rev.","Dr."]は省略
#これらの称号にそれぞれ1,2,3,4を割り振る。その他は0とする。
#import re ←使わなかった

train = pd.read_csv('../input/titanic/train.csv')
train["Title"] = 0
test  = pd.read_csv('../input/titanic/test.csv')
test["Title"] = 0

y_train = train['Survived'].copy()
train.drop('Survived', axis=1, inplace=True)

# 処理しやすいようにまとめておく
train_test = (train, test)

# 不要列の削除
# test データの PassengerId は Submit に必要になるため取っておく
passengerid = train_test[1]['PassengerId']
for df in train_test:
    df.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
title =["Mr.","Miss.","Mrs.","Master."]
#n = len(train_test[1]["Name"])

for df in train_test:
    for i in range(len(df["Age"])):
        for j in title:
            if j in df["Name"][i]:
                df["Title"][i] = j
pd.crosstab(index=train_test[0]['Title'],columns=train_test[0]['Sex'])
#数値化（Title）出現頻度順に["Mr.","Miss.","Mrs.","Master."]= 4, 3, 2, 1 とする
for df in train_test:
    df['Title'] = df['Title'].map({"Mr.": 4, 'Miss.': 3, 'Mrs.': 2, 'Master.': 1,0:0})

# 不要列(Name)の削除
for df in train_test:
    df.drop(['Name'], axis=1, inplace=True)
pd.crosstab(index=train_test[0]['Title'],columns=train_test[0]['Sex'])
#欠損値の補完　→　代表値（平均値）で補完する
for df in train_test:
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)

#欠損値の補完　→　代表値（最頻値）で補完する
for df in train_test:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

#欠損値から新たな特徴量を生成するcabinのデータを削除せずに0か1かで残す、「欠損していること」自体が情報かも知れないので
for df in train_test:
    df['Cabin'] = df['Cabin'].isnull().astype(int)
    
#モデルが受け付けるデータ型に変換する
#数値化(Sex)
for df in train_test:
    df['Sex'] = (df['Sex']=='female').astype(int) # female -> 1, male -> 0
    
#数値化（Embarked）出現頻度順に 0, 1, 2 とする
for df in train_test:
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
#標準化したうえで、SVMを用いて予測を行う。
X_train = train_test[0].copy()
X_test  = train_test[1].copy()

scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

model = SVC(kernel='linear', random_state=123, gamma=0.05, C=1.0)
model.fit(X_train_std, y_train)
y_pred = model.predict(X_test_std)

submission_scaled = pd.DataFrame({
        "PassengerId": passengerid,
        "Survived": y_pred
    })
display(submission_scaled.head())
submission_scaled.to_csv('./submission_title_scaled.csv', index=False)
#print(plt.hist(train_test[0]["Age"],bins = 16,range=(0, 80)))

bin_edges = [0,15,50,float("inf")]
binned_age = pd.cut(train_test[0]["Age"],bin_edges,labels=False)

#Ageのデータを「子供、老人、その他」の3値に分類する
for df in train_test:
    df['Age'] = pd.cut(df["Age"],bin_edges,labels=False)
#Ageのデータが3値にbinningされた状態で、SVM（標準化）を用いて予測を行う。
X_train = train_test[0].copy()
X_test  = train_test[1].copy()

scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

model = SVC(kernel='linear', random_state=123, gamma=0.05, C=1.0)
model.fit(X_train_std, y_train)
y_pred = model.predict(X_test_std)

submission_scaled = pd.DataFrame({
        "PassengerId": passengerid,
        "Survived": y_pred
    })
display(submission_scaled.head())
submission_scaled.to_csv('./submission_title_Agebin_scaled.csv', index=False)
#学習データをtarget encording用のfoldに分割してtarget encordingを適用する。
from sklearn.model_selection import KFold


cat_cols = ["Pclass","Sex","Age","SibSp","Parch","Cabin","Embarked","Title"]

# 変数をループしてtarget encoding
for df in train_test:
    for c in cat_cols:
        # 学習データ全体で各カテゴリにおけるtargetの平均を計算
        data_tmp = pd.DataFrame({c: df[c], 'target': train_y})
        target_mean = data_tmp.groupby(c)['target'].mean()
        # テストデータのカテゴリを置換
        df[c] = df[c].map(target_mean)
        # 学習データの変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, df.shape[0])
        
        # 学習データを分割
        kf = KFold(n_splits=4, shuffle=True, random_state=72)
        for idx_1, idx_2 in kf.split(df):
            # out-of-foldで各カテゴリにおける目的変数の平均を計算
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            # 変換後の値を一時配列に格納
            tmp[idx_2] = df[c].iloc[idx_2].map(target_mean)

        # 変換後のデータで元の変数を置換
        df[c] = tmp

train_x.head()
train_x = train_test[0].drop(["Parch"],axis = 1)
model = SVC(kernel='linear', random_state=123, gamma=0.05, C=1.0)
model.fit(train_x, y_train)

test_x = train_test[1].drop(["Parch"],axis = 1)
y_pred = model.predict(test_x)

submission_scaled = pd.DataFrame({
        "PassengerId": passengerid,
        "Survived": y_pred
    })
display(submission_scaled.head())
submission_scaled.to_csv('./submission_title_Agebin_target_scaled.csv', index=False)
train_test[0].isnull().sum()
train_test[0].head()
