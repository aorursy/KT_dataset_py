# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/titanic/train.csv') # dfはDataFrameのこと

print('===データ型を表示===')

print(df.dtypes) # データ型を表示

print('===欠損値の個数を表示===')

print(df.isnull().sum()) # 欠損値の個数を表示
# Sexを変換

# One-Hot値化はpandas.get_dummiesで行う

# "drop_first=True"は変換後の最初の列を削除する(n個の名義尺度を表現するには、(n-1)個の列で足りるため)

df_dummy = pd.get_dummies(df['Sex'], drop_first=True)

df = pd.concat([df.drop(['Sex'],axis=1),df_dummy],axis=1)



# Embarkedは欠損値があるため、ここでは変換しない



print(df.dtypes) # データ型を表示
df.hist()
import seaborn as sns

sns.heatmap(df.corr(),annot = True)
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



# 学習用に、乗客情報と生存情報を分離

df_data = df.loc[:, ['Pclass', 'male']] # Pclassとmaleを抜き出した乗客情報を作成

df_survived = df['Survived'] # Survivedのみで生存情報を作成



# ロジスティック回帰モデルを生成

#   random_state=0: 実行ごとの結果のばらつきをなくすため乱数を固定

clf = LogisticRegression(random_state=0)



clf.fit(df_data, df_survived) # 学習用データを用いて学習



# 交差検証(Cross Validation)のメソッドを設定

# ※後述のcross_val_scoreを使うだけでできるが、乱数固定のためここでメソッド設定している

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)



# 交差検証(Cross Validation)の実施

print('===交差検証の結果===')

print(cross_val_score(clf, df_data, df_survived, cv=cv))
df_test = pd.read_csv('/kaggle/input/titanic/test.csv') # test.csvを読み込み



# SexをOne-Hot値に変換

df_dummy = pd.get_dummies(df_test['Sex'], drop_first=True)

df_test = pd.concat([df_test.drop(['Sex'],axis=1),df_dummy],axis=1)



df_test_data = df_test.loc[:, ['Pclass', 'male']] # PclassとSexを抜き出した乗客情報を作成



test_result = clf.predict(df_test_data) # テストデータを予測



# 提出用ファイルを作成

df_out = pd.read_csv("/kaggle/input/titanic/test.csv")

df_out["Survived"] = test_result # 予測結果をSurvived列として追加

df_out[["PassengerId","Survived"]].to_csv("/kaggle/working/titanic_result.csv",index=False) # 必要な列だけ抽出して提出用ファイルに保存
df_data = df.loc[:, ['Pclass', 'Age', 'SibSp', 'Parch', 'Embarked','male']].copy() # 学習に使用する乗客情報を再度抽出



# ランダム値を設定するために、まずはEmbarkedを整数化しておく

df_data['Embarked'] = df_data['Embarked'].replace(['C', 'S', 'Q'], [0, 1, 2])



# 欠損値をランダムな整数に置き換える関数を定義

def replase_nan_to_random_value(data):

    mean = data.mean() #平均値

    std = data.std()  #標準偏差

    nullcount = data.isnull().sum() #null値の数＝補完する数



    # 正規分布に従うとし、標準偏差の範囲内でランダムに整数を作る

    rand = np.random.randint(mean - std, mean + std , size = nullcount)



    data[np.isnan(data)] = rand # 欠損値をランダム値に置き換える



replase_nan_to_random_value(df_data['Age']) # Ageの欠損値補完

replase_nan_to_random_value(df_data['Embarked']) # Enbarkedの欠損値補完



print(df_data.isnull().sum()) # 欠損値の個数を表示



# EnbarkedをOne-Hot値化する。

df_dummy = pd.get_dummies(df_data['Embarked'], drop_first=True)

df_data = pd.concat([df_data.drop(['Embarked'],axis=1),df_dummy],axis=1)



# 学習用データを用いて再度学習

clf.fit(df_data, df_survived)



# 交差検証(Cross Validation)の実施

print('===交差検証の結果(データ追加版)===')

print(cross_val_score(clf, df_data, df_survived, cv=cv))
from sklearn.preprocessing import StandardScaler



stdsc = StandardScaler()

df_data_std = pd.DataFrame(stdsc.fit_transform(df_data), columns=df_data.columns) # 学習用データを標準化



# 標準化した学習用データで再度学習

clf.fit(df_data_std, df_survived)



# 交差検証(Cross Validation)の実施

print('===交差検証の結果(標準化版)===')

print(cross_val_score(clf, df_data_std, df_survived, cv=cv))
from sklearn import svm



clf_svm = svm.SVC(random_state=0) # SVMモデルを生成

clf_svm.fit(df_data, df_survived) # 学習データを用いて学習



# 交差検証(Cross Validation)の実施

print('===交差検証の結果(SVM)===')

print(cross_val_score(clf_svm, df_data, df_survived, cv=cv))



clf_svm.fit(df_data_std, df_survived) # 学習データを用いて学習

print('===交差検証の結果(SVM+標準化)===')

print(cross_val_score(clf_svm, df_data_std, df_survived, cv=cv))
from sklearn.ensemble import RandomForestClassifier



clf_rf = RandomForestClassifier(random_state=0)

clf_rf.fit(df_data, df_survived) # 学習データを用いて学習



# 交差検証(Cross Validation)の実施

print('===交差検証の結果(ランダムフォレスト)===')

print(cross_val_score(clf_rf, df_data, df_survived, cv=cv))



# ランダムフォレストの場合、標準化は不要
df_test_data = df_test.loc[:, ['Pclass', 'Age', 'SibSp', 'Parch', 'Embarked', 'male']].copy()  # 必要部分を抜き出した乗客情報を作成



replase_nan_to_random_value(df_test_data['Age']) # Ageの欠損値補完



# Enbarkedの欠損値補完

df_test_data['Embarked'] = df_test_data['Embarked'].replace(['C', 'S', 'Q'], [0, 1, 2])

replase_nan_to_random_value(df_test_data['Embarked'])



# EnbarkedをOne-Hot値化

df_dummy = pd.get_dummies(df_test_data['Embarked'], drop_first=True)

df_test_data = pd.concat([df_test_data.drop(['Embarked'],axis=1),df_dummy],axis=1)



df_test_data_std = pd.DataFrame(stdsc.transform(df_test_data), columns=df_test_data.columns) # 学習用データを標準化



test_result = clf_svm.predict(df_test_data_std) # テストデータを予測



# 提出用ファイルを作成

df_out["Survived"] = test_result # 予測結果をSurvived列として追加

df_out[["PassengerId","Survived"]].to_csv("/kaggle/working/titanic_result.csv",index=False) # 必要な列だけ抽出して提出用ファイルに保存