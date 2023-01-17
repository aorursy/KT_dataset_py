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
df_train = pd.read_csv('/kaggle/input/1056lab-wine-quality-prediction/train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/1056lab-wine-quality-prediction/test.csv', index_col=0)
# 欠損値を含む行を訓練データから削除
df_train = df_train.dropna()

# 色の列を削除（使わない）
df_train = df_train.drop('color', axis=1)
df_test = df_test.drop('color', axis=1)

# 欠損に中央値を補完
df_test = df_test.fillna(df_test.median())  
df_train  # 内容を確認
df_test  # 内容を確認
import seaborn as sns
from matplotlib import pyplot

sns.set_style("darkgrid")
pyplot.figure(figsize=(10, 10))  # 図の大きさを大き目に設定
sns.heatmap(df_train.corr(), square=True, annot=True)  # 相関係数でヒートマップを作成
df_train = df_train[['alcohol', 'density', 'volatile acidity', 'chlorides', 'quality']]  # 列を選択
df_test = df_test[['alcohol', 'density', 'volatile acidity', 'chlorides']]  # 列を選択
df_train # 内容を確認
df_test  # 内容を確認
from sklearn.decomposition import PCA

X_train = df_train.drop('quality', axis=1).values  # 目的変数を除いてndarray化
pca = PCA()  # 次元圧縮なし
pca.fit(X_train)  # 主成分分析
sns.set_style("darkgrid")
ev_ratio = pca.explained_variance_ratio_
ev_ratio = np.hstack([0,ev_ratio.cumsum()])
sns.lineplot(data=ev_ratio)
X_train = df_train.drop('quality', axis=1).values
pca = PCA(n_components=2)  # 2次元
pca.fit(X_train)  # 主成分分析
X_train_pca = pca.transform(X_train)  # 訓練データを2次元に変換
df_train_pca = pd.DataFrame(X_train_pca, columns=['Comp. 1', 'Comp. 2'], index=df_train.index)
df_train_pca['quality'] = df_train['quality']

sns.set_style("darkgrid")
sns.relplot(data=df_train_pca, x='Comp. 1', y='Comp. 2', hue='quality')
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train = df_train.drop('quality', axis=1).values
y_train = df_train['quality'].values
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)  # 訓練用と検証用に分ける
model = SVR(gamma='auto')
model.fit(X_train, y_train)  # 訓練用で学習
predict = model.predict(X_valid)
mean_squared_error(predict, y_valid)  # 検証用で評価（デフォルトの精度を確認する）
import optuna
from sklearn.svm import SVR

def objective(trial):
    #kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])  # カーネル
    svr_c = trial.suggest_loguniform('svr_c', 1e0, 1e2)  # C
    epsilon = trial.suggest_loguniform('epsilon', 1e-1, 1e1)  # epsilon
    #gamma = trial.suggest_loguniform('gamma', 1e-3, 3e1)  # gamma
    model = SVR(kernel='rbf', C=svr_c, epsilon=epsilon, gamma='auto')  # SVR
    model.fit(X_train, y_train)  # 訓練用で学習
    y_pred = model.predict(X_valid)  # 検証用を予測
    return mean_squared_error(y_valid, y_pred) # 検証用に対する予測を評価

study = optuna.create_study()  # Oputuna
study.optimize(objective, n_trials=100)  # 最適か
study.best_params  # 最適パラメーター
svr_c = study.best_params['svr_c']  # Cの最適値
epsilon = study.best_params['epsilon']  # epsilonの最適値
model = SVR(kernel='rbf', C=svr_c, epsilon=epsilon, gamma='auto')  # 最適パラメーターのSVR
model.fit(X_train, y_train)  # 訓練用で学習
predict = model.predict(X_valid)  # 検証用を予測
mean_squared_error(predict, y_valid)  # 検証用に対する予測を評価
X_valid_pca = pca.transform(X_valid)
df_valid_pca = pd.DataFrame(X_valid_pca, columns=['Comp. 1', 'Comp. 2'])
df_valid_pca['quality'] = predict

sns.set_style("darkgrid")
sns.relplot(data=df_valid_pca, x='Comp. 1', y='Comp. 2', hue='quality')
from sklearn.svm import SVR

X_train = df_train.drop('quality', axis=1).values
y_train = df_train['quality'].values

model = SVR(kernel='rbf', C=svr_c, epsilon=epsilon, gamma='auto')  # 最適パラメーターのSVR
model.fit(X_train, y_train)
X_test = df_test.values
predict = model.predict(X_test)
X_test_pca = pca.transform(X_test)
df_test_pca = pd.DataFrame(X_test_pca, columns=['Comp. 1', 'Comp. 2'])
df_test_pca['quality'] = predict

sns.set_style("darkgrid")
sns.relplot(data=df_test_pca, x='Comp. 1', y='Comp. 2', hue='quality')
submit = pd.read_csv('/kaggle/input/1056lab-wine-quality-prediction/sampleSubmission.csv')
submit['quality'] = predict
submit.to_csv('submission.csv', index=False)
