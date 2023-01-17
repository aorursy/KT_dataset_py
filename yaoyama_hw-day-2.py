# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#ks_projects_201612 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")

df = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")



# df = pd.read_csv("../kaggle/input/ks-projects-201801.csv")
display(df.head())

# display(df.tail())
# 使ってはいけない列削除

df.drop(['pledged', 'backers', 'backers', 'usd pledged', 'usd_pledged_real'], axis=1) 
df.dtypes
# objectをdatetimeに変換

df.deadline = pd.to_datetime(df.deadline)

df.launched = pd.to_datetime(df.launched)

df['period'] = (df.deadline - df.launched).dt.days



# nameの文字数を入れてみる

df['len_name'] = df.name.apply(lambda x: len(str(x).replace(' ', '')))
display(df.state.unique())
# failedとsuccessful以外を削除

df=df[df['state'].isin(['failed', 'successful'])]

df_failed=df[df['state'].isin(['failed'])]

df_successful=df[df['state'].isin(['successful'])]



print('successful : {:.0f}'.format(len(df_successful)))

print('failed     : {:.0f}'.format(len(df_failed)))
# 欠損値の確認

df.isnull().sum(axis = 0)
import seaborn as sns

sns.heatmap(df.corr(), fmt='g', cmap='Blues')

plt.show()
# usd_goal_realのグラフ

plt.figure()

plt.hist(np.log10(df_successful['usd_goal_real']), bins=100, alpha=0.3, histtype='stepfilled', color='b', log=True)

plt.hist(np.log10(df_failed['usd_goal_real']), bins=100, alpha=0.3, histtype='stepfilled', color='r', log=True)

plt.legend(['successful','failed'])

plt.xlabel('log10(usd_goal_real)')
# periodのグラフ

plt.figure()

plt.hist(df_successful['period'], bins=92, alpha=0.3, histtype='stepfilled', color='b', log=True)

plt.hist(df_failed['period'], bins=92, alpha=0.3, histtype='stepfilled', color='r', log=True)

plt.legend(['successful','failed'])

plt.xlabel('period')



df_temp = df.groupby('period')

df_temp = df_temp['state'].value_counts(normalize=True).unstack()

df_temp[['successful','failed']].plot(kind='bar', stacked=True, figsize=(20,5))

plt.xlabel('period')



df_temp = df.groupby('period')

df_temp = df_temp['state'].value_counts().unstack()

df_temp[['successful','failed']].plot(kind='bar', stacked=True, figsize=(20,5), log=True)

plt.xlabel('period')
# main_categoryのグラフ

df_temp = df.groupby('main_category')

df_temp = df_temp['state'].value_counts(normalize=True).unstack()

df_temp = df_temp.sort_values(by=['successful'],ascending=True)

df_temp[['successful','failed']].plot(kind='bar', stacked=True,figsize=(20,5))



df_temp = df.groupby('main_category')

df_temp = df_temp['state'].value_counts().unstack()

df_temp = df_temp.sort_values(by=['successful'],ascending=True)

df_temp[['successful','failed']].plot(kind='bar', stacked=True,figsize=(20,5))
# categoryのグラフ

df_temp = df.groupby('category')

df_temp = df_temp['state'].value_counts(normalize=True).unstack()

df_temp = df_temp.sort_values(by=['successful'],ascending=True)

df_temp[['successful','failed']].plot(kind='barh', stacked=True,figsize=(13,30))



df_temp = df.groupby('category')

df_temp = df_temp['state'].value_counts().unstack()

df_temp = df_temp.sort_values(by=['successful'],ascending=True)

df_temp[['successful','failed']].plot(kind='barh', stacked=True,figsize=(13,30))
# currencyのグラフ

df_temp = df.groupby('currency')

df_temp = df_temp['state'].value_counts(normalize=True).unstack()

df_temp = df_temp.sort_values(by=['successful'],ascending=True)

df_temp[['successful','failed']].plot(kind='bar', stacked=True,figsize=(20,5))



df_temp = df.groupby('currency')

df_temp = df_temp['state'].value_counts().unstack()

df_temp = df_temp.sort_values(by=['successful'],ascending=True)

df_temp[['successful','failed']].plot(kind='bar', stacked=True,figsize=(20,5))
# countryのグラフ

df_temp = df.groupby('country')

df_temp = df_temp['state'].value_counts(normalize=True).unstack()

df_temp = df_temp.sort_values(by=['successful'],ascending=True)

df_temp[['successful','failed']].plot(kind='bar', stacked=True,figsize=(20,5))



df_temp = df.groupby('country')

df_temp = df_temp['state'].value_counts().unstack()

df_temp = df_temp.sort_values(by=['successful'],ascending=True)

df_temp[['successful','failed']].plot(kind='bar', stacked=True,figsize=(20,5))
# len_nameのグラフ

df_temp = df.groupby('len_name')

df_temp = df_temp['state'].value_counts(normalize=True).unstack()

df_temp = df_temp.sort_values(by=['successful'],ascending=True)

df_temp[['successful','failed']].plot(kind='bar', stacked=True,figsize=(20,5))



df_temp = df.groupby('len_name')

df_temp = df_temp['state'].value_counts().unstack()

df_temp = df_temp.sort_values(by=['successful'],ascending=True)

df_temp[['successful','failed']].plot(kind='bar', stacked=True,figsize=(20,5))
from sklearn.linear_model import SGDClassifier,Ridge,Lasso,ElasticNet

from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support, confusion_matrix

from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler,RobustScaler

from sklearn.model_selection import train_test_split,KFold



from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, confusion_matrix



#ホールドアウト法

y = df['state'].values

X = df[["usd_goal_real", "period"]].values #'state'以外の変数をXに代入する

test_size = 0.2        # 全データのうち、何%をテストデータにするか（今回は20%に設定）

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) # ホールドアウト法を実行（テストデータはランダム選択）



#学習の実行

clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True,tol=1e-3)

clf_std = clf.fit(X_train, y_train)

y_pred_test = clf_std.predict(X_test)
# 結果の計算、蓄積

accuracy = accuracy_score(y_test, y_pred_test) # 正答率

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_test) # 適合率・再現率・F1値



#結果の表示

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy))

print('適合率（Precision） = {:.3f}%'.format(100 * precision[1]))

print('再現率（Recall） = {:.3f}%'.format(100 * recall[1]))

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[1]))



# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred_test), 

                        index=['actual = failed', 'actual = successful'], 

                        columns=['predict = failed', 'predict = successful'])

display(conf_mat)
# 3. 交差検証（クロスバリデーション）法

# 交差検証法とは、データを複数のグループにわけ、テスト役と学習役を交代させていくことで少ないデータでも汎化誤差を評価する方法



from sklearn.model_selection import train_test_split # ホールドアウト法に関する関数

from sklearn.model_selection import KFold # 交差検証法に関する関数

from sklearn.metrics import mean_absolute_error # 回帰問題における性能評価に関する関数



n_split = 5 # グループ数を設定（今回は5分割）



split_num = 1



result_df = pd.DataFrame( columns=['正答率（Accuracy）','適合率（Precision）','再現率（Recall）','F1値（F1-score）'] )



# テスト役を交代させながら学習と評価を繰り返す

for train_idx, test_idx in KFold(n_splits=n_split, shuffle=True, random_state=1234).split(X, y):

    X_train, y_train = X[train_idx], y[train_idx] #学習用データ

    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

    # 学習の実行

    clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True,tol=1e-3)

    clf_std = clf.fit(X_train, y_train)

    y_pred_test = clf_std.predict(X_test)

    

    # 結果の計算、蓄積

    accuracy = accuracy_score(y_test, y_pred_test) # 正答率

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_test) # 適合率・再現率・F1値

    tmp_se = pd.Series( [100 * accuracy, 100 * precision[1], 100 * recall[1], 100 * f1_score[1]], index=result_df.columns )

    result_df = result_df.append( tmp_se, ignore_index=True )



# 結果の表示

result_df2 = pd.concat([result_df,pd.DataFrame(result_df.mean(axis=0),columns=['平均']).T])

result_df2.head(100)