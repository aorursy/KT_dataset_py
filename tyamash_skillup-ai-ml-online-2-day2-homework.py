# Choose my task
print("I choose 'Kickstarter Projects'")

# import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
%matplotlib inline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import Ridge,Lasso,ElasticNet #正則化項付き最小二乗法を行うためのライブラリ
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split # ホールドアウト法に関する関数
from sklearn.model_selection import KFold # 交差検証法に関する関数
from sklearn.svm import SVC
from IPython.core.display import display 
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data is downloaded and ready to use!
#df_data = pd.read_csv("../input/ks-projects-201801.csv")
df_data = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")
print("Row lengths of imported data: ", len(df_data))
#まずはHeaderを確認
display(df_data.head())
df_data.describe()
# 項目ごとの説明を表示する
df_data_exp = pd.read_csv("../input/data-explanation/data_explanation.csv",encoding='cp932')
display(df_data_exp)
print("(KaggleおよびSlackより取得した情報)")
# currencyとcountryの関係
df_currency_country = df_data.groupby('country')
df_currency_country = df_currency_country['currency'].value_counts(normalize=True).unstack(fill_value=0)
display(df_currency_country)

#countryの種類と数を調べる
print(df_data['country'].value_counts(dropna=False))

# 考察
print('countryから異常値と思われる『N,0"』を除けば、countryによりcurrencyが一意に決まる')
print('従って、説明変数からcurrencyを除外することが出来る')
# categoryとmain categoryの関係
df_categories = df_data.groupby('category')
df_categories = df_categories['main_category'].value_counts(normalize=True).unstack(fill_value=0)
display(df_categories)

#categoryの種類と数を調べる
print('categoryの種類と数を調べる')
print(df_data['category'].value_counts(dropna=False))

#main_categoryの種類と数を調べる
print('main_categoryの種類と数を調べる')
print(df_data['main_category'].value_counts(dropna=False))

# 考察
print('AnthologiesやSpacesのような一部の例外を除けば、categoryによりmain_categoryが一意に決まる')
print('従って、説明変数からmain_categoryを除外しても、分析への影響は希少と思料する')
# goalとusd_goal_realの関係
# まずは散布図行列を書いてみる
pd.plotting.scatter_matrix(df_data[['goal','usd_goal_real']], figsize=(15,15))
plt.show()
print(' ')
# 次に相関係数を確認
corr_ = df_data[['goal','usd_goal_real']].corr()
print(corr_)
print(' ')
# 考察
print('goalとusd_goal_realは視覚的に相関が確認出来て、相関係数も0.94と極めて高い')
print('従って、説明変数としてusd_goal_realを使わず、goalで代用する')
#Stateの種類と数を調べる
print(df_data['state'].value_counts(dropna=False))
#Category毎にStateとの相関をグラフ化する
category_ = df_data.groupby('category')
category_ = category_['state'].value_counts(normalize=True).unstack()
category_ = category_.sort_values(by=['successful'],ascending=True)
category_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
print("成功しやすいCategoryと成功しにくいCateogryが存在する")
print("Maxは80%近く、Minは10%以下")
#deadline毎にStateとの相関をグラフ化する
#数が多いので、年月別にする
df_data_deadline = df_data.copy()
df_data_deadline['deadline_YM'] = df_data_deadline['deadline'].apply(lambda x: x[0:7])
deadline_ = df_data_deadline.groupby('deadline_YM')
deadline_ = deadline_['state'].value_counts(normalize=True).unstack()
ax = deadline_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
plt.legend(loc='upper left')
print("2018年にLiveが多い。が、他の明確な傾向はつかみにくい")
#goal毎にStateとの相関をグラフ化する
#goalを10万単位で丸めて相関を見る
df_data_goal = df_data.copy()
df_data_goal['goal_r'] = df_data_goal['goal'].apply(lambda x: round(x/100000))
goal_ = df_data_goal.groupby('goal_r')
goal_ = goal_['state'].value_counts(normalize=True).unstack()
#goal_ = goal_.sort_values('goal_r',ascending=False)
ax = goal_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
plt.legend(loc='upper left')
print("Goalが大き過ぎると、成功しにくいようだ")
#launched毎にStateとの相関をグラフ化する
#数が多いので、年月別にする
df_data_launched = df_data.copy()
df_data_launched['launched_YM'] = df_data_launched['launched'].apply(lambda x: x[0:7])
launched_ = df_data_launched.groupby('launched_YM')
launched_ = launched_['state'].value_counts(normalize=True).unstack()
#launched_ = launched_.sort_values('launched_YM',ascending=False)
ax = launched_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
plt.legend(loc='upper left')
print("1970年に開始したものは失敗、そもそも、これは他から離れた異常値と扱うべきか")
print("2017年12月以降開始はLive、それ以外の傾向は見にくい")
#backers毎にStateとの相関をグラフ化する
#backersを1000単位で丸めて相関を見る
df_data_backers = df_data.copy()
df_data_backers['backers_r'] = df_data_backers['backers'].apply(lambda x: round(x/1000))
backers_ = df_data_backers.groupby('backers_r')
backers_ = backers_['state'].value_counts(normalize=True).unstack()
#backers_ = backers_.sort_values('backers_r',ascending=False)
ax = backers_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,20))
plt.legend(loc='upper left')
print("Backersが一定以上になると、ほぼ成功している")
print("100000以上は連続性に乏しいので外れ値として扱う")
#country毎にStateとの相関をグラフ化する
country_ = df_data.groupby('country')
country_ = country_['state'].value_counts(normalize=True).unstack()
country_ = country_.sort_values(by=['successful'],ascending=True)
country_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,7))
print("成功しやすい国と成功しにくい国が存在するが、")
print("Maxは40%近く、Minは20%以下で幅はそれほど大きくない")
# 欠測値を確認する
df_data.isnull().any(axis=0)
# 考察
print("欠測値はnameとusd pledgedにある")
print("異常値/外れ値は、上記までの分析でcountryとlaunchedとbackersに観察されている")
print("これらの欠測値/異常値/外れ値は、説明変数からは除外する")
#成功かどうかを判断するため、stateが"Successful"なるTrue、それ以外はFalseとする
df_data_test = df_data.copy()
df_data_test['Success'] = df_data_test['state'] == "successful"

# カテゴリー変数をダミー変数に変換
df_data_dummy1 = pd.get_dummies(df_data_test['category'])
df_data_dummy2 = pd.get_dummies(df_data_test['country'])
df_data_test = pd.merge(df_data_test, df_data_dummy1, left_index=True, right_index=True)
df_data_test = pd.merge(df_data_test, df_data_dummy2, left_index=True, right_index=True)

# 欠測値/異常値/外れ値を削除する
df_data_test = df_data_test[df_data_test['country'] != 'N,0"']
df_data_test = df_data_test[df_data_test['launched'] > '2000-01-01']
df_data_test = df_data_test[df_data_test['backers'] < 100000]

# 不要な列を削除する
df_data_test = df_data_test.drop(['ID','name','category','main_category','currency','deadline','launched'], axis=1)
df_data_test = df_data_test.drop(['state','country','pledged','usd pledged','usd_pledged_real','usd_goal_real'], axis=1)

display(df_data_test.head())
df_data_test.describe()

# 絞った説明変数を使って，ロジスティック回帰
y = df_data_test['Success'].values
X = df_data_test.drop('Success', axis=1).values
clf = SGDClassifier(loss='log', penalty='none', fit_intercept=True, random_state=1234)
clf.fit(X, y)

# 重みの一部（最初のいくつかのみ）を取得して表示
w0 = clf.intercept_[0]
w1 = clf.coef_[0, 0]
w2 = clf.coef_[0, 1]
w3 = clf.coef_[0, 2]
print("w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}".format(w0, w1, w2, w3))
# ラベルを予測
y_est = clf.predict(X)

# 対数尤度を表示
print('対数尤度 = {:.3f}'.format(-log_loss(y, y_est)))

# 正答率を計算
accuracy =  accuracy_score(y, y_est)
print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy))

# Precision, Recall, F1-scoreを計算, 表示
precision, recall, f1_score, _ = precision_recall_fscore_support(y, y_est)
print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))
print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))
print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
# 予測値と正解のクロス集計
conf_mat = pd.DataFrame(confusion_matrix(y, y_est), index=['正解 = 0', '正解 = 1'], columns=['予測値 = 0', '予測値 = 1'])
print(conf_mat)
print('まずはa. goalに対して標準化、backersに対して正規化を行う')

#成功かどうかを判断するため、stateが"Successful"なるTrue、それ以外はFalseとする
df_data_test = df_data.copy()
df_data_test['Success'] = df_data_test['state'] == "successful"

# 欠測値/異常値/外れ値を削除する
df_data_test = df_data_test[df_data_test['country'] != 'N,0"']
df_data_test = df_data_test[df_data_test['launched'] > '2000-01-01']
df_data_test = df_data_test[df_data_test['backers'] < 100000]

# 標準化とは、平均を引いて、標準偏差で割る操作
stdsc = StandardScaler()
stdsc.fit_transform(df_data_test[['goal']].values)

# 正規化とは、全データを0-1の範囲におさめる操作
mms = MinMaxScaler()
mms.fit_transform(df_data_test[['backers']].values)

# カテゴリー変数をダミー変数に変換
df_data_dummy1 = pd.get_dummies(df_data_test['category'])
df_data_dummy2 = pd.get_dummies(df_data_test['country'])
df_data_test = pd.merge(df_data_test, df_data_dummy1, left_index=True, right_index=True)
df_data_test = pd.merge(df_data_test, df_data_dummy2, left_index=True, right_index=True)

# 不要な列を削除する
df_data_test = df_data_test.drop(['ID','name','category','main_category','currency','deadline','launched'], axis=1)
df_data_test = df_data_test.drop(['state','country','pledged','usd pledged','usd_pledged_real','usd_goal_real'], axis=1)

display(df_data_test.head())
df_data_test.describe()

# 絞った説明変数を使って，ロジスティック回帰
y = df_data_test['Success'].values
X = df_data_test.drop('Success', axis=1).values
clf = SGDClassifier(loss='log', penalty='none', fit_intercept=True, random_state=1234)
clf.fit(X, y)

# ラベルを予測
y_est = clf.predict(X)

# 対数尤度を表示
print('対数尤度 = {:.3f}'.format(-log_loss(y, y_est)))

# 正答率を計算
accuracy =  accuracy_score(y, y_est)
print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy))

# Precision, Recall, F1-scoreを計算, 表示
precision, recall, f1_score, _ = precision_recall_fscore_support(y, y_est)
print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))
print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))
print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
print(' ')

# 予測値と正解のクロス集計
conf_mat = pd.DataFrame(confusion_matrix(y, y_est), index=['正解 = 0', '正解 = 1'], columns=['予測値 = 0', '予測値 = 1'])
print(conf_mat)
print(' ')

#結果
print("結果、正規化と標準化による改善は見られなかった。")
print('b. goalとusd_goal_realを白色化し、usd_goal_realを説明変数として加える')

#成功かどうかを判断するため、stateが"Successful"なるTrue、それ以外はFalseとする
df_data_test = df_data.copy()
df_data_test['Success'] = df_data_test['state'] == "successful"

# 欠測値/異常値/外れ値を削除する
df_data_test = df_data_test[df_data_test['country'] != 'N,0"']
df_data_test = df_data_test[df_data_test['launched'] > '2000-01-01']
df_data_test = df_data_test[df_data_test['backers'] < 100000]

# goalとusd_goal_realを白色化
print('goalとusd_goal_realの相関係数: {:.3f}'.format(np.corrcoef(df_data_test['goal'], df_data_test['usd_goal_real'])[0,1]))
#  無相関化を行うための一連の処理
cov = np.cov(df_data_test[['goal','usd_goal_real']], rowvar=0) # 分散・共分散を求める
_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて
df_data_test[['goal','usd_goal_real']] = np.dot(S.T, df_data_test[['goal','usd_goal_real']].T).T #データを無相関化
stdsc = StandardScaler()
stdsc.fit_transform(df_data_test[['usd_goal_real']].values)
print('白色化後のgoalとusd_goal_realの相関係数: {:.3f}'.format(np.corrcoef(df_data_test['goal'], df_data_test['usd_goal_real'])[0,1]))

# 標準化とは、平均を引いて、標準偏差で割る操作
stdsc = StandardScaler()
stdsc.fit_transform(df_data_test[['goal']].values)

# 正規化とは、全データを0-1の範囲におさめる操作
mms = MinMaxScaler()
mms.fit_transform(df_data_test[['backers']].values)

# カテゴリー変数をダミー変数に変換
df_data_dummy1 = pd.get_dummies(df_data_test['category'])
df_data_dummy2 = pd.get_dummies(df_data_test['country'])
df_data_test = pd.merge(df_data_test, df_data_dummy1, left_index=True, right_index=True)
df_data_test = pd.merge(df_data_test, df_data_dummy2, left_index=True, right_index=True)

# 不要な列を削除する
df_data_test = df_data_test.drop(['ID','name','category','main_category','currency','deadline','launched'], axis=1)
df_data_test = df_data_test.drop(['state','country','pledged','usd pledged','usd_pledged_real'], axis=1) 

display(df_data_test.head())
df_data_test.describe()

# 絞った説明変数を使って，ロジスティック回帰
y = df_data_test['Success'].values
X = df_data_test.drop('Success', axis=1).values
clf = SGDClassifier(loss='log', penalty='none', fit_intercept=True, random_state=1234)
clf.fit(X, y)

# ラベルを予測
y_est = clf.predict(X)

# 対数尤度を表示
print('対数尤度 = {:.3f}'.format(-log_loss(y, y_est)))

# 正答率を計算
accuracy =  accuracy_score(y, y_est)
print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy))

# Precision, Recall, F1-scoreを計算, 表示
precision, recall, f1_score, _ = precision_recall_fscore_support(y, y_est)
print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))
print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))
print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
print(' ')

# 予測値と正解のクロス集計
conf_mat = pd.DataFrame(confusion_matrix(y, y_est), index=['正解 = 0', '正解 = 1'], columns=['予測値 = 0', '予測値 = 1'])
print(conf_mat)
print(' ')

#結果
print("白色化の結果、若干精度が向上した！")
print('c. Ridge（L2正則化）を例に正則化を行ってみる')

#成功かどうかを判断するため、stateが"Successful"なるTrue、それ以外はFalseとする
df_data_test = df_data.copy()
df_data_test['Success'] = df_data_test['state'] == "successful"

# 欠測値/異常値/外れ値を削除する
df_data_test = df_data_test[df_data_test['country'] != 'N,0"']
df_data_test = df_data_test[df_data_test['launched'] > '2000-01-01']
df_data_test = df_data_test[df_data_test['backers'] < 100000]

# goalとusd_goal_realを白色化
# print('goalとusd_goal_realの相関係数: {:.3f}'.format(np.corrcoef(df_data_test['goal'], df_data_test['usd_goal_real'])[0,1]))
#  無相関化を行うための一連の処理
cov = np.cov(df_data_test[['goal','usd_goal_real']], rowvar=0) # 分散・共分散を求める
_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて
df_data_test[['goal','usd_goal_real']] = np.dot(S.T, df_data_test[['goal','usd_goal_real']].T).T #データを無相関化
stdsc = StandardScaler()
stdsc.fit_transform(df_data_test[['usd_goal_real']].values)
# print('白色化後のgoalとusd_goal_realの相関係数: {:.3f}'.format(np.corrcoef(df_data_test['goal'], df_data_test['usd_goal_real'])[0,1]))

# 標準化とは、平均を引いて、標準偏差で割る操作
stdsc = StandardScaler()
stdsc.fit_transform(df_data_test[['goal']].values)

# 正規化とは、全データを0-1の範囲におさめる操作
mms = MinMaxScaler()
mms.fit_transform(df_data_test[['backers']].values)

# カテゴリー変数をダミー変数に変換
df_data_dummy1 = pd.get_dummies(df_data_test['category'])
df_data_dummy2 = pd.get_dummies(df_data_test['country'])
df_data_test = pd.merge(df_data_test, df_data_dummy1, left_index=True, right_index=True)
df_data_test = pd.merge(df_data_test, df_data_dummy2, left_index=True, right_index=True)

# 不要な列を削除する
df_data_test = df_data_test.drop(['ID','name','category','main_category','currency','deadline','launched'], axis=1)
df_data_test = df_data_test.drop(['state','country','pledged','usd pledged','usd_pledged_real'], axis=1) 

display(df_data_test.head())
df_data_test.describe()

# 絞った説明変数を使って，ロジスティック回帰
y = df_data_test['Success'].values
X = df_data_test.drop('Success', axis=1).values

# Test用データを分離する
test_size = 0.2        # 全データのうち、何%をテストデータにするか（今回は20%に設定）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) # ホールドアウト法を実行（テストデータはランダム選択）

# パラメータ検証用のクロスバリデーションを実施
alphas = [0.0, 0.2, 0.4, 0.6]#alpha(数式ではλ)の値を4つ指定する
n_split = 5 # グループ数を設定（今回は5分割）
cross_valid_losses = []

for alpha in alphas:
    cross_valid_loss = 0
    split_num = 1

    # テスト役を交代させながら学習と評価を繰り返す
    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X_train, y_train):
        X_train_p, y_train_p = X_train[train_idx], y_train[train_idx] #学習用データ
        X_test_p, y_test_p = X_train[test_idx], y_train[test_idx]     #テスト用データ

        clf = SGDClassifier(loss='log', penalty='none', fit_intercept=True, random_state=1234)
        clf.fit(X_train_p, y_train_p)

        # リッジのモデルを生成
        ridge = Ridge(alpha)
        ridge.fit(X_train_p, y_train_p)

        # ラベルを予測
        y_est_test_p = clf.predict(X_test_p)

        # 対数尤度を表示
        loss_ = -log_loss(y_test_p, y_est_test_p)
        print("Alpha %s"%alpha)
        #print("L1_ratio %s"%l1_ratio)
        print("Fold %s"%split_num)
        print('対数尤度 = {:.3f}'.format(loss_))

        cross_valid_loss += loss_ #後で平均を取るためにloss_を加算
        split_num += 1

    # cross_valid_lossの結果を保管する
    cross_valid_losses.append(cross_valid_loss/split_num)

# パラメータごとの結果グラフ
print(' ')
print('下記のグラフはalpha値毎の対数尤度')
print(' ')
plt.plot(alphas, cross_valid_losses)

# テストデータを用いて、ラベルを予測
y_est_test = clf.predict(X_test)

# 対数尤度を表示
loss_ = -log_loss(y_test, y_est_test)
print('対数尤度 = {:.3f}'.format(loss_))

# 正答率を計算
accuracy =  accuracy_score(y_test, y_est_test)
print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy))

# Precision, Recall, F1-scoreを計算, 表示
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_est_test)
print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))
print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))
print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
print(' ')

# 予測値と正解のクロス集計
conf_mat = pd.DataFrame(confusion_matrix(y_test, y_est_test), index=['正解 = 0', '正解 = 1'], columns=['予測値 = 0', '予測値 = 1'])
print(conf_mat)
print(' ')

#結果
print('結果、４つのパラメータ試したが正則化による効果・変化は見られなかった。') 
print('何かやり方がまずいのかもしれない') 
print('他の課題と前後したが、交差検証による汎化性能を確認する')

#成功かどうかを判断するため、stateが"Successful"なるTrue、それ以外はFalseとする
df_data_test = df_data.copy()
df_data_test['Success'] = df_data_test['state'] == "successful"

# 欠測値/異常値/外れ値を削除する
df_data_test = df_data_test[df_data_test['country'] != 'N,0"']
df_data_test = df_data_test[df_data_test['launched'] > '2000-01-01']
df_data_test = df_data_test[df_data_test['backers'] < 100000]

# goalとusd_goal_realを白色化
# print('goalとusd_goal_realの相関係数: {:.3f}'.format(np.corrcoef(df_data_test['goal'], df_data_test['usd_goal_real'])[0,1]))
#  無相関化を行うための一連の処理
cov = np.cov(df_data_test[['goal','usd_goal_real']], rowvar=0) # 分散・共分散を求める
_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて
df_data_test[['goal','usd_goal_real']] = np.dot(S.T, df_data_test[['goal','usd_goal_real']].T).T #データを無相関化
stdsc = StandardScaler()
stdsc.fit_transform(df_data_test[['usd_goal_real']].values)
# print('白色化後のgoalとusd_goal_realの相関係数: {:.3f}'.format(np.corrcoef(df_data_test['goal'], df_data_test['usd_goal_real'])[0,1]))

# 標準化とは、平均を引いて、標準偏差で割る操作
stdsc = StandardScaler()
stdsc.fit_transform(df_data_test[['goal']].values)

# 正規化とは、全データを0-1の範囲におさめる操作
mms = MinMaxScaler()
mms.fit_transform(df_data_test[['backers']].values)

# カテゴリー変数をダミー変数に変換
df_data_dummy1 = pd.get_dummies(df_data_test['category'])
df_data_dummy2 = pd.get_dummies(df_data_test['country'])
df_data_test = pd.merge(df_data_test, df_data_dummy1, left_index=True, right_index=True)
df_data_test = pd.merge(df_data_test, df_data_dummy2, left_index=True, right_index=True)

# 不要な列を削除する
df_data_test = df_data_test.drop(['ID','name','category','main_category','currency','deadline','launched'], axis=1)
df_data_test = df_data_test.drop(['state','country','pledged','usd pledged','usd_pledged_real'], axis=1) 

display(df_data_test.head())
df_data_test.describe()

# 絞った説明変数を使って，ロジスティック回帰
y = df_data_test['Success'].values
X = df_data_test.drop('Success', axis=1).values

n_split = 5 # グループ数を設定（今回は5分割）
alpha = 0.4
cross_valid_loss = 0
split_num = 1

# テスト役を交代させながら学習と評価を繰り返す
for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
    X_train, y_train = X[train_idx], y[train_idx] #学習用データ
    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    clf = SGDClassifier(loss='log', penalty='none', fit_intercept=True, random_state=1234)
    clf.fit(X_train, y_train)

    # リッジのモデルを生成
    ridge = Ridge(alpha)
    ridge.fit(X_train, y_train)

    # テストデータでラベルを予測
    y_est_test = clf.predict(X_test)

    # 対数尤度を表示
    loss_ = -log_loss(y_test, y_est_test)
    print("Fold %s"%split_num)
    print('各回の対数尤度 = {:.3f}'.format(loss_))

    cross_valid_loss += loss_ #後で平均を取るためにloss_を加算
    split_num += 1

# 対数尤度を表示
final_loss_ = cross_valid_loss/(split_num-1)
print(' ')
print('最終的な平均の対数尤度 = {:.3f}'.format(final_loss_))
print(' ')

#結果
print('今までのテストと比較して、対数尤度に大きな違いはなく、汎化性能があるのではないか') 
#成功かどうかを判断するため、stateが"Successful"なるTrue、それ以外はFalseとする
df_data_test = df_data.copy()
df_data_test['Success'] = df_data_test['state'] == "successful"

# 欠測値/異常値/外れ値を削除する
df_data_test = df_data_test[df_data_test['country'] != 'N,0"']
df_data_test = df_data_test[df_data_test['launched'] > '2000-01-01']
df_data_test = df_data_test[df_data_test['backers'] < 100000]

# goalとusd_goal_realを白色化
# print('goalとusd_goal_realの相関係数: {:.3f}'.format(np.corrcoef(df_data_test['goal'], df_data_test['usd_goal_real'])[0,1]))
#  無相関化を行うための一連の処理
cov = np.cov(df_data_test[['goal','usd_goal_real']], rowvar=0) # 分散・共分散を求める
_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて
df_data_test[['goal','usd_goal_real']] = np.dot(S.T, df_data_test[['goal','usd_goal_real']].T).T #データを無相関化
stdsc = StandardScaler()
stdsc.fit_transform(df_data_test[['usd_goal_real']].values)
# print('白色化後のgoalとusd_goal_realの相関係数: {:.3f}'.format(np.corrcoef(df_data_test['goal'], df_data_test['usd_goal_real'])[0,1]))

# 標準化とは、平均を引いて、標準偏差で割る操作
stdsc = StandardScaler()
stdsc.fit_transform(df_data_test[['goal']].values)

# 正規化とは、全データを0-1の範囲におさめる操作
mms = MinMaxScaler()
mms.fit_transform(df_data_test[['backers']].values)

# カテゴリー変数をダミー変数に変換
# df_data_dummy1 = pd.get_dummies(df_data_test['category'])
# df_data_test = pd.merge(df_data_test, df_data_dummy1, left_index=True, right_index=True)
#df_data_dummy2 = pd.get_dummies(df_data_test['country'])
#df_data_test = pd.merge(df_data_test, df_data_dummy2, left_index=True, right_index=True)

# 不要な列を削除する
df_data_test = df_data_test.drop(['ID','name','category','main_category','currency','deadline','launched'], axis=1)
df_data_test = df_data_test.drop(['state','country','pledged','usd pledged','usd_pledged_real'], axis=1) 

display(df_data_test.head())
df_data_test.describe()

# 絞った説明変数を使って，SVMを実施
y = df_data_test['Success'].values
X = df_data_test.drop('Success', axis=1).values

# Test用データを分離する
test_size = 0.2        # 全データのうち、何%をテストデータにするか（今回は20%に設定）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) # ホールドアウト法を実行（テストデータはランダム選択）

# SVMの実行
C = 5
clf = SVC(C=C, kernel="linear")
# clf.fit(X_train, y_train)  ### ★☆★処理が完了しないので、いったんコメントアウト★☆★

#結果
print('SVCをうまくうまく動かせない') 