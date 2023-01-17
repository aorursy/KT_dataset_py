## 参考：DAY２の宿題

### 目次
#### 8. DAY2で学んだことの取り組み
##### 8-1. 交差検証、ホールドアウト法などで汎化性能を確認する
##### 8-2. 欠測値と異常値を確認し、適切に処理する
##### 8-3. DAY2、3で学んだアルゴリズムを利用してモデルをつくり、DAY1宿題提出時の精度と比較する
##### 8-4. 交差検証によるパラメータチューニングを行う
##### 8-5. パラメータチューニング後のモデルによって、精度および結果の評価を行う
##### 8-6. その他、精度の向上ができるような処理に取り組み、精度を上げる
##### 8-7. できたところまでをNotebookでまとめ、宿題として提出する
##### 8-8. 前回から取り組んだ内容・工夫、精度がどのように変化したかのコメントをNotebookに含めること

## 参考：DAY1の宿題
### 目次
#### 1. 自分が取り組む通し課題を1つ選択する
##### • Kaggleアカウントを作成し、該当課題のデータをダウンロードする
#### 2. 目的変数と説明変数の関係を確認するためのグラフを作成する
#### 3. 目的変数を説明するのに有効そうな説明変数を見つける
#### 4. DAY1で学んだアルゴリズムを利用する
##### • 回帰の場合は線形回帰、分類の場合はロジスティック回帰
##### • 質的変数が扱えないアルゴリズムを使う場合は、ダミー変数に置き換える
#### 5. 予測精度または識別精度を確認する
##### • 回帰問題の場合は、MSE、RMSE、MAEを求める
##### • 分類問題の場合は、混同行列を作成し、Accuracy、Recall、Precisionを求める
#### 6. できたところまでをNotebookでまとめ、KernelsまたはGithubで公開する
##### • 公開方法がわからない方は、ipynbファイルを#generalに貼る事前準備のお願い
print('取り組む課題を１つ選択する')
print("I choose 'Kickstarter Projects'")

print('必要なLibraryをImportする')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
%matplotlib inline
from sklearn.linear_model import SGDClassifier, LassoCV
from sklearn.linear_model import Ridge,RidgeClassifier, Lasso,ElasticNet #正則化項付き最小二乗法を行うためのライブラリ
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split # ホールドアウト法に関する関数
from sklearn.model_selection import KFold # 交差検証法に関する関数
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals.six import StringIO
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV, SelectFromModel
from IPython.core.display import display 
from IPython.display import Image
from datetime import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD,RMSprop, Adagrad, Adadelta, Adam
import warnings
warnings.filterwarnings('ignore')

print('データをダウンロードし、ダウンロードした結果を確認する')
#df_data = pd.read_csv("../input/ks-projects-201801.csv")
df_data = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")
print("Row lengths of imported data: ", len(df_data))
print('まずはHeaderを確認')
display(df_data.head())
df_data.describe()
print("項目ごとの説明(KaggleおよびSlackより取得した情報)を表示する")
df_data_exp = pd.read_csv("../input/data-explanation/data_explanation.csv",encoding='cp932')
display(df_data_exp)
print('currencyとcountryの関係')
df_currency_country = df_data.groupby('country')
df_currency_country = df_currency_country['currency'].value_counts(normalize=True).unstack(fill_value=0)
display(df_currency_country)

print('countryの種類と数を調べる')
print(df_data['country'].value_counts(dropna=False))

print('\n考察')
print('countryから異常値と思われる『N,0"』を除けば、countryによりcurrencyが一意に決まる')
print('従って、説明変数からcurrencyを除外することが出来る')
print('categoryとmain categoryの関係')
df_categories = df_data.groupby('category')
df_categories = df_categories['main_category'].value_counts(normalize=True).unstack(fill_value=0)
display(df_categories)

#categoryの種類と数を調べる
print('categoryの種類と数を調べる')
print(df_data['category'].value_counts(dropna=False))

#main_categoryの種類と数を調べる
print('main_categoryの種類と数を調べる')
print(df_data['main_category'].value_counts(dropna=False))

print('\n考察')
print('AnthologiesやSpacesのような一部の例外を除けば、categoryによりmain_categoryが一意に決まる')
print('従って、説明変数からmain_categoryを除外しても、分析への影響は希少と思料する')
print('goalとusd_goal_realの関係')
print('まずは散布図行列を書いてみる')
pd.plotting.scatter_matrix(df_data[['goal','usd_goal_real']], figsize=(15,15))
plt.show()
print(' ')
print('次に相関係数を確認')
corr_ = df_data[['goal','usd_goal_real']].corr()
print(corr_)
print('\n考察')
print('goalとusd_goal_realは視覚的に相関が確認出来て、相関係数も0.94と極めて高い')
print('従って、説明変数としてusd_goal_realを使わず、goalで代用する')
print('Stateの種類と数を調べる')
print(df_data['state'].value_counts(dropna=False))
print('Category毎にStateとの相関をグラフ化する\n')
category_ = df_data.groupby('category')
category_ = category_['state'].value_counts(normalize=True).unstack()
category_ = category_.sort_values(by=['successful'],ascending=True)
category_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
print("成功しやすいCategoryと成功しにくいCateogryが存在する")
print("Maxは80%近く、Minは10%以下")
print('deadline毎にStateとの相関をグラフ化する')
print('deadlineの数が多いので、年月別にする')
df_data_deadline = df_data.copy()
df_data_deadline['deadline_YM'] = df_data_deadline['deadline'].apply(lambda x: x[0:7])
deadline_ = df_data_deadline.groupby('deadline_YM')
deadline_ = deadline_['state'].value_counts(normalize=True).unstack()
ax = deadline_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
plt.legend(loc='upper left')
print("\n2018年にLiveが多い。が、他の明確な傾向はつかみにくい")
print('goal毎にStateとの相関をグラフ化する')
print('goalの数が多すぎるので、10万単位で丸めて相関を見る')
df_data_goal = df_data.copy()
df_data_goal['goal_r'] = df_data_goal['goal'].apply(lambda x: round(x/100000))
goal_ = df_data_goal.groupby('goal_r')
goal_ = goal_['state'].value_counts(normalize=True).unstack()
#goal_ = goal_.sort_values('goal_r',ascending=False)
ax = goal_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
plt.legend(loc='upper left')
print("\nGoalが大き過ぎると、成功しにくいようだ")
print('launched毎にStateとの相関をグラフ化する')
print('launchedの数が多いので、年月別にする')
df_data_launched = df_data.copy()
df_data_launched['launched_YM'] = df_data_launched['launched'].apply(lambda x: x[0:7])
launched_ = df_data_launched.groupby('launched_YM')
launched_ = launched_['state'].value_counts(normalize=True).unstack()
#launched_ = launched_.sort_values('launched_YM',ascending=False)
ax = launched_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
plt.legend(loc='upper left')
print("\n1970年に開始したものは失敗、そもそも、これは他から離れた異常値と扱うべきか")
print("2017年12月以降開始はLive、それ以外の傾向は見にくい")
print('backers毎にStateとの相関をグラフ化する')
print('backersの数が多いので、1000単位で丸めて相関を見る')
df_data_backers = df_data.copy()
df_data_backers['backers_r'] = df_data_backers['backers'].apply(lambda x: round(x/1000))
backers_ = df_data_backers.groupby('backers_r')
backers_ = backers_['state'].value_counts(normalize=True).unstack()
#backers_ = backers_.sort_values('backers_r',ascending=False)
ax = backers_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,20))
plt.legend(loc='upper left')
print("\nbackersが一定以上になると、ほぼ成功している")
print("100000以上は連続性に乏しいので外れ値として扱う")
print('country毎にStateとの相関をグラフ化する')
country_ = df_data.groupby('country')
country_ = country_['state'].value_counts(normalize=True).unstack()
country_ = country_.sort_values(by=['successful'],ascending=True)
country_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,7))
print("\n成功しやすい国と成功しにくい国が存在するが、")
print("Maxは40%近く、Minは20%以下で幅はそれほど大きくない")
print('欠測値を確認する')
print(df_data.isnull().any(axis=0))
print('\n考察')
print("欠測値はnameとusd pledgedにある")
print("異常値/外れ値は、上記までの分析でcountryとlaunchedとbackersに観察されている")
print("これらの欠測値/異常値/外れ値は、説明変数からは除外する")
def make_test_data(show_comment=False, \
                   show_data=False, \
                   stdsc=False, \
                   mms=False, \
                   decorre=False, \
                   show_decorre_detail = False, \
                   whitening=False, \
                   launched_del=True, \
                   deadline_del=True, \
                   usd_goal_real_del=True, \
                   category_keep=True, \
                   country_keep=True \
                  ):
    if(show_comment):
        print('成功かどうかを判断するため、stateが"Successful"なるTrue、それ以外はFalseとする')
    df_data_test = df_data.copy()
    df_data_test['Success'] = df_data_test['state'] == "successful"

    # カテゴリー変数をダミー変数に変換
    if(category_keep):
        df_data_dummy1 = pd.get_dummies(df_data_test['category'])
        df_data_test = pd.merge(df_data_test, df_data_dummy1, left_index=True, right_index=True)
    if(country_keep):
        df_data_dummy2 = pd.get_dummies(df_data_test['country'])
        df_data_test = pd.merge(df_data_test, df_data_dummy2, left_index=True, right_index=True)

    # 欠測値/異常値/外れ値を削除する
    df_data_test = df_data_test[df_data_test['country'] != 'N,0"']
    df_data_test = df_data_test[df_data_test['launched'] > '2000-01-01']
    df_data_test = df_data_test[df_data_test['backers'] < 100000]

    #日付関連の変換処理
    df_data_test['launched'] = pd.to_datetime(df_data_test['launched'].apply(lambda x: x[0:10])).map(pd.Timestamp.timestamp)
    df_data_test['deadline'] = pd.to_datetime(df_data_test['deadline']).map(pd.Timestamp.timestamp)

    # 標準化とは、平均を引いて、標準偏差で割る操作
    if(stdsc):
        stdsc_ = StandardScaler()
        df_data_test[['goal']] = stdsc_.fit_transform(df_data_test[['goal']].values)
        df_data_test[['usd_goal_real']] = stdsc_.fit_transform(df_data_test[['usd_goal_real']].values)
        df_data_test[['backers']] = stdsc_.fit_transform(df_data_test[['backers']].values)
        df_data_test[['launched']] = stdsc_.fit_transform(df_data_test[['launched']].values)
        df_data_test[['deadline']] = stdsc_.fit_transform(df_data_test[['deadline']].values)

    # 正規化とは、全データを0-1の範囲におさめる操作
    if(mms):
        mms_ = MinMaxScaler()
        df_data_test[['goal']] = mms_.fit_transform(df_data_test[['goal']].values)
        df_data_test[['usd_goal_real']] = mms_.fit_transform(df_data_test[['usd_goal_real']].values)
        df_data_test[['backers']] = mms_.fit_transform(df_data_test[['backers']].values)
        df_data_test[['launched']] = mms_.fit_transform(df_data_test[['launched']].values)
        df_data_test[['deadline']] = mms_.fit_transform(df_data_test[['deadline']].values)

    if(decorre):
        if(show_decorre_detail):
            # goalとusd_goal_realを白色化
            print('goalとusd_goal_realの相関係数: {:.3f}'.format(np.corrcoef(df_data_test['goal'], df_data_test['usd_goal_real'])[0,1]))
        #  無相関化を行うための一連の処理
        cov = np.cov(df_data_test[['goal','usd_goal_real']], rowvar=0) # 分散・共分散を求める
        _, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて
        df_data_test[['goal','usd_goal_real']] = np.dot(S.T, df_data_test[['goal','usd_goal_real']].T).T #データを無相関化
        if(show_decorre_detail):
            print('白色化後のgoalとusd_goal_realの相関係数: {:.3f}'.format(np.corrcoef(df_data_test['goal'], df_data_test['usd_goal_real'])[0,1]))
    if(whitening):
        stdsc_ = StandardScaler()
        df_data_test[['usd_goal_real']] = stdsc_.fit_transform(df_data_test[['usd_goal_real']].values)
        df_data_test[['goal']] = stdsc_.fit_transform(df_data_test[['goal']].values)

    # 不要な列を削除する
    df_data_test = df_data_test.drop(['ID','name','category','main_category','currency'], axis=1)
    df_data_test = df_data_test.drop(['state','country','pledged','usd pledged','usd_pledged_real'], axis=1)
    if(launched_del):
        df_data_test = df_data_test.drop(['launched'], axis=1)
    if(deadline_del):
        df_data_test = df_data_test.drop(['deadline'], axis=1)
    if(usd_goal_real_del):
        df_data_test = df_data_test.drop(['usd_goal_real'], axis=1)

    if(show_data):
         display(df_data_test.head())
         df_data_test.describe()

    return df_data_test

def logistic_(df_data_test):
    y = df_data_test['Success'].values
    X = df_data_test.drop('Success', axis=1).values
    clf = SGDClassifier(loss='log', penalty='none', fit_intercept=True, random_state=1234)
    clf.fit(X, y)
    y_est = clf.predict(X) # ラベルを予測
    return y, y_est

def cross_valid(df_data_test, \
                show_lap=False, \
                SGDClass_use = True, \
                lasso_use=False, \
                lasso_alpha=0, \
                ridge_use=False, \
                ridge_alpha=0 \
               ):
    # 絞った説明変数を使って，ロジスティック回帰
    y = df_data_test['Success'].values
    X = df_data_test.drop('Success', axis=1).values

    n_split = 5 # グループ数を設定（今回は5分割）
    cross_valid_loss = 0
    split_num = 1

    # テスト役を交代させながら学習と評価を繰り返す
    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
        X_train, y_train = X[train_idx], y[train_idx] #学習用データ
        X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

        if(SGDClass_use):
            clf = SGDClassifier(loss='log', penalty='none', fit_intercept=True, random_state=1234)
        if(lasso_use):
            clf = Lasso(alpha=lasso_alpha)
        if(ridge_use):
            #clf = Ridge(alpha=ridge_alpha)
            clf = RidgeClassifier(alpha=ridge_alpha)
        clf.fit(X_train, y_train)

        # テストデータでラベルを予測
        y_est_test = clf.predict(X_test)

        if(show_lap):
            # 対数尤度を表示
            loss_ = -log_loss(y_test, y_est_test)
            print("Fold %s"%split_num)
            print('各回の対数尤度 = {:.3f}'.format(loss_))
            cross_valid_loss += loss_ #後で平均を取るためにloss_を加算

        if(split_num==1):
            y_ = y_test
            y_est = y_est_test
        else:
            y_ = np.append(y_, y_test)
            y_est = np.append(y_est, y_est_test)
            #y_ = y_.append(y_test)
            #y_est = y_est.append(y_est_test)
            
        split_num += 1

    return y_, y_est
        

def show_results(y, y_est):
    # 対数尤度を表示
    print('対数尤度 = {:.3f}'.format(-log_loss(y, y_est)))

    # 正答率を計算
    accuracy =  accuracy_score(y, y_est)
    # Precision, Recall, F1-scoreを計算, 表示
    precision, recall, f1_score, _ = precision_recall_fscore_support(y, y_est)
    print('正答率（Accuracy） = {:.3f}%, '.format(100 * accuracy), \
          '適合率（Precision） = {:.3f}%, '.format(100 * precision[0]) \
         )
    print('再現率（Recall） = {:.3f}%, '.format(100 * recall[0]), \
          'F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0])\
         )

    # 予測値と正解のクロス集計
    print('予測値と正解のクロス集計')
    conf_mat = pd.DataFrame(confusion_matrix(y, y_est), index=['正解 = 0', '正解 = 1'], columns=['予測値 = 0', '予測値 = 1'])
    print(conf_mat)
    
# データ作成関数を呼ぶ
df_data_test_4 = make_test_data(show_comment=True, show_data=True)

# 絞った説明変数を使って，ロジスティック回帰
# 基本的にクロスバリデーションで評価する
y, y_est = cross_valid(df_data_test_4)

# 結果を表示
show_results(y, y_est)
print('a. goalとbackersに対して標準化と正規化を行う。')
print('まずは標準化から')
# データ作成関数を呼ぶ
df_data_test_8_3a1 = make_test_data(show_data=True, stdsc=True, mms=False)
# 絞った説明変数を使って，ロジスティック回帰
y, y_est = cross_valid(df_data_test_8_3a1)
# 結果を表示
show_results(y, y_est)

print('\n次に正規化')
# データ作成関数を呼ぶ
df_data_test_8_3a2 = make_test_data(show_data=True, stdsc=False, mms=True)
# 絞った説明変数を使って，ロジスティック回帰
y, y_est = logistic_(df_data_test_8_3a2)
# 結果を表示
show_results(y, y_est)

print('\n結果、正規化と標準化のどちらも精度が悪化した。')
print('以降、正規化・標準化無しをデフォルトとする。')
print('b. goalとusd_goal_realを無相関化あるいは白色化し、usd_goal_realを説明変数として加える')
print('まずは無相関化から')

# データ作成関数を呼ぶ
df_data_test_8_3b1 = make_test_data(show_data=True, decorre=True, show_decorre_detail = True, whitening=False, usd_goal_real_del=False)
# 絞った説明変数を使って，ロジスティック回帰
y, y_est = cross_valid(df_data_test_8_3b1)
# 結果を表示
show_results(y, y_est)

print('\n次に白色化')
# データ作成関数を呼ぶ
df_data_test_8_3b2 = make_test_data(show_data=True, decorre=True, show_decorre_detail = True, whitening=True, usd_goal_real_del=False)
# 絞った説明変数を使って，ロジスティック回帰
y, y_est = cross_valid(df_data_test_8_3b2)
# 結果を表示
show_results(y, y_est)

#結果
print("\n無相関化の結果、若干精度が向上したが、白色化の結果、精度は悪化した。")
print("以降、無相関化をデフォルトとする")
print('c. Ridge（L2正則化）を例に正則化を行ってみる')

# データ作成関数を呼ぶ
df_data_test_8_3c1 = make_test_data(decorre=True, whitening=False, usd_goal_real_del=False)

# Test用データを分離する
test_size = 0.2        # 全データのうち、何%をテストデータにするか（今回は20%に設定）
dataset_train, dataset_test, dummy_train, dummy_test = \
train_test_split(df_data_test_8_3c1, df_data_test_8_3c1, test_size=test_size, random_state=1234) # ホールドアウト法を実行（テストデータはランダム選択）

# パラメータ検証用のクロスバリデーションを実施
alphas = [0, 10, 103, 1000, 10000] #alpha(数式ではλ)の値を4つ指定する
n_split = 5 # グループ数を設定（今回は5分割）
cross_valid_losses = []
print("Alphas = ", alphas)

for alpha in alphas:
    cross_valid_loss = 0
    split_num = 1

    # テスト役を交代させながら学習と評価を繰り返す
    y, y_est = cross_valid(dataset_train, SGDClass_use=False, ridge_use=True, ridge_alpha=alpha)
    # 結果を表示
    print("\nAlpha = %s"%alpha)
    show_results(y, y_est)

# Best Scoreを表示
alpha=103
print("\nBest Alpha = %s"%alpha)
y, y_est = cross_valid(dataset_test, SGDClass_use=False, ridge_use=True, ridge_alpha=alpha)
show_results(y, y_est)
    
#結果
print('幅広くパラメータを試した結果、alpha=103の時に対数尤度が最も小さくなった。') 
print('結果として、今までの結果よりも精度は下がっている') 
# データ作成関数を呼ぶ
df_data_test_9_4 = make_test_data(decorre=True, show_data=True, usd_goal_real_del=False, category_keep=False, country_keep=False)
#make_test_data(decorre=True, usd_goal_real_del=False)

# 絞った説明変数を使って，SVMを実施
y = df_data_test_9_4['Success'].values
X = df_data_test_9_4.drop('Success', axis=1).values

# Test用データを分離する
test_size = 0.98        # 全データのうち、何%をテストデータにするか
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) # ホールドアウト法を実行（テストデータはランダム選択）

# SVMの実行
C = 5
clf = SVC(C=C, kernel="rbf")
clf.fit(X_train, y_train)  

y_test_est = clf.predict(X_test)
show_results(y_test, y_test_est)

#結果
print('\nデータ項目・データ数を絞ったところ、ようやくサポートベクトルマシンが動いた') 
print('\nデータが少ないためか、そもそもの性質か、精度は低い') 
print('10-1-1. ラッパー法を用いる')
# データ作成関数を呼ぶ
df_data_test_10_1_1 = make_test_data(decorre=True, \
                                     show_data=True, \
                                     mms=True, \
                                     launched_del=False, \
                                     deadline_del=False, \
                                     usd_goal_real_del=False, \
                                     category_keep=False, \
                                     country_keep=False)

print('ラッパー法使用前')
y = df_data_test_10_1_1['Success'].values
X = df_data_test_10_1_1.drop('Success', axis=1).values
y, y_est = cross_valid(df_data_test_10_1_1)
show_results(y, y_est)

# estimatorにモデルをセット
estimator = SGDClassifier()
# RFECVは交差検証によってステップワイズ法による特徴選択を行う
# cvにはFold（=グループ）の数，scoringには評価指標を指定する
# 今回は回帰なのでneg_mean_absolute_errorを評価指標に指定（分類ならaccuracy）
rfecv = RFECV(estimator, cv=5, scoring='accuracy')

# fitで特徴選択を実行
rfecv.fit(X, y)
# 特徴のランキングを表示（1が最も重要な特徴）
print('\nFeature ranking: \n{}'.format(rfecv.ranking_))

# bool型の配列に ~ をつけるとTrueとFalseを反転させることができる
# ここでTrueになっている特徴が削除してもよい特徴
remove_idx = ~rfecv.support_
print(remove_idx)

# 削除してもよい特徴の名前を取得する
remove_feature = df_data_test_10_1_1.drop('Success', axis=1).columns[remove_idx]
print('削除してもよい特徴量は、', remove_feature)

# drop関数で特徴を削除
df_data_test_10_1_1b = df_data_test_10_1_1.drop(remove_feature, axis=1)
display(df_data_test_10_1_1b.head())

print('ラッパー法使用後')
y, y_est = cross_valid(df_data_test_10_1_1b)
show_results(y, y_est)

print('\nラッパー法により、項目を削除しても、精度にあまり違いの出ないことが確認できた。')
print('10-1-2. 埋め込み法を用いる')
# データ作成関数を呼ぶ
df_data_test_10_1_2 = make_test_data(decorre=True, \
                                     show_data=True, \
                                     mms=True, \
                                     launched_del=False, \
                                     deadline_del=False, \
                                     usd_goal_real_del=False, \
                                     category_keep=False, \
                                     country_keep=False)

print('埋め込み法使用前')
y = df_data_test_10_1_2['Success'].values
X = df_data_test_10_1_2.drop('Success', axis=1).values
y, y_est = cross_valid(df_data_test_10_1_2)
show_results(y, y_est)

# estimatorにモデルをセット
# LassoCVを使って、正則化の強さは自動決定
estimator = LassoCV(normalize=True, cv=10)

# モデルの情報を使って特徴選択を行う場合は、SelectFromModelを使う
# 今回は係数が0.1以下である特徴を削除する
# 係数のしきい値はthresholdで指定する
sfm = SelectFromModel(estimator, threshold=0.1)

# fitで特徴選択を実行
sfm.fit(X, y)

# get_support関数で使用する特徴のインデックスを使用
# Trueになっている特徴が使用する特徴
sfm.get_support()

# bool型の配列に ~ をつけるとTrueとFalseを反転させることができる
# ここでTrueになっている特徴が削除してもよい特徴
remove_idx = ~sfm.get_support()
print(remove_idx)

# 削除してもよい特徴の名前を取得する
remove_feature = df_data_test_10_1_1.drop('Success', axis=1).columns[remove_idx]
print('削除してもよい特徴量は、', remove_feature)

# LASSOで得た各特徴の係数の値を確認してみよう
# 係数の絶対値を取得
abs_coef = np.abs(sfm.estimator_.coef_)
print(abs_coef)

# 係数を棒グラフで表示
plt.barh(np.arange(0, len(abs_coef)), abs_coef, tick_label=df_data_test_10_1_1.drop('Success', axis=1).columns.values)
plt.show()

# drop関数で特徴を削除
df_data_test_10_1_2b = df_data_test_10_1_2.drop(remove_feature, axis=1)
display(df_data_test_10_1_2b.head())

print('埋め込み法使用後')
y, y_est = cross_valid(df_data_test_10_1_2b)
show_results(y, y_est)

print('\n埋め込み法により、項目を削除しても、精度に違いの出ないことが確認できた。')
print('10-2-1. 決定木を用いる')
# データ作成関数を呼ぶ
df_data_test_10_2_1 = make_test_data(decorre=True, \
                                     show_data=True, \
                                     usd_goal_real_del=False, \
                                     category_keep=False, \
                                     country_keep=False)

# Test用データを分離する
y = df_data_test_10_2_1['Success'].values
X = df_data_test_10_2_1.drop('Success', axis=1).values
test_size = 0.9995        # 全データのうち、何%をテストデータにするか
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) # ホールドアウト法を実行（テストデータはランダム選択）

clf = DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=3, min_samples_leaf=3, random_state=1234)
clf = clf.fit(X_train, y_train)
print("score=", clf.score(X_train, y_train))
print(clf.predict(X_test)) #予測したい場合

# 説明変数の重要度を出力する
# scikit-learnで算出される重要度は、ある説明変数による不純度の減少量合計である。
print(clf.feature_importances_)
pd.DataFrame(clf.feature_importances_, index=["goal","backers","usd_goal_real"]).plot.bar(figsize=(7,2))
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()

# 決定木の描画
dot_data = StringIO() #dotファイル情報の格納先
export_graphviz(clf, out_file='tree_limited.dot',  
                     feature_names=["goal","backers","usd_goal_real"],  
                     class_names=["0","1"],  
                     filled=True, rounded=True,  
                     special_characters=True) 
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
#Image(graph.create_png())
!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600
Image(filename = 'tree_limited.png')

# 決定木が上手く描けたので、
#改めて、80%のデータを用いて学習
# データ作成関数を呼ぶ
df_data_test_10_2_1 = make_test_data(decorre=True, \
                                     show_data=False, \
                                     usd_goal_real_del=False, \
                                     category_keep=False, \
                                     country_keep=False)

# Test用データを分離する
y = df_data_test_10_2_1['Success'].values
X = df_data_test_10_2_1.drop('Success', axis=1).values

test_size = 0.2        # 全データのうち、何%をテストデータにするか
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) # ホールドアウト法を実行（テストデータはランダム選択）
clf = DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=3, min_samples_leaf=3, random_state=1234)
clf = clf.fit(X_train, y_train)

y_est = clf.predict(X_test) # ラベルを予測
show_results(y_test, y_est)

print('\n決定木により、Accuracy91%超の精度を得ることができた')
print('10-2-2. ランダムフォレストを用いる')
# データ作成関数を呼ぶ
df_data_test_10_2_1 = make_test_data(decorre=True, \
                                     show_data=True, \
                                     usd_goal_real_del=False, \
                                     category_keep=False, \
                                     country_keep=False)

# Test用データを分離する
y = df_data_test_10_2_1['Success'].values
X = df_data_test_10_2_1.drop('Success', axis=1).values
test_size = 0.2        # 全データのうち、何%をテストデータにするか
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) # ホールドアウト法を実行（テストデータはランダム選択）

# clf = RandomForestClassifier(n_estimators=10, max_depth=2, criterion="gini",
                             # min_samples_leaf=2, min_samples_split=2, random_state=1234)
clf = RandomForestClassifier(n_estimators=100, max_depth=5, criterion="gini",
                             min_samples_leaf=4, min_samples_split=4, random_state=1234)
clf.fit(X_train, y_train)
print("score=", clf.score(X_train, y_train))

# 説明変数の重要度を出力する
# scikit-learnで算出される重要度は、ある説明変数による不純度の減少量合計である。
print(clf.feature_importances_)
pd.DataFrame(clf.feature_importances_, index=["goal","backers","usd_goal_real"]).plot.bar(figsize=(7,2))
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()

y_est = clf.predict(X_test) # ラベルを予測
show_results(y_test, y_est)

print('\nランダムフォレストにより、決定木を超える精度を得ることができた')
print('10-2-3. アダブーストを用いる')
# データ作成関数を呼ぶ
df_data_test_10_2_1 = make_test_data(decorre=True, \
                                     show_data=True, \
                                     usd_goal_real_del=False, \
                                     category_keep=False, \
                                     country_keep=False)

# Test用データを分離する
y = df_data_test_10_2_1['Success'].values
X = df_data_test_10_2_1.drop('Success', axis=1).values
test_size = 0.2        # 全データのうち、何%をテストデータにするか
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) # ホールドアウト法を実行（テストデータはランダム選択）

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5,
                                                                                 min_samples_leaf=2,
                                                                                 min_samples_split=2, 
                                                                                 random_state=1234,
                                                                                 criterion="gini"),
                                           n_estimators=100, random_state=1234)
clf.fit(X_train, y_train)
print("score=", clf.score(X_train, y_train))

# 説明変数の重要度を出力する
# scikit-learnで算出される重要度は、ある説明変数による不純度の減少量合計である。
print(clf.feature_importances_)
pd.DataFrame(clf.feature_importances_, index=["goal","backers","usd_goal_real"]).plot.bar(figsize=(7,2))
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()

y_est = clf.predict(X_test) # ラベルを予測
show_results(y_test, y_est)

print('\nアダブーストにより、ランダムフォレストを超える精度を得ることができた')
print('10-3. ニューラルネットワークを用いる')
# データ作成関数を呼ぶ
df_data_test_10_3 = make_test_data(decorre=True, usd_goal_real_del=False)

# Test用データを分離する
y = df_data_test_10_3['Success'].values
X = df_data_test_10_3.drop('Success', axis=1).values
test_size = 0.2        # 全データのうち、何%をテストデータにするか（今回は20%に設定）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) # ホールドアウト法を実行（テストデータはランダム選択）

# one-hotベクトルに変換
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# モデルを作成する
model = Sequential()
model.add(Dense(6, activation='relu', input_dim=185))
model.add(Dense(12, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(2, activation='softmax'))#最終層のactivationは変更しないこと
# ------ 最適化手法 ------
adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# 計算の実行
fit = model.fit(X_train, y_train,
          epochs=300,
          batch_size=3000,validation_data=(X_test, y_test))

# 各epochにおける損失と精度をdfに入れる
df = pd.DataFrame(fit.history)

# グラフ化
df[["loss", "val_loss"]].plot()
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

df[["acc", "val_acc"]].plot()
plt.ylabel("acc")
plt.xlabel("epoch")
plt.ylim([0,1.0])
plt.show()

y_test_est = np.array(np.argmax(model.predict(X_test),axis=1), dtype='bool')
y_test = np.array(np.argmax(y_test,axis=1), dtype='bool')
show_results(y_test, y_test_est)

print('\n結果、Accuracyが92%超となり、最高の精度を出した')
print('Neural Networkはスゴい！')