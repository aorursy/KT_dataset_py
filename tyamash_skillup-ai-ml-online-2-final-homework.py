print('取り組む課題を１つ選択する')
print("I choose 'Kickstarter Projects'")

print('必要なLibraryをImportする')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt #グラフを描く
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
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
from keras.utils import np_utils
import warnings
warnings.filterwarnings('ignore')

print('データをダウンロードし、ダウンロードした結果を確認する')
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
print('launchedとdeadlineの差でterm(期間)という説明変数を作成し、Stateとの相関をグラフ化する')
df_data_term = df_data.copy()
df_data_term['term'] = pd.to_datetime(df_data_term['deadline']).map(pd.Timestamp.timestamp) - pd.to_datetime(df_data_term['launched'].apply(lambda x: x[0:10])).map(pd.Timestamp.timestamp)
term_ = df_data_term.groupby('term')
term_ = term_['state'].value_counts(normalize=True).unstack()
ax = term_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
plt.legend(loc='upper left')
print("\n２つのピークのある相関が描けた")
print("成功の確率を上げる期間があるのかもしれない")
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
                   launched_del=True, \
                   deadline_del=True, \
                   usd_goal_real_del=True, \
                   category_keep=True, \
                   country_keep=True \
                  ):
    if(show_comment):
        print('成功かどうかを判断するため、stateが"Successful"はTrue、"Failed"はFalse、それ以外は削除する')
    df_data_test = df_data.copy()
    df_data_test = df_data_test[df_data_test['state'] != 'live']
    df_data_test = df_data_test[df_data_test['state'] != 'canceled']
    df_data_test = df_data_test[df_data_test['state'] != 'suspended']
    df_data_test = df_data_test[df_data_test['state'] != 'undefined']
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

    # Goalとusd_goal_realをInt化する
    df_data_test['goal'] = df_data_test['goal'].astype(np.int64)
    df_data_test['usd_goal_real'] = df_data_test['usd_goal_real'].astype(np.int64)
    
    #日付関連の変換処理
    df_data_test['launched'] = pd.to_datetime(df_data_test['launched'].apply(lambda x: x[0:10])).map(pd.Timestamp.timestamp)
    df_data_test['deadline'] = pd.to_datetime(df_data_test['deadline']).map(pd.Timestamp.timestamp)
    df_data_test['term'] = df_data_test['deadline'] - df_data_test['launched']


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

def preprocessing_(df_data_test,\
                   stdsc=False, \
                   mms=False, \
                   decorre=False, \
                   show_decorre_detail = False, \
                   whitening=False \
                  ):
    
    if(decorre):
        if(show_decorre_detail):
            # goalとusd_goal_realを白色化
            print('goalとusd_goal_realの相関係数: {:.3f}'.format( \
                  np.corrcoef( \
                  df_data_test['goal'], \
                  df_data_test['usd_goal_real'])[0,1]))
        #  無相関化を行うための一連の処理
        cov = np.cov(df_data_test[['goal','usd_goal_real']], rowvar=0) # 分散・共分散を求める
        _, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて
        df_data_test[['goal','usd_goal_real']] = np.dot(S.T, df_data_test[['goal','usd_goal_real']].T).T #データを無相関化
        if(show_decorre_detail):
            print('無相関後のgoalとusd_goal_realの相関係数: {:.3f}'.format(np.corrcoef(df_data_test['goal'], df_data_test['usd_goal_real'])[0,1]))

    if(whitening):
        stdsc = True

    # 標準化とは、平均を引いて、標準偏差で割る操作
    if(stdsc):
        df_data_test = (df_data_test - df_data_test.mean()) / df_data_test.std()

    # 正規化とは、全データを0-1の範囲におさめる操作
    if(mms):
        min_max_scaler = MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(df_data_test)
        df_data_test = pd.DataFrame(np_scaled)

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
                ridge_alpha=0, \
                stdsc=False, \
                mms=False, \
                decorre=False, \
                show_decorre_detail = False, \
                whitening=False, \
                DecisionTree_use=False, \
                RandomForest_use=False, \
                AdaBoost_use=False, \
                GradientBoost_use=False,\
                n_estimators=100, \
                min_samples_split=3, \
                min_samples_leaf=3, \
                max_depth=4 \
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

        # ここで前処理を行う
        X_train = preprocessing_(X_train, stdsc=stdsc, mms=mms, decorre=decorre, show_decorre_detail = show_decorre_detail, whitening=whitening)
        X_test = preprocessing_(X_test, stdsc=stdsc, mms=mms, decorre=decorre, show_decorre_detail = False, whitening=whitening)
        
        if(SGDClass_use):
            clf = SGDClassifier(loss='log', penalty='none', fit_intercept=True, random_state=1234)
        if(lasso_use):
            clf = Lasso(alpha=lasso_alpha)
        if(ridge_use):
            #clf = Ridge(alpha=ridge_alpha)
            clf = RidgeClassifier(alpha=ridge_alpha)
        if(DecisionTree_use):
            clf = DecisionTreeClassifier(criterion="gini", \
                                         max_depth=None, \
                                         min_samples_split=min_samples_split, \
                                         min_samples_leaf=min_samples_leaf, \
                                         random_state=1234)
        if(RandomForest_use):
            clf = RandomForestClassifier(n_estimators=n_estimators, \
                                         max_depth=max_depth, \
                                         criterion="gini", \
                                         min_samples_leaf=min_samples_leaf, \
                                         min_samples_split=min_samples_split, \
                                         random_state=1234)

        if(AdaBoost_use):
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth,
                                                            min_samples_leaf=min_samples_leaf,
                                                            min_samples_split=min_samples_split, 
                                                            random_state=1234,
                                                            criterion="gini"),
                                                            n_estimators=n_estimators, 
                                     random_state=1234)
        if(GradientBoost_use):
            clf = GradientBoostingClassifier(n_estimators=n_estimators, 
                                             max_depth=max_depth, 
                                             criterion="friedman_mse", \
                                             min_samples_leaf=min_samples_leaf, 
                                             min_samples_split=min_samples_split, 
                                             random_state=1234)
            
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
    
print("最も基本となる方法で正答率を求める\n")
# データ作成関数を呼ぶ
df_data_test_4 = make_test_data(show_comment=True, show_data=True)

# 絞った説明変数を使って，ロジスティック回帰
# 基本的にクロスバリデーションで評価する
y, y_est = cross_valid(df_data_test_4)

# 結果を表示
show_results(y, y_est)
print('まずは正規化から')
# データ作成関数を呼ぶ
df_data_test_8_3a1 = make_test_data(show_data=False)
# 絞った説明変数を使って，ロジスティック回帰
y, y_est = cross_valid(df_data_test_8_3a1, stdsc=True, mms=False)
# 結果を表示
show_results(y, y_est)

print('\n次に標準化')
# データ作成関数を呼ぶ
df_data_test_8_3a2 = make_test_data(show_data=False)
# 絞った説明変数を使って，ロジスティック回帰
y, y_est = cross_valid(df_data_test_8_3a2, stdsc=False, mms=True)
# 結果を表示
show_results(y, y_est)

print('\n結果、標準化・正規化とも正答率が上がり、標準化の方が効果が高いようだ')
print('以降、標準化をデフォルトで実施する')
print('Ridge（L2正則化）を例に正則化を行ってみる')

# データ作成関数を呼ぶ
df_data_test_8_3c1 = make_test_data()

# Test用データを分離する
test_size = 0.2        # 全データのうち、何%をテストデータにするか（今回は20%に設定）
dataset_train, dataset_test, dummy_train, dummy_test = \
train_test_split(df_data_test_8_3c1, df_data_test_8_3c1, test_size=test_size, random_state=1234) # ホールドアウト法を実行（テストデータはランダム選択）

# パラメータ検証用のクロスバリデーションを実施
alphas = [0, 0.0005, 0.001, 0.005, 0.01] #alpha(数式ではλ)の値を4つ指定する
n_split = 5 # グループ数を設定（今回は5分割）
cross_valid_losses = []
print("Alphas = ", alphas)

for alpha in alphas:
    cross_valid_loss = 0
    split_num = 1

    # テスト役を交代させながら学習と評価を繰り返す
    y, y_est = cross_valid(dataset_train, mms=True, SGDClass_use=False, ridge_use=True, ridge_alpha=alpha)
    # 結果を表示
    print("\nAlpha = %s"%alpha)
    show_results(y, y_est)

# Best Scoreを表示
alpha=0.001
print("\nBest Alpha = %s"%alpha)
y, y_est = cross_valid(dataset_test, mms=True, SGDClass_use=False, ridge_use=True, ridge_alpha=alpha)
show_results(y, y_est)
    
#結果
print('幅広くパラメータを試した結果、alpha=0.001前後に対数尤度が最も小さくなり、正答率も上がった') 
print('まずは決定木を用いる')
print('木モデルなので、標準化は用いない')
# データ作成関数を呼ぶ
df_data_test_10_2_1 = make_test_data(show_data=False)
# クロスバリデーションで学習及び予測
y, y_est = cross_valid(df_data_test_10_2_1, mms=False, SGDClass_use=False, DecisionTree_use=True)
# 結果を表示
show_results(y, y_est)

print('次にランダムフォレストを用いる')
# データ作成関数を呼ぶ
df_data_test_10_2_1 = make_test_data(show_data=False)
# クロスバリデーションで学習及び予測
y, y_est = cross_valid(df_data_test_10_2_1, mms=False, SGDClass_use=False, RandomForest_use=True)
# 結果を表示
show_results(y, y_est)

print('その次は、アダブーストを用いる')
# データ作成関数を呼ぶ
df_data_test_10_2_1 = make_test_data(show_data=False)
# クロスバリデーションで学習及び予測
y, y_est = cross_valid(df_data_test_10_2_1, mms=False, SGDClass_use=False, AdaBoost_use=True)
# 結果を表示
show_results(y, y_est)

print('木モデルの最後として、勾配ブースティングを用いる')
# データ作成関数を呼ぶ
df_data_test_10_2_1 = make_test_data(show_data=False)
# クロスバリデーションで学習及び予測
y, y_est = cross_valid(df_data_test_10_2_1, mms=False, SGDClass_use=False, GradientBoost_use=True)
# 結果を表示
show_results(y, y_est)

print('ニューラルネットワークを用いる')
# データ作成関数を呼ぶ
df_data_test_10_3 = make_test_data()

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

y = df_data_test_10_3['Success'].values
X = df_data_test_10_3.drop('Success', axis=1).values

n_split = 5 # グループ数を設定（今回は5分割）
cross_valid_loss = 0
split_num = 1

# テスト役を交代させながら学習と評価を繰り返す
for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
    X_train, y_train = X[train_idx], y[train_idx] #学習用データ
    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    # ここで前処理を行う
    X_train = preprocessing_(X_train, mms=True)
    X_test = preprocessing_(X_test, mms=True)
        
    # one-hotベクトルに変換
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    #学習する        
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

    # テストデータでラベルを予測
    y_test_est = np.array(np.argmax(model.predict(X_test),axis=1), dtype='bool')
    y_test = np.array(np.argmax(y_test,axis=1), dtype='bool')

    if(split_num==1):
        y_ = y_test
        y_est = y_test_est
    else:
        y_ = np.append(y_, y_test)
        y_est = np.append(y_est, y_test_est)
            
    split_num += 1

show_results(y_, y_est)
