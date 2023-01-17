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
from IPython.core.display import display 
from datetime import datetime

# Data is downloaded and ready to use!
df_data = pd.read_csv("../input/ks-projects-201801.csv")
print("Row lengths of imported data: ", len(df_data))
#まずはHeaderを確認
display(df_data.head())
df_data.describe()
#Stateの種類と数を調べる
print(df_data['state'].value_counts(dropna=False))
#Category毎にStateとの相関をグラフ化する
category_ = df_data.groupby('category')
category_ = category_['state'].value_counts(normalize=True).unstack()
category_ = category_.sort_values(by=['successful'],ascending=True)
category_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
print("成功しやすいCategoryと成功しにくいCateogryが存在する")
print("Maxは80%近く、Minは10%以下")
#main_Category毎にStateとの相関をグラフ化する
m_category_ = df_data.groupby('main_category')
m_category_ = m_category_['state'].value_counts(normalize=True).unstack()
m_category_ = m_category_.sort_values(by=['successful'],ascending=True)
m_category_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,5))
print("成功しやすいMain_categoryと成功しにくいMain_Cateogryが存在する")
print("Maxは60%超、Minは20%以下")
#currency毎にStateとの相関をグラフ化する
currency_ = df_data.groupby('currency')
currency_ = currency_['state'].value_counts(normalize=True).unstack()
currency_ = currency_.sort_values(by=['successful'],ascending=True)
currency_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,5))
print("成功しやすい通貨と成功しにくい通貨が存在するが、")
print("Maxは40%近く、Minは20%以下で幅はそれほど大きくない")
#deadline毎にStateとの相関をグラフ化する
#数が多いので、年月別にする
df_data['deadline_YM'] = df_data['deadline'].apply(lambda x: x[0:7])
deadline_ = df_data.groupby('deadline_YM')
deadline_ = deadline_['state'].value_counts(normalize=True).unstack()
deadline_ = deadline_.sort_values('deadline_YM',ascending=False)
ax = deadline_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
plt.legend(loc='upper left')
print("2018年にLiveが多い。が、他の明確な傾向はつかみにくい")
#goal毎にStateとの相関をグラフ化する
#goalを10万単位で丸めて相関を見る
df_data['goal_r'] = df_data['goal'].apply(lambda x: round(x/100000))
goal_ = df_data.groupby('goal_r')
goal_ = goal_['state'].value_counts(normalize=True).unstack()
goal_ = goal_.sort_values('goal_r',ascending=False)
ax = goal_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
plt.legend(loc='upper left')
print("Goalが大き過ぎると、成功しにくいようだ")
#launched毎にStateとの相関をグラフ化する
#数が多いので、年月別にする
df_data['launched_YM'] = df_data['launched'].apply(lambda x: x[0:7])
launched_ = df_data.groupby('launched_YM')
launched_ = launched_['state'].value_counts(normalize=True).unstack()
launched_ = launched_.sort_values('launched_YM',ascending=False)
ax = launched_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
plt.legend(loc='upper left')
print("1970年に開始したものは失敗、2017年12月以降開始はLive、それ以外の傾向は見にくい")
#pledged毎にStateとの相関をグラフ化する
#pledgedを10万単位で丸めて相関を見る
df_data['pledged_r'] = df_data['pledged'].apply(lambda x: round(x/100000))
pledged_ = df_data.groupby('pledged_r')
pledged_ = pledged_['state'].value_counts(normalize=True).unstack()
pledged_ = pledged_.sort_values('pledged_r',ascending=False)
ax = pledged_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,20))
plt.legend(loc='upper left')
print("Pledgedが一定以上になると、ほぼ成功している")
#backers毎にStateとの相関をグラフ化する
#backersを1000単位で丸めて相関を見る
df_data['backers_r'] = df_data['backers'].apply(lambda x: round(x/1000))
backers_ = df_data.groupby('backers_r')
backers_ = backers_['state'].value_counts(normalize=True).unstack()
backers_ = backers_.sort_values('backers_r',ascending=False)
ax = backers_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,20))
plt.legend(loc='upper left')
print("Backersが一定以上になると、ほぼ成功している")
#country毎にStateとの相関をグラフ化する
country_ = df_data.groupby('country')
country_ = country_['state'].value_counts(normalize=True).unstack()
country_ = country_.sort_values(by=['successful'],ascending=True)
country_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,7))
print("成功しやすい国と成功しにくい国が存在するが、")
print("Maxは40%近く、Minは20%以下で幅はそれほど大きくない")
#usd pledged毎にStateとの相関をグラフ化する
#usd pledgedを10万単位で丸めて相関を見る 

#usd pledgeには欠損値があるので、欠損値のある行を削除してグラフ化する
df_data_t = df_data.dropna(subset=['usd pledged']).copy()
df_data_t['usd_pledged_r'] = df_data_t['usd pledged'].apply(lambda x: round(x/100000))
usd_pledged_ = df_data_t.groupby('usd_pledged_r')
usd_pledged_ = usd_pledged_['state'].value_counts(normalize=True).unstack()
usd_pledged_ = usd_pledged_.sort_values('usd_pledged_r',ascending=False)
#ax = usd_pledged_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,20))
ax = usd_pledged_[['successful','failed','live','canceled','suspended']].plot(kind='barh', stacked=True,figsize=(13,15))
plt.legend(loc='upper left')
print("USD Pledgedが一定以上だとほぼ成功、Pledgedと同様")
#usd_pledged_real毎にStateとの相関をグラフ化する
#usd_pledged_realを10万単位で丸めて相関を見る
df_data['usd_pledged_real_r'] = df_data['usd_pledged_real'].apply(lambda x: round(x/100000))
usd_pledged_real_ = df_data.groupby('usd_pledged_real_r')
usd_pledged_real_ = usd_pledged_real_['state'].value_counts(normalize=True).unstack()
usd_pledged_real_ = usd_pledged_real_.sort_values('usd_pledged_real_r',ascending=False)
ax = usd_pledged_real_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,20))
plt.legend(loc='upper left')
print("USD Pledged Realが一定以上になるとほぼ成功、Pledgedと同じ")
#usd_goal_real毎にStateとの相関をグラフ化する
#usd_goal_realを100000単位で丸めて相関を見る
df_data['usd_goal_real_r'] = df_data['usd_goal_real'].apply(lambda x: round(x/100000))
usd_goal_real_ = df_data.groupby('usd_goal_real_r')
usd_goal_real_ = usd_goal_real_['state'].value_counts(normalize=True).unstack()
usd_goal_real_ = usd_goal_real_.sort_values('usd_goal_real_r',ascending=False)
ax = usd_goal_real_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
plt.legend(loc='upper left')
print("USD Goal Realが大きすぎると成功しにくい、Goalと同じ")
#成功かどうかを判断するため、stateが"Successful"なるTrue、それ以外はFalseとする
df_data_test = df_data.copy()
df_data_test['Success'] = df_data_test['state'] == "successful"

# カテゴリー変数をダミー変数に変換
df_data_dummy1 = pd.get_dummies(df_data_test['category'])
df_data_dummy2 = pd.get_dummies(df_data_test['currency'])
df_data_test = pd.merge(df_data_test, df_data_dummy1, left_index=True, right_index=True)
df_data_test = pd.merge(df_data_test, df_data_dummy2, left_index=True, right_index=True)

# 不要な列を削除する
df_data_test = df_data_test.drop(['ID','name','category','main_category','currency','deadline'], axis=1)
df_data_test = df_data_test.drop(['launched','state','country','usd pledged','usd_pledged_real','usd_goal_real'], axis=1)
df_data_test = df_data_test.drop(['deadline_YM','goal_r','launched_YM','pledged_r','backers_r','usd_pledged_real_r','usd_goal_real_r'], axis=1)

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
conf_mat