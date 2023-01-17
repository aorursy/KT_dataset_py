# データを展開します（※毎回実行してください）
!unzip -o ../input/data.zip
# 各種ライブラリを import します
import pandas as pd
from pandas import DataFrame

df_den1 = pd.read_csv(u'den1.csv')
df_den2 = pd.read_csv(u'den2.csv')
df_sei1 = pd.read_csv(u'sei1.csv')
df_sei2 = pd.read_csv(u'sei2.csv')
df_sei3 = pd.read_csv(u'sei3.csv')
df_syu = pd.read_csv(u'syu.csv')
df_ket5 = pd.read_csv(u'ket5.csv')
df_uma  = pd.read_csv(u'uma.csv')
df = pd.merge(df_den2, df_sei2, on=['開催場所コード', '開催回次', '開催日次', 'レース番号', '馬番'], how='left')

df_20180527 = df[df['年月日_x']==20180527].copy()
df = df[df['年月日_x']!=20180527]

df.reset_index(drop=True, inplace=True)
df_20180527.reset_index(drop=True, inplace=True)
df['FLG'] = 0
df.loc[(df['確定着順'] > 0)  &  (df['確定着順'] <= 3), 'FLG'] = 1
df_X = df['レイティング']
df_X[df_X == '   '] = '0'
df_X = df_X.map(lambda x : int(x))
df_y = df['FLG']

X = df_X.values.reshape(-1, 1)
y = df_y.values.reshape(-1, 1)
from sklearn import linear_model, svm
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
%matplotlib inline

# Evaluate the performance
train_sizes, train_scores, valid_scores = learning_curve(svm.SVC(), X, y, train_sizes=[1000, 2000, 3000, 4000, 5000], cv=5)
plt.plot([1000, 2000, 3000, 4000, 5000], train_scores.mean(axis=1), label='train')
plt.plot([1000, 2000, 3000, 4000, 5000], valid_scores.mean(axis=1), label='valid')
plt.legend()
plt.show()
print(valid_scores.mean(axis=1))


model = svm.SVC()
model.fit(X, y)
df_X_test = df_20180527['レイティング']
df_X_test[df_X_test == '   '] = '0'
df_X_test = df_X_test.map(lambda x : int(x))
X_test = df_X_test.values.reshape(-1, 1)
y_pred = model.predict(X_test)
df_result = DataFrame()
df_result['ID'] = df_20180527['開催場所コード'].apply(lambda x: '{0:02d}'.format(x))
df_result['ID'] += df_20180527['レース番号'].apply(lambda x: '-{0:02d}'.format(x))
df_result['ID'] += df_20180527['馬番'].apply(lambda x: '-{0:02d}'.format(x))
df_result['FLG'] = y_pred

df_result.to_csv('prediction_sample.csv', index=None)

