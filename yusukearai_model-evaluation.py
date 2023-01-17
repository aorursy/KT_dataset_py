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
#https://teratail.com/questions/111293
#決定領域の可視化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def plot_decision_regions(X, y, classifier, resolution=0.02):

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

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

# 学習データを学習データとバリデーションデータに分ける--------------------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

#ランダムフォレストで学習する-----------------------------------------------------------------
forest = RandomForestClassifier(criterion='gini', n_estimators=150, random_state=1, n_jobs=2)

#scikit-learnによる主成分分析-特徴量を2個に変換する
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(tr_x)
X_test_pca = pca.transform(va_x)
X_train_pca = X_train_pca[:50, :]
tr_yv = tr_y.values
tr_yv = tr_yv[:50]

forest.fit(X_train_pca, tr_yv)

print(X_train_pca)
print(tr_yv)

plot_decision_regions(X_train_pca, tr_yv, classifier=forest)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
#学習アルゴリズムに過学習または学習不足の問題がるかどうかを学習曲線を使って診断
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

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

# 学習データを学習データとバリデーションデータに分ける--------------------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

#学習曲線の可視化----------------------------------------------------------------------------
pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', random_state=1))
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=tr_x, y=tr_y, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()

plt.show()

#検証曲線を使って過学習と学習不足を明らかにする
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

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

# 学習データを学習データとバリデーションデータに分ける--------------------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

#検証曲線の可視化---------------------------------------------------------------------------
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', random_state=1))
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=tr_x, y=tr_y, param_name='logisticregression__C', param_range=param_range, cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.tight_layout()
# plt.savefig('images/06_06.png', dpi=300)
plt.show()
'''
■混同行列
二値分類の評価指標
・予測値と真の値の組み合わせは予測値を正例としたか負例としたかその予測が正しいか誤りかによって以下の4つに分けられる
TP(True Positive:真陽性)：予測値を正例としてその予測が正しい場合
TN(True Negative:真陰性)：予測値を負例としてその予測が正しい場合
FP(False Positive:偽陽性)：予測値を正例としてその予測が誤りの場合
FN(False Negative:偽陰性)：予測値を負例としてその予測が誤りの場合
'''
#線形モデル
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

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

# データのスケーリング---------------------------------------------------------------------
scaler = StandardScaler()
tr_x = scaler.fit_transform(tr_x)
va_x = scaler.transform(va_x)
test_x = scaler.transform(test_x)

# 線形モデルの構築・学習-------------------------------------------------------------------
model = LogisticRegression(C=1.0)
model.fit(tr_x, tr_y)

# バリデーションデータでのスコアの確認
# predict_probaを使うことで確率を出力できます。(predictでは二値のクラスの予測値が出力されます。)
va_pred = model.predict_proba(va_x)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

# 予測
pred = model.predict(test_x)
#print(pred)
print(pred.shape)
va_y01 = np.where(va_y < 0.5, 0, 1)
va_pred01 = np.where(va_pred < 0.5, 0, 1)
#va_pred01 = va_pred01[:, 1:2]
#配列をそろえる1次元のリストにする
va_pred01 = va_pred01[:, 1:2].flatten()
print(va_pred01)
print(va_y01)

#混同行列を作成する
#y_trueは真の値、予測するべき値、y_predは学習器を使って予測する値
y_true = va_y01
y_pred =va_pred01

confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()

#適合率、再現率、F1スコアを出力
print('Precision: %.3f' % precision_score(y_true=y_true, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_true, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_true, y_pred=y_pred))

'''
モデルが出力する予測確率がゆがんでいる場合予測確率に調整を加えることでスコアがよくなることがある
以下のような場合モデルによって出力される予測確率はゆがんでいると考えられる
・データが十分でない場合
・モデルの学習アルゴリズム上、妥当な確率を予測するように最適化されない場合

■予測確率の調整
・予測値をn乗する：予測を十分学習しきれてないとみて補正を考える
・極端に0や1に近い確率のクリップ：出力する確率の範囲を0.1-99.9%に制限
・スタッキング
・calibratedclassifeir：予測値の補正方法

'''


'''
■ROC曲線
横軸に偽陽性を縦軸にとりクラス分離の閾値を変化させたときの
それぞれの値のプロットすたもの
偽陽性：陰性と予測したうちの陽性の割合
真陽性：陽性と予測したうちの陽性の割合
＊偽陽性が0の状態で真陽性が高いモデルが予測率の高いモデル（図の左上）


■AUC
ROC曲線の横軸と縦軸に囲まれた部分の面積のこと

②グラフをどう解釈する?
"スコアが0.999999以上のみを陽性と見なすよ！"という厳しい条件においても、
陽性を正しく分類できるモデルは優れたモデルと言えますよね。
偽陽性(横軸)が０の状態というのは、まさに上記のような条件を再現したものだと言えます。
(実際に閾値がいくらに設定されているかは、入力によって異なるので、あくまで例です。)

③どんなグラフになる?
そして、偽陽性が高まる = (判定閾値が低くなり)陽性判定が増える = 真陽性は増えるという関係が
常に成り立つので、ROC曲線は必ず右上がりになります。

④AUCはこういうもの
っで、あれば曲線と横軸との間の面積が大きいモデルというのは、
'偽陽性が低い段階から正しく分類できていたモデル'となるわけですから、
AUC(ROC曲線の横軸と縦軸に囲まれた部分の面積)は分類モデルの
パフォーマンス評価指標として有用なわけです。

https://qiita.com/osapiii/items/a2ed9f638b51f3b22cd6
'''
#roc_auc_score()に、正解ラベルと予測スコアを渡すとAUCを計算
import numpy as np
from sklearn.metrics import roc_auc_score
y = np.array([1, 1, 2, 2])
pred = np.array([0.1, 0.4, 0.35, 0.8])
roc_auc_score(y, pred)
