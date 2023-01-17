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
'''

■クロスバリデーション(モデルが過学習してないか、精度はどうかの評価に使う、クリアしたモデルで予測値を出す)

１．モデルの種類とハイパパラメタを指定し、モデルを作成する

２．学習データを与えてモデルを学習させるとともにバリデーションを行いモデルを評価する

３．学習したモデルでテストデータに対して予測を行い、予測値を提出



■xgboost

1.目的変数と予測値から計算される目的変数を改善するように決定木を作成してモデルに追加

2.1をハイパパラメタで定めた決定木の本数の分だけ繰り返す

・特徴量は数値

・欠損値を扱える

・変数間の相互作用が反映される

'''

import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.metrics import log_loss



#xgboostモデルの実装-----------------------------------------------------------

class Model:



    def __init__(self, params=None):

        self.model = None

        if params is None:

            self.params = {}

        else:

            self.params = params



    def fit(self, tr_x, tr_y):

        #params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}

        params.update(self.params)

        num_round = 10

        dtrain = xgb.DMatrix(tr_x, label=tr_y)

        self.model = xgb.train(params, dtrain, num_round)



    def predict(self, x):

        data = xgb.DMatrix(x)

        pred = self.model.predict(data)

        return pred



#データの準備-------------------------------------------------------------------

train = pd.read_csv('../input/titanic-data/train.csv')

#NameはすべてユニークでPassengerIDと被るため削除,TicketとCabinも同様

train_x = train.drop(["Name","Ticket", "Cabin", "Survived"], axis=1)

train_y = train['Survived']

test_x = pd.read_csv('../input/titanic-data/test.csv')

test_x = test_x.drop(["Name", "Ticket", "Cabin"], axis=1)



#今回はひとまずlabel型のnullデータは適当な値で埋める→nullデータの決め方はFeature Engineering へ

#train_x = train_x.fillna({'Cabin': 'A00', 'Embarked': "D"})

train_x = train_x.fillna({'Embarked': "D"})

#test_x = test_x.fillna({'Cabin': 'A00'})



#同様に数値型のnullデータはひとまず平均値で埋める

train_x['Age'] = train_x['Age'].fillna(train_x['Age'].mean()) 

test_x['Age'] = test_x['Age'].fillna(test_x['Age'].mean()) 

test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean()) 



#train_x.info()

#test_x.info()



#変換するlabel変数のリスト

label_cols = ["Sex", "Embarked"]



#カテゴリ変数をループしてlabel encoding---------------------------------------------

#テストデータにはトレーニングデータセットに含まれていない値があるとうまく変換できない

for c in label_cols:

    # 学習データに基づいて定義する

    le = LabelEncoder()

    le.fit(train_x[c])

    train_x[c] = le.transform(train_x[c])

    test_x[c] = le.transform(test_x[c])

    

#train_x.info()

#test_x.info()



print(train_x)

print(test_x)



# クロスバリデーション-------------------------------------------------------------

#モデルの学習・評価のため、予測値を出すのは別

# 学習データを4つに分け、うち1つをバリデーションデータとする

# どれをバリデーションデータとするかを変えて学習・評価を4回行う

scores_ll = []

params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71, 'eval_metric': 'logloss'}



kf = KFold(n_splits=4, shuffle=True, random_state=71)

for tr_idx, va_idx in kf.split(train_x):

    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]

    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    model = Model(params)

    model.fit(tr_x, tr_y)

    va_pred = model.predict(va_x)

    score_ll = log_loss(va_y, va_pred)

    scores_ll.append(score_ll) 



print(scores_ll)

# クロスバリデーションの平均のスコアを出力する

print(f'logloss: {np.mean(scores_ll):.4f}')

print()

#print(list(kf.split(train_x)))

#この時点でtr_x,va_xはなにがはいっているか

#print(tr_x)

print(va_pred.shape)



# 学習データとバリデーションデータのスコアのモニタリング----------------------------------

# モニタリングをloglossで行い、アーリーストッピングの観察するroundを20とする

dtrain = xgb.DMatrix(tr_x, label=tr_y)

dvalid = xgb.DMatrix(va_x, label=va_y)

dtest = xgb.DMatrix(test_x)



params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71, 'eval_metric': 'logloss'}

num_round = 500

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

model = xgb.train(params, dtrain, num_round, evals=watchlist, early_stopping_rounds=20)



# 最適な決定木の本数で予測を行う

pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)



#予測（二値の予測値ではなく、1である確率を出力するようにしている）

print(pred)

print(pred.shape)
'''

■ニューラルネット

・特徴量は数値

・欠損値を扱うことはできない

・非線形や変数間の相互作用が反映される

・基本的には特徴量を標準化などでスケーリングする必要がある

・ハイパパラメタ次第で精度が出ないことがある

・多クラス分類に強い

・GPUで高速化



■ニューラルネット―kerasでの使い方のポイント

・回帰の場合はmean_squared_errorを設定するすることで平均二乗誤差を最小化するように学習

・二値分類の時はbinary_crossentropyを設定することでloglossを最小化するように学習

・マルチクラス分類の場合はcategorical_crossentropyを設定しmulti-class loglossを最小化するように学習

'''

import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential

from keras.layers import Input, Dense, Dropout, BatchNormalization

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping



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



# 学習データを学習データとバリデーションデータに分ける

kf = KFold(n_splits=4, shuffle=True, random_state=71)

tr_idx, va_idx = list(kf.split(train_x))[0]

tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]

tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]



# データのスケーリング

scaler = StandardScaler()

tr_x = scaler.fit_transform(tr_x)

va_x = scaler.transform(va_x)

test_x = scaler.transform(test_x)



#tr_x_std_df = pd.DataFrame(tr_x_std)

#va_x_std_df = pd.DataFrame(va_x_std)

#print(train_x)



#ニューラルネットの実装-------------------------------------------------------------------

model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(train_x.shape[1],)))

model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# 学習の実行

# バリデーションデータもモデルに渡し、学習の進行とともにスコアがどう変わるかモニタリングする

batch_size = 128

epochs = 50

history = model.fit(tr_x, tr_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(va_x, va_y))



#lossの可視化

plt.plot(history.epoch, history.history["loss"], label="loss")

plt.xlabel("epoch")

plt.legend()

plt.show()



# バリデーションデータでのスコアの確認

va_pred = model.predict(va_x)

score = log_loss(va_y, va_pred, eps=1e-7)

print(f'logloss: {score:.4f}')



# 予測データ

pred = model.predict(test_x)

#print(pred.shape)

#print(pred)

pred_01 = np.where(pred < 0.5, 0, 1)

print(pred_01)





# アーリーストッピング--------------------------------------------------------------------

# アーリーストッピングの観察するroundを20とする

# restore_best_weightsを設定することで、最適なエポックでのモデルを使用する

epochs = 50

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)



history = model.fit(tr_x, tr_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(va_x, va_y), callbacks=[early_stopping])

pred = model.predict(test_x)



# アーリーストッピングによる予測データ

print(pred)

pred_01 = np.where(pred < 0.5, 0, 1)

print(pred_01)



#lossの可視化

plt.plot(history.epoch, history.history["loss"], label="loss")

plt.xlabel("epoch")

plt.legend()

plt.show()
'''

■線形モデル

単体では精度は高くなくGBDTやニューラルネットとアンサンブルで組み合わせることで効果を発揮する

データが不十分であったりノイズが多いなど過学習しやすいようなデータでは活躍する



目的関数について

・回帰の場合Ridgeモデルなど平均二乗誤差を最小化するように学習する

・分類の場合

　二値分類の場合はloglossを最小化するように学習

 マルチクラス分類はone-vs-restと呼ばれるあるクラスとそれ以外のクラスの2値分類を繰り返す方法で学習(multi-class loglossを最小化もあり)

・ハイパパラメタは基本的に正則化の強さを表す係数のみ



特徴

・特徴量は数値

・欠損値を扱うことはでできない

・GBDTやニューラルネットと比較して精度は高くない

・非線形性を表現するためには明示的に特徴量を作成する必要がある

・相互作用を表現するには明示的に特徴量を作成する必要がある

・特徴量は基本的には標準化が必要

・特徴量を作るときには丁寧な処理が必要

・L1正則化を行った場合予測に寄与していない特徴量の係数が0になる性質がある



'''

import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt



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

print(pred)
'''

■k近傍法

レコード間の距離をそれらの特徴量の値の差を用いて定義しその距離が最も近いk個のレコードの目的変数から分類、回帰を行う

値のスケールが大きい特徴量が重視されすぎないように特徴量の標準化などのスケーリングを行う



'''

import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt



#データの前処理----------------------------------------------------------------------------

train_df = pd.read_csv('../input/titanic-data/train.csv')

test_df = pd.read_csv('../input/titanic-data/test.csv')

train_y = train_df['Survived']

#test_dfから、TicketとCabinの特徴量を削除

train_df = train_df.drop(["Ticket", "Cabin"], axis=1)

test_df = test_df.drop(["Ticket", "Cabin"], axis=1)



combine = [train_df, test_df]

#train_dfとtest_dfそれぞれに対して、Titleという新しい特徴量の中に、ドット(.)より前の敬称を格納してください。

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



#SexとTitleでクロス集計をしてください。クロス集計はcrosstabを利用するとできます。

df_list = pd.crosstab(train_df['Title'], train_df['Sex'])

#print(df_list)



#Master, Mr, Miss, Mrsなどの敬称が存在することが分かりました。今度は頻出な値以外をRareという値に置き換え

for dataset in combine:

    # 1. train_df, test_dfのTitleで、'Lady', 'Countess','Capt', 'Col',　'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'以外の項目に関しては'Rare'に書き換え

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    #同様に、MileはMissに書き換えてください。

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    #MmeはMrsに書き換えてください。

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



#TitleとSurvivedで、Titleで集計してSurvivedの平均値を算出    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()



#これらの敬称を、予測モデルにしやすいように順序データに変換

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in combine:

    dataset["Title"] = dataset["Title"] .map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})

    dataset['Title'] = dataset['Title'].fillna(0)



#NameとPassangerIdをtrain_dfから削除----------------------------------------------------

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)



#Nameをtest_dfから削除

test_df = test_df.drop(['Name'], axis=1)



combine = [train_df, test_df]

#train_dfとtest_dfの行数と列数を確認。

#print(train_df.shape, test_df.shape)    

    

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



#先頭行を抽出

#print(train_df.head())



#連続値であるAgeを任意の個数で分割し離散値に変換----------------------------------------------

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)



#AgeBandとSurvivedのピボットテーブルを作成

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)



for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

    

train_df = train_df.drop(['AgeBand'], axis=1)



combine = [train_df, test_df]

#ParchとSibSpを足し合わせた、FamilySizeという特徴量を新規に作成

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

#FamilySizeとSurvivedの平均値をグループで集計してください。--------------------------------

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)



#isAloneという新しい特徴量を作成します。この特徴量には独身者なら1、家族持ちなら0が格納---------

for dataset in combine:

    # 1. IsAloneという特徴量を全て0として作成

    dataset['IsAlone'] = 0

    # 2. FamilySizeが1の時、isAloneを1。

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

#IsAloneとSurvivedをグループ集計して、Survivedの平均値を出力

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()



#train_dfとtest_dfからParch、SibSp、およびFamilySizeを削除------------------------------

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)



combine = [train_df, test_df]



for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)



#Embarkedでnaを削除し、最頻値を新たな特徴量freq_portに格納="S"---------------------------

freq_port = train_df.Embarked.dropna().mode()[0]

#train_dfとtest_dfの欠損値を、freq_portの値に置き換え

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)



#EmbarkedとSurvivedをグループ集計して、Survivedの平均値を出力

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)



#Embarkedを{'S': 0, 'C': 1, 'Q': 2}に置き換えて下さい。

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



#test_dfのFareの欠損値を、Fareのmedianで埋める----------------------------------------

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



#Fareを4階層に分けたFareBandという特徴量を作成-----------------------------------------



#連続値であるFareを4個に分割し離散値へ変換

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)



#FareBandとSurvivedのピボットテーブルを作成

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

#Fareの値が7.91以下なら0、7.91超え14.454以下なら1、14.454超え31以下なら2、31超え3へ変換

for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand','Survived'], axis=1)

test_df = test_df.drop(['PassengerId'], axis=1)

combine = [train_df, test_df]



train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean()) 

train_df['Age*Class'] = train_df['Age*Class'].fillna(train_df['Age*Class'].mean()) 

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean()) 

test_df['Age*Class'] = test_df['Age*Class'].fillna(test_df['Age*Class'].mean()) 

    

train_x = train_df.copy()

test_x = test_df.copy()



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



#k近傍法の学習

knn = KNeighborsClassifier(n_neighbors=6, p=2, metric='minkowski')

knn.fit(tr_x, tr_y)



#va_pred = knn.predict(va_x)

va_pred = knn.predict(va_x)

score = log_loss(va_y, va_pred)

print(f'logloss: {score:.4f}')



# 予測

pred = knn.predict(test_x)

print(pred)
'''

■ランダムフォレスト

決定木の集合により予測を行うモデル。GBDTとは違い並列に決定木を作成する。



モデルの作成

1．学習データからレコードをサンプリングして抽出する

2．1に対して学習を行い決定木を作成する

　　分岐を作成するときに特徴量の一部のみをサンプリングして抽出し特徴量の候補とします。

  　それらの特徴量の候補からデータを最もよく分割する特徴量と閾値を選び分岐とする



決定木作成のポイント

・分岐は回帰タスクでは二乗誤差、分類タスクではジニ不純度が最も減少するように行う

・決定木ごとに元の個数と同じレコード数を復元抽出するブーストトラップサンプリングが行われる

・分岐ごとに特徴量の一部をサンプリングしたものを候補としその中から分岐の特徴量を選ぶ

・決定木の本数と精度の関係

・out-of-bag

・予測確率の妥当性

'''

import numpy as np

import pandas as pd

from sklearn.metrics import log_loss

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier



#データの前処理----------------------------------------------------------------------------

train_df = pd.read_csv('../input/titanic-data/train.csv')

test_df = pd.read_csv('../input/titanic-data/test.csv')

train_y = train_df['Survived']

#test_dfから、TicketとCabinの特徴量を削除

train_df = train_df.drop(["Ticket", "Cabin"], axis=1)

test_df = test_df.drop(["Ticket", "Cabin"], axis=1)



combine = [train_df, test_df]

#train_dfとtest_dfそれぞれに対して、Titleという新しい特徴量の中に、ドット(.)より前の敬称を格納してください。

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



#SexとTitleでクロス集計をしてください。クロス集計はcrosstabを利用するとできます。

df_list = pd.crosstab(train_df['Title'], train_df['Sex'])

#print(df_list)



#Master, Mr, Miss, Mrsなどの敬称が存在することが分かりました。今度は頻出な値以外をRareという値に置き換え

for dataset in combine:

    # 1. train_df, test_dfのTitleで、'Lady', 'Countess','Capt', 'Col',　'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'以外の項目に関しては'Rare'に書き換え

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    #同様に、MileはMissに書き換えてください。

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    #MmeはMrsに書き換えてください。

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



#TitleとSurvivedで、Titleで集計してSurvivedの平均値を算出    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()



#これらの敬称を、予測モデルにしやすいように順序データに変換

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in combine:

    dataset["Title"] = dataset["Title"] .map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})

    dataset['Title'] = dataset['Title'].fillna(0)



#NameとPassangerIdをtrain_dfから削除----------------------------------------------------

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)



#Nameをtest_dfから削除

test_df = test_df.drop(['Name'], axis=1)



combine = [train_df, test_df]

#train_dfとtest_dfの行数と列数を確認。

#print(train_df.shape, test_df.shape)    

    

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



#先頭行を抽出

#print(train_df.head())



#連続値であるAgeを任意の個数で分割し離散値に変換----------------------------------------------

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)



#AgeBandとSurvivedのピボットテーブルを作成

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)



for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

    

train_df = train_df.drop(['AgeBand'], axis=1)



combine = [train_df, test_df]

#ParchとSibSpを足し合わせた、FamilySizeという特徴量を新規に作成

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

#FamilySizeとSurvivedの平均値をグループで集計してください。--------------------------------

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)



#isAloneという新しい特徴量を作成します。この特徴量には独身者なら1、家族持ちなら0が格納---------

for dataset in combine:

    # 1. IsAloneという特徴量を全て0として作成

    dataset['IsAlone'] = 0

    # 2. FamilySizeが1の時、isAloneを1。

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

#IsAloneとSurvivedをグループ集計して、Survivedの平均値を出力

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()



#train_dfとtest_dfからParch、SibSp、およびFamilySizeを削除------------------------------

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)



combine = [train_df, test_df]



for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)



#Embarkedでnaを削除し、最頻値を新たな特徴量freq_portに格納="S"---------------------------

freq_port = train_df.Embarked.dropna().mode()[0]

#train_dfとtest_dfの欠損値を、freq_portの値に置き換え

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)



#EmbarkedとSurvivedをグループ集計して、Survivedの平均値を出力

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)



#Embarkedを{'S': 0, 'C': 1, 'Q': 2}に置き換えて下さい。

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



#test_dfのFareの欠損値を、Fareのmedianで埋める----------------------------------------

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



#Fareを4階層に分けたFareBandという特徴量を作成-----------------------------------------



#連続値であるFareを4個に分割し離散値へ変換

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)



#FareBandとSurvivedのピボットテーブルを作成

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

#Fareの値が7.91以下なら0、7.91超え14.454以下なら1、14.454超え31以下なら2、31超え3へ変換

for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand','Survived'], axis=1)

test_df = test_df.drop(['PassengerId'], axis=1)

combine = [train_df, test_df]



train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean()) 

train_df['Age*Class'] = train_df['Age*Class'].fillna(train_df['Age*Class'].mean()) 

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean()) 

test_df['Age*Class'] = test_df['Age*Class'].fillna(test_df['Age*Class'].mean()) 

    

train_x = train_df.copy()

test_x = test_df.copy()



# 学習データを学習データとバリデーションデータに分ける--------------------------------------------

kf = KFold(n_splits=4, shuffle=True, random_state=71)

tr_idx, va_idx = list(kf.split(train_x))[0]

tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]

tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]



#ランダムフォレストで学習する-----------------------------------------------------------------

forest = RandomForestClassifier(criterion='gini', n_estimators=150, random_state=1, n_jobs=2)

forest.fit(tr_x, tr_y)

va_pred = forest.predict(va_x)

score = log_loss(va_y, va_pred)

print(f'logloss: {score:.4f}')



# 予測

pred = forest.predict(test_x)

pred = pred.tolist()

print(pred)



#submission用にデータを変換

sub = pd.read_csv('../input/titanic-data/gender_submission.csv')

sub['Survived'] = list(map(int, pred))

sub.to_csv('submission_rf2.csv', index=False)


