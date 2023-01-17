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
#カテゴリデータ⇔数値データ変換 【手動ラベル変換】
import numpy as np
import pandas as pd

# train_xは学習データ、train_yは目的変数、test_xはテストデータ
# pandasのDataFrame, Seriesで保持します。（numpyのarrayで保持することもあります）
train = pd.read_csv('../input/titanic-data/train.csv')
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']
test_x = pd.read_csv('../input/titanic-data/test.csv')
#print(train_x)
#print()
#print(test_x)
train_x.info()
test_x.info()

#ラベルデータ→数値データに変換
sex_mapping = {"male":0, "female":1}
train_x["Sex"] = train_x["Sex"].map(sex_mapping)
print(train_x)

#数値データ→ラベルデータの変換
inv_sex_mapping = {value: key for key, value in sex_mapping.items()}
train_x["Sex"] = train_x["Sex"].map(inv_sex_mapping)
print(train_x)
## 学習データとテストデータを結合してget_dummiesによるone-hot encodingを行う
import numpy as np
import pandas as pd

train = pd.read_csv('../input/titanic-data/train.csv')
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']
test_x = pd.read_csv('../input/titanic-data/test.csv')

#データの結合
#変換するlabel変数のリスト
label_cols = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
#縦に結合する
all_x = pd.concat([train_x, test_x])
print(all_x)
all_x = pd.get_dummies(all_x, columns=label_cols)
print(all_x)

# 学習データとテストデータに再分割
#.reset_indexはインデックスの振り直し
train_x = all_x.iloc[:train_x.shape[0], :].reset_index(drop=True)
test_x = all_x.iloc[train_x.shape[0]:, :].reset_index(drop=True)
print(train_x)
print(test_x)
#one-hot encoderでの変換
#onehotencoderのtransformメソッドの返り値はnumpyの配列に変換されてしまい、元の列名や水準の情報が失われてしまう
#そのため残りの変数と結合する際に再度データフレームに変換しなおす必要がある
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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

print(train_x)
#label encodernによる変換
#各水準を整数に置き換える、大小関係に意味はない
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

#カテゴリ変数をループしてlabel encoding
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
#feature hashing　特徴量の数を減らす
#onehotencodingの省メモリ化
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

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

##n_featuresで各列ごとに生成する列の数を指定
n_features = 4

# カテゴリ変数をループしてfeature hashing
for c in label_cols:
    # FeatureHasherの使い方は、他のencoderとは少し異なる
    #n_featuresで各列ごとに生成する列の数を指定
    fh = FeatureHasher(n_features=n_features, input_type='string')
    # 変数を文字列に変換してからFeatureHasherを適用
    hash_train = fh.transform(train_x[[c]].astype(str).values)
    hash_test = fh.transform(test_x[[c]].astype(str).values)
    # データフレームに変換.todense()はに次元配列に変換する
    hash_train = pd.DataFrame(hash_train.todense(), columns=[f'{c}_{i}' for i in range(n_features)])
    hash_test = pd.DataFrame(hash_test.todense(), columns=[f'{c}_{i}' for i in range(n_features)])
    # 元のデータフレームと結合.axis=1で横方向に結合
    train_x = pd.concat([train_x, hash_train], axis=1)
    test_x = pd.concat([test_x, hash_test], axis=1)

# 元のカテゴリ変数を削除
train_x.drop(label_cols, axis=1, inplace=True)
test_x.drop(label_cols, axis=1, inplace=True)

print(train_x)
print(test_x)
# frequency encoding　出現頻度でカテゴリ分け
import numpy as np
import pandas as pd

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

# 変数をループしてfrequency encoding
for c in label_cols:
    freq = train_x[c].value_counts()
    # カテゴリの出現回数で置換
    train_x[c] = train_x[c].map(freq)
    test_x[c] = test_x[c].map(freq)
    
print(train_x)
print(test_x)
# target encoding
# 学習データ全体で各カテゴリにおけるtargetの平均を計算
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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

# 変数をループしてtarget encoding
for c in label_cols:
    # 学習データ全体で各カテゴリにおけるtargetの平均を計算
    data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
    target_mean = data_tmp.groupby(c)['target'].mean()
    # テストデータのカテゴリを置換
    test_x[c] = test_x[c].map(target_mean)

    # 学習データの変換後の値を格納する配列を準備
    #np.repeat(a,b)配列aの要素をb回繰り返す
    tmp = np.repeat(np.nan, train_x.shape[0])

    # 学習データを分割
    kf = KFold(n_splits=4, shuffle=True, random_state=72)
    for idx_1, idx_2 in kf.split(train_x):
        # out-of-foldで各カテゴリにおける目的変数の平均を計算.iloc[a]a行の値を指定.iloc[a, b]a行b列の値に指定
        #.groupby(c)はCのカテゴリをまとめて平均を出す
        target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        # 変換後の値を一時配列に格納
        tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

    # 変換後のデータで元の変数を置換
    train_x[c] = tmp
    # 必要に応じてencodeされた特徴量を保存し、あとで読み込めるようにしておく
    
print(train_x)
print(test_x)
# target encoding - クロスバリデーションのfoldごとの場合
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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

# クロスバリデーションのfoldごとにtarget encodingをやり直す
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):

    # 学習データからバリデーションデータを分ける
    tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy()
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 変数をループしてtarget encoding
    for c in label_cols:
        # 学習データ全体で各カテゴリにおけるtargetの平均を計算
        data_tmp = pd.DataFrame({c: tr_x[c], 'target': tr_y})
        target_mean = data_tmp.groupby(c)['target'].mean()
        # バリデーションデータのカテゴリを置換
        va_x.loc[:, c] = va_x[c].map(target_mean)

        # 学習データの変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, tr_x.shape[0])
        kf_encoding = KFold(n_splits=4, shuffle=True, random_state=72)
        for idx_1, idx_2 in kf_encoding.split(tr_x):
            # out-of-foldで各カテゴリにおける目的変数の平均を計算
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            # 変換後の値を一時配列に格納
            tmp[idx_2] = tr_x[c].iloc[idx_2].map(target_mean)

        tr_x.loc[:, c] = tmp
        # 必要に応じてencodeされた特徴量を保存し、あとで読み込めるようにしておく
    
print(tr_x)
print()
print(va_x)
print()
print(tr_y)
print()
print(va_y)
