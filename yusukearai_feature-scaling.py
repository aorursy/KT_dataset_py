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
#標準化(standardization)変数の平均を0に標準偏差を1に変換
#スケールが大きい変数ほど回帰係数は小さくなるので標準化を行わないと正則化がかかりにくくなる
#ニューラルネットについても変数同士のスケールの差が大きいままでは学習がうまく進まないことが多い
#標準化を適用すると2次元配列を返す
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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

#学習データに基づいて複数列の標準化を定義
scaler = StandardScaler()
scaler.fit(tr_x)

#変数後のデータで各列を置換
tr_x_std = scaler.transform(tr_x)
va_x_std = scaler.transform(va_x)

print(tr_x)
print(tr_x_std)
print(va_x)
print(va_x_std)
#標準化(standardization)変数の平均を0に標準偏差を1に変換
#学習データとバリデーションデータ(テストデータ)を結合して学習させる→データを増やせる
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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

#学習データに基づいて複数列の標準化を定義
scaler = StandardScaler()
scaler.fit(pd.concat([tr_x,va_x]))

#変数後のデータで各列を置換
tr_x_std = scaler.transform(tr_x)
va_x_std = scaler.transform(va_x)

print(tr_x)
print(tr_x_std)
print(va_x)
print(va_x_std)
#Min-Maxスケーリング
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

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

#学習データに基づいて複数列の標準化を定義
scaler = MinMaxScaler()
scaler.fit(tr_x)

#変数後のデータで各列を置換
tr_x_mm = scaler.transform(tr_x)
va_x_mm = scaler.transform(va_x)

print(tr_x)
print(tr_x_mm)
print(va_x)
print(va_x_mm)
#非線形変換
#変数の分布を変えることが可能
#一般的に変数の分布はあまり偏ってないほうがいい
#対数変換―負の値は絶対値に対数変換をかけた後に負の値を付加する
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

plt.hist(train_x['Fare'], bins=20)
plt.show()
#偏りがあるデータに対数変換する
#log(0)にならないようにlog(1+x)にする

train_x['Fare'] = np.log1p(train_x['Fare'])
plt.hist(train_x['Fare'], bins=20)
plt.show()
#非線形変換-Box-Cox変換
#変数の分布を変えることが可能
#一般的に変数の分布はあまり偏ってないほうがいい
#0が項目にある場合はbox-coxをあきらめる
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

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

plt.hist(train_x['Fare'], bins=20)
plt.show()

num_cols = ["Fare"]

# 正の値のみをとる変数を変換対象としてリストに格納する
#これは0でもエラーが出る：ValueError: The Box-Cox transformation can only be applied to strictly positive data
# なお、欠損値も含める場合は、(~(train_x[c] <= 0.0)).all() などとする必要があるので注意
pos_cols = [c for c in num_cols if (train_x[c] > 0.0).all() and (test_x[c] > 0.0).all()]

# 学習データに基づいて複数列のBox-Cox変換を定義
pt = PowerTransformer(method='box-cox')
pt.fit(train_x["Fare"].values.reshape(1, -1))

# 変換後のデータで各列を置換
train_x["Fare"] = pt.transform(train_x["Fare"].values)
test_x["Fare"] = pt.transform(test_x["Fare"].values)

plt.hist(train_x['Fare'], bins=20)
plt.show()


#非線形変換-Yeo-Johnson変換
#変数の分布を変えることが可能
#一般的に変数の分布はあまり偏ってないほうがいい
#Yeo-Johnson変換なら可能
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

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

plt.hist(train_x['Fare'], bins=20, label="train_x")
plt.hist(test_x['Fare'], bins=20, label="test_x")
plt.legend()
plt.show()


plt.show()

num_cols = ["Fare"]

# 学習データに基づいて複数列のYeo-Johnson変換を定義
pt = PowerTransformer(method='yeo-johnson')
pt.fit(train_x[num_cols])

# 変換後のデータで各列を置換
train_x[num_cols] = pt.transform(train_x[num_cols])
test_x[num_cols] = pt.transform(test_x[num_cols])

plt.hist(train_x['Fare'], bins=20, label="train_x")
plt.hist(test_x['Fare'], bins=20, label="test_x")
plt.legend()
plt.show()
#clipping
#上限と下限を設定して外れ値を排除できる
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

plt.hist(train_x['Fare'], bins=20, label="train_x")
plt.hist(test_x['Fare'], bins=20, label="test_x")
plt.legend()
plt.show()

#train_x.info()
#test_x.info()

#変換するlabel変数のリスト
label_cols = ["Fare"]

# 列ごとに学習データの1％点、99％点を計算
p01 = train_x[label_cols].quantile(0.01)
p99 = train_x[label_cols].quantile(0.99)

# 1％点以下の値は1％点に、99％点以上の値は99％点にclippingする
train_x[label_cols] = train_x[label_cols].clip(p01, p99, axis=1)
test_x[label_cols] = test_x[label_cols].clip(p01, p99, axis=1)

plt.hist(train_x['Fare'], bins=20, label="train_x")
plt.hist(test_x['Fare'], bins=20, label="test_x")
plt.legend()
plt.show()
#binning
#数値変数を区間ごとにグループ分け
#binningしてからonehot-encodingなどを適合することもできる
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

plt.hist(train_x['Fare'], bins=20, label="train_x")
plt.hist(test_x['Fare'], bins=20, label="test_x")
plt.legend()
plt.show()

#train_x.info()
#test_x.info()

#変換するlabel変数のリスト
label_cols = ["Fare"]

#pandasのcut関数でbinningを行う
#該当する配列、binの数を指定
binned_tr_x = pd.cut(train_x["Fare"], 10, labels=False)
binned_ts_x = pd.cut(test_x["Fare"], 10, labels=False)

print(binned_tr_x)

plt.hist(binned_tr_x, bins=20, label="train_x")
plt.hist(binned_ts_x, bins=20, label="test_x")
plt.legend()
plt.show()

# binの範囲を指定する場合（50.0以下、50.0より大きく150.0以下、150.0より大きい）
bin_edges = [-float('inf'), 50.0, 150.0, float('inf')]
binned_ed_tr_x = pd.cut(train_x["Fare"], bin_edges, labels=False)
binned_ed_ts_x = pd.cut(test_x["Fare"], bin_edges, labels=False)

print(binned)

plt.hist(binned_ed_tr_x, bins=20, label="train_x")
plt.hist(binned_ed_ts_x, bins=20, label="test_x")
plt.legend()
plt.show()


# 順位への変換
# pandasのrank関数で順位に変換する
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

#print(test_x['Fare'])

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

#c1,c2,c3,c4 = "blue","green","red"
#l1,l2,l3,l4 = "sin","cos","abs(sin)"

ax1.hist(train_x['Fare'], bins=20, label="train_x")
ax1.hist(test_x['Fare'], bins=20, label="test_x")
ax1.legend()

rank_tr = train_x['Fare'].rank()
rank_ts = test_x['Fare'].rank()
print(rank.values)

ax2.hist(rank_tr, bins=20, label="train_x")
ax2.hist(rank_ts, bins=20, label="test_x")
ax2.legend()


# numpyのargsort関数を2回適用する方法で順位に変換する
order_tr = np.argsort(train_x['Fare'])
order_ts = np.argsort(test_x['Fare'])
rank_ar_tr = np.argsort(order_tr)
rank_ar_ts = np.argsort(order_ts)

print(rank)

ax3.hist(rank_ar_tr, bins=20, label="train_x")
ax3.hist(rank_ar_ts, bins=20, label="test_x")
ax3.legend()

fig.tight_layout() 
plt.show()
#RankGauss
#数値変数を順位に変換した後順序を保ったまま、無理やりに正規分布化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

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

plt.hist(train_x['Fare'], bins=20, label="train_x")
plt.hist(test_x['Fare'], bins=20, label="test_x")
plt.legend()
plt.show()

num_cols = ["Fare"]

# 学習データに基づいて複数列のRankGaussによる変換を定義
transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
transformer.fit(train_x[num_cols])

# 変換後のデータで各列を置換
train_x[num_cols] = transformer.transform(train_x[num_cols])
test_x[num_cols] = transformer.transform(test_x[num_cols])

plt.hist(train_x["Fare"], bins=20, label="train_x")
plt.hist(test_x["Fare"], bins=20, label="test_x")
plt.legend()
plt.show()
