import pandas as pd
import numpy as np

columns = ['Name', 'English', 'Mathematics', 'Japanese']
data = [['kawasaki', 71, 84, 81 ],
             ['fujisawa', 70, 77, 73 ],
             ['kikukawa', 60, 68, 56 ],
             ['kanaya', np.NaN, 89, 72],
             ['takatsuka', 18, 93, 87 ],
             ['totsuka', 56, 73, np.NaN],
             ['okazaki', 86, 71, 94],
             ['ogaki', 99, 74, np.NaN],
             ['mishima', 79, 71, 86],
             ['oofuna', 50, 62, 53],
             ['oiso', 40, 71, 37],
             ['ninomiya', 68, 53, 49],
             ['hayakawa', 83, np.NaN, 39],
             ['toyohashi', 72, 15, 88],
            ]
train = pd.DataFrame(data, columns=columns)
## 概要確認
train.head(3)
## トレーニング、テストデータのデータ数確認
len(train.index)
## 値がnullの項目数を数える
train.isnull().sum()
# リストワイズ削除
train.dropna()
#ペアワイズ削除
train[train.English.notna() & train.Mathematics.notna()]
#平均値代入
train['Japanese'].fillna(train['Japanese'].mean())
# 平均値代入(Imputer)
from sklearn.preprocessing import Imputer
#Nameを除かないとエラーが出ます
new_train = train.drop(columns='Name')
imp = Imputer(missing_values='NaN', strategy='mean')
imp.fit(new_train)
new_train = pd.DataFrame(imp.transform(new_train), columns=new_train.columns)
new_train
reg_train = train.copy()

#線形回帰、英語のスコアから日本語のスコアを予想する。英語のスコアと日本語のスコアには相関があると仮定する
import sklearn.linear_model as lm
reg = lm.LinearRegression()

#英語と日本語のスコアが両方あるデータから回帰モデルを作成
indexer = reg_train['Japanese'].notna() & reg_train['English'].notna()
reg_data = reg_train.loc[indexer]
X = reg_data.loc[:, ['English']]
Y = reg_data['Japanese']
reg.fit(X,Y)

#英語のスコアがあるデータから日本語のスコアを埋める
indexer = reg_train['Japanese'].isnull() & reg_train['English'].notna()
reg_data = reg_train.loc[indexer]
X = reg_data.loc[:, ['English']]
if(len(X)!=0):
  predicted = reg.predict(X)
  reg_train.loc[indexer, 'Japanese'] = predicted

reg_train
#karnelなら必要ないが、ローカルで実行する場合は必要
#!pip install fancyimpute

# SimpleFill　： 平均値や中央値で置き換え
# KNN　：　k近傍法, 欠損値から近しいK個のデータを用いて決める
# SoftImpute　：　詳しくは原論文を参照
# IterativeSVD
# MICE
# MatrixFactorization
# NuclearNormMinimization
# BiScaler
#などが使えます

from fancyimpute import SimpleFill,KNN,SoftImpute,IterativeSVD,MICE,MatrixFactorization,NuclearNormMinimization,BiScaler
SimpleFill().complete(train.drop(columns='Name'))
KNN(k=3).complete(train.drop(columns='Name'))
#help(SoftImpute)
# Instead of solving the nuclear norm objective directly, instead
# induce sparsity using singular value thresholding
#入力は標準化する必要がある
biscaler = BiScaler()
X_normalized = biscaler.fit_transform(train.drop(columns='Name').values)
SoftImpute().complete(X_normalized)
#行数が十分多いときに使った方が良い
#今回は行数が４なので、rankを2にしている
#行数よりもrankが大きければエラーになる　デフォルトでは10

iterativeSVD_train = train.copy()
iterativeSVD_train['Science'] = [78,70,50,90,78,65,90,80,68,56,54,58,100,90]
iterativeSVD_train

IterativeSVD(rank=2).complete(iterativeSVD_train.drop(columns='Name'))
#連鎖方程式による多変量代入
#詳しくは原論文参照
MICE(n_imputations=200, impute_type='col', verbose=False).complete(train.drop(columns='Name'))
# 行列分解を用いた方法
MatrixFactorization().complete(train.drop(columns='Name'))
#The solver SCS is not installed.　や　ImportError: dlopen(...): Library not loaded: @rpath/libmkl_intel_lp64.dylib
#が出て来たらOpenBLASというものが必要と言われているが、まだ成功していない
#pipに慣れている人しかお勧めしない

#NuclearNormMinimization().complete(train.drop(columns='Name'))
#biScalerはデータのスケーリングをする。欠損値補完ではないため注意
#BiScaler is not a matrix completion algorithm. It's a pre-processing step. So it does have fit and fit_transform.
#help(BiScaler)

# simultaneously normalizes the rows and columns of your observed data,
# sometimes useful for low-rank imputation methods
biscaler = BiScaler()

#ここで一度正規化します。
X_normalized = biscaler.fit_transform(train.drop(columns='Name').values)
print(X_normalized)
#例えばsoftimputeを用いて欠損値補完します
softimpute_normalized = SoftImpute().complete(X_normalized)
print(softimpute_normalized)
#inverse_transformを使うことで正規化前のデータを再現することができます
softimpute = biscaler.inverse_transform(softimpute_normalized)
print(softimpute)
#reference
#http://machine-learning.hatenablog.com/entry/2017/08/30/221801
#http://norimune.net/1811
#https://qiita.com/siseru/items/7b4aa0d97be23439ada7
#https://qiita.com/siseru/items/7b4aa0d97be23439ada7
#http://jotkn.ciao.jp/wp/2017/08/22/post-76/
#https://qiita.com/nanairoGlasses/items/339ed9cb6297a1cb81bd
#http://sinhrks.hatenablog.com/entry/2016/02/01/080859#%E5%B9%B3%E5%9D%87%E4%BB%A3%E5%85%A5%E6%B3%95
#http://smrmkt.hatenablog.jp/entry/2013/01/14/141158
#https://stackoverflow.com/questions/45321406/missing-value-imputation-in-python-using-knn
#https://github.com/iskandr/fancyimpute
#https://www.kaggle.com/athi94/investigating-imputation-methods
