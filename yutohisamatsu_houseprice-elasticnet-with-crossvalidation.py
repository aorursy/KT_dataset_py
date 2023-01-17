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
pd.options.display.max_rows = 1000

pd.options.display.max_columns = 20
import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew

from sklearn.model_selection import train_test_split, KFold

from sklearn.linear_model import ElasticNet

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_squared_error
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Select only confirmed features

# See https://www.kaggle.com/jimthompson/house-prices-advanced-regression-techniques/boruta-feature-importance-analysis



features = ['GrLivArea','OverallQual','2ndFlrSF','TotalBsmtSF','1stFlrSF','GarageCars','YearBuilt', 'GarageArea','ExterQual',

            'YearRemodAdd','FireplaceQu','GarageYrBlt','FullBath','MSSubClass','LotArea','Fireplaces','KitchenQual','MSZoning',

            'GarageType','BsmtFinSF1','Neighborhood','BsmtQual','TotRmsAbvGrd','HalfBath','BldgType','GarageFinish',

            'Foundation','BedroomAbvGr','HouseStyle','CentralAir','OpenPorchSF','HeatingQC','BsmtFinType1','BsmtUnfSF',

            'GarageCond','GarageQual','KitchenAbvGr','OverallCond','BsmtCond','MasVnrArea','BsmtFullBath','Exterior1st',

            'Exterior2nd','PavedDrive','WoodDeckSF','LandContour','MasVnrType','BsmtFinType2','Functional','Fence']
train = train[features+['SalePrice']]

test = test[['Id']+features]



all_data = pd.concat((train.loc[:,'GrLivArea':'Fence'],

                      test.loc[:,'GrLivArea':'Fence']))

all_data['SalePrice'] = train['SalePrice']
all_data.shape
obj_list = []

for i in all_data.columns :

    if all_data[i].dtypes == 'object':

        obj_list.append(i)
# カテゴリ変数の状況

all_data[obj_list].info()
# 数値変数の状況

plt.figure(figsize=(16, 10));

train.corr()['SalePrice'].sort_values().plot(kind='barh');
# 相関0.4以下

# f_drop_num = [

#     'WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea','BsmtFullBath','BsmtUnfSF','BedroomAbvGr','OverallCond','MSSubClass','KitchenAbvGr'

# ]



# 相関0.2以下

f_drop_num = [

    'BedroomAbvGr','OverallCond','MSSubClass','KitchenAbvGr'

]
basement = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF','BsmtFinType2', "BsmtFullBath"]

print(*basement, sep=", ")
# Basement - 数値変数



def corr_heatmap(columns=None, saleprice=['SalePrice'], df=all_data

                 , figsize=(8,6), vmin=-1, vmax=1, showvalue=True):

    columns = df.columns if columns == None else columns + saleprice

    corr = df[columns].corr();

    plt.figure(figsize=figsize);

    return sns.heatmap(corr, vmin=vmin, vmax=vmax, annot=showvalue, cmap='coolwarm');



corr_heatmap(basement);
def pairplot(columns, include_sale=True, data=all_data, kwargs={}):

    if include_sale & ("SalePrice" not in columns):

        columns = columns + ["SalePrice"]

    sns.pairplot(data=data[columns], **kwargs)

    

pairplot(basement, kwargs={"markers":"+", "height":1.25});
# Basement - カテゴリ変数

print(*[c for c in train.select_dtypes(include=['object']).columns if c in basement], sep=", ")



basement_categorical = [c for c in train.select_dtypes(include=['object']).columns if c in basement]
# カテゴリ変数

f_categorical = [f for f in all_data.columns if all_data.dtypes[f] == 'object']



# 新カテゴリを作成し、そこに欠損値を補う



for c in f_categorical:

    # astypeで型をキャスト(object→category)

    all_data[c] = all_data[c].astype('category')

    

    if all_data[c].isnull().any(): # 列に一個以上、欠損値が含まれるならTrue

        # カテゴリ追加

        all_data[c] = all_data[c].cat.add_categories(['MISSING'])

        all_data[c] = all_data[c].fillna('MISSING') # 「MISSING」の文字列で埋める
# カテゴリ変数用　欠損値を「MISSING」で埋める

def make_MISSING(data):

    # カテゴリ変数

    f_categorical = [f for f in data.columns if data.dtypes[f] == 'object']



    # 新カテゴリを作成し、そこに欠損値を補う



    for c in f_categorical:

        # astypeで型をキャスト(object→category)

        data[c] = data[c].astype('category')

    

    if data[c].isnull().any(): # 列に一個以上、欠損値が含まれるならTrue

        # カテゴリ追加

        data[c] = data[c].cat.add_categories(['MISSING'])

        data[c] = data[c].fillna('MISSING') # 「MISSING」の文字列で埋める
# ボックスプロットを描画する関数

def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y);

    x=plt.xticks(rotation=90) # X軸のラベル回転



    

# バイオリンプロットを描画する関数

def violinplot(x, y, **kwargs):

    sns.violinplot(x=x, y=y);

    x=plt.xticks(rotation=90) # X軸のラベル回転

    

categorical_data = pd.melt(all_data, id_vars=['SalePrice'], value_vars=basement_categorical);

    

# カテゴリ別にグラフ化

FacetGrid = sns.FacetGrid(categorical_data, col='variable', col_wrap=4, sharex=False, sharey=False, size=5);



# map、関数とリストを引数として実行

FacetGrid = FacetGrid.map(boxplot, 'value', 'SalePrice');
# 関数化

def make_FacetGrid(data, value_vars, plotType):

    select_data = pd.melt(data, id_vars=['SalePrice'], value_vars=value_vars);

    # カテゴリ別にグラフ化

    FacetGrid = sns.FacetGrid(select_data, col='variable', col_wrap=4, sharex=False, sharey=False, size=5);



    # map、関数とリストを引数として実行

    FacetGrid = FacetGrid.map(plotType, 'value', 'SalePrice');
make_FacetGrid(all_data, basement_categorical, violinplot);
BsmtQual_map = {"Ex":3, "Gd":2, "TA":2, "Fa":1, "MISSING":1}

BsmtFinType1_map = {"GLQ":2, "ALQ":2, "Unf":2,"Rec":1, "LwQ":1, "BLQ":1, "MISSING":1}

BsmtCond_map = {"Gd":3, "TA":3, "MISSING":2, "Fa":2, "Po":1}

BsmtFinType2_map = {"Unf":4, "ALQ":4, "MISSING":3, "Rec":3, "BLQ":2, "LwQ":2, "GLQ":1 }



all_data["BsmtQual"] = all_data.BsmtQual.replace(BsmtQual_map)

all_data["BsmtFinType1"] = all_data.BsmtFinType1.replace(BsmtFinType1_map)

all_data["BsmtCond"] = all_data.BsmtCond.replace(BsmtCond_map)

all_data["BsmtFinType2"] = all_data.BsmtFinType2.replace(BsmtFinType2_map)
all_data['BsmtFinType2'].unique()
all_data["BsmtQual"]
make_FacetGrid(all_data, ["BsmtQual", "BsmtFinType1", "BsmtCond", "BsmtFinType2"], violinplot);
make_FacetGrid(all_data, f_categorical, violinplot);
pd.options.display.max_rows = 1000

pd.options.display.max_columns = 25
all_data[f_categorical].head(2)
all_data.MasVnrType.unique()
# 残りのカテゴリ変数を、ラベルエンコーディング

ExterQual_map = {"Gd":2, "TA":2, "Ex":3, "Fa":1}

FireplaceQu_map = {"MISSING":2, "TA":3, "Gd":3,"Fa":1, "Ex":4, "Po":1}

KitchenQual_map = {"Gd":1, "TA":2, "MISSING":5, "Fa":4, "Ex":3} # MISSING:5のデータ行を削除する

MSZoning_map = {"RL":1, "RM":1, "C（all）":2, "FV":3, "RH":3, "MISSING":4}



GarageType_map = {"Attchd":1, "Detchd":1, "BuiltIn":2, "CarPort":3, "MISSING":1, "Basment":4, "2Types":4}

Neighborhood_map = {} # 25個ある

BldgType_map = {"1Fam":1, "2fmCon":2, "Duplex":2, "TwnhsE":1, "Twnhs":2}

GarageFinish_map = {"RFn":1, "Unf":2, "Fin":1,"MISSING":2}

Foundation_map = {"PConc":1, "CBlock":1, "BrkTil":2, "Wood":3, "Slab":2, "Stone":3}

HouseStyle_map = {"2Story":1, "1Story":1, "1.5Fin":2,"1.5Unf":3, "SFoyer":4, "SLvl":2, "2.5Unf":4, "2.5Fin":5}



CentralAir_map = {"Y":1, "N":2}

HeatingQC_map = {"Ex":1, "Gd":1, "TA":1,"Fa":2, "Po":3}

GarageCond_map = {"TA":1, "Fa":2, "MISSING":1,"Gd":2, "Po":2, "Ex":3}

GarageQual_map = {"TA":1, "Fa":2, "Gd":3,"MISSING":1, "Ex":4, "Po":5}

Exterior1st_map = {} # 16個ある

Exterior2nd_map = {} # 17個ある

PavedDrive_map = {"Y":1, "N":2, "P":2}

LandContour_map = {"Lvl":1, "Bnk":2, "Low":3,"HLS":1}

MasVnrType_map = {"BrkFace":1, "None":1, "Stone":3,"BrkCmn":2, "MISSING":3}



Functional_map = {"Typ":1, "Min1":4, "Maj1":2,"Min2":1, "Mod":1, "Maj2":2, "Sev":3, "MISSING":2} #  カラム自体削除してもいいかも

Fence_map = {"MISSING":1, "MnPrv":1, "GdWo":2,"GdPrv":2, "MnWw":3}





all_data["ExterQual"] = all_data.ExterQual.replace(ExterQual_map)

all_data["FireplaceQu"] = all_data.FireplaceQu.replace(FireplaceQu_map)

all_data["KitchenQual"] = all_data.KitchenQual.replace(KitchenQual_map)

all_data["MSZoning"] = all_data.MSZoning.replace(MSZoning_map)





all_data["GarageType"] = all_data.GarageType.replace(GarageType_map)

all_data["BldgType"] = all_data.BldgType.replace(BldgType_map)

all_data["GarageFinish"] = all_data.GarageFinish.replace(GarageFinish_map)

all_data["Foundation"] = all_data.Foundation.replace(Foundation_map)

all_data["HouseStyle"] = all_data.HouseStyle.replace(HouseStyle_map)





all_data["CentralAir"] = all_data.CentralAir.replace(CentralAir_map)

all_data["HeatingQC"] = all_data.HeatingQC.replace(HeatingQC_map)

all_data["GarageCond"] = all_data.GarageCond.replace(GarageCond_map)

all_data["GarageQual"] = all_data.GarageQual.replace(GarageQual_map)

all_data["PavedDrive"] = all_data.PavedDrive.replace(PavedDrive_map)

all_data["LandContour"] = all_data.LandContour.replace(LandContour_map)

all_data["MasVnrType"] = all_data.MasVnrType.replace(MasVnrType_map)





all_data["Functional"] = all_data.Functional.replace(Functional_map)

all_data["Fence"] = all_data.Fence.replace(Fence_map)
all_data
# Neighborhood_map = {} # 25個ある

plt.figure(figsize=(16, 10));

sns.violinplot(x=all_data['Neighborhood'], y=all_data['SalePrice']);

x=plt.xticks(rotation=90) # X軸のラベル回転
Neighborhood_map = {

    'CollgCr':6, 'Veenker':4, 'Crawfor':6, 'NoRidge':1, 'Mitchel':5,

    'Blmngtn':5, 'Blueste':5, 'BrDale':1, 'ClearCr':7, 'Edwards':2,

    'Gilbert':5, 'IDOTRR':5, 'MeadowV':5, 'NAmes':5, 'NPkVill':5,

    'NWAmes':0, 'NridgHt':4, 'OldTown':0, 'SWISU':5, 'Sawyer':0,

    'SawyerW':4, 'Somerst':0, 'StoneBr':3, 'Timber':4, 'BrkSide':5,



} # 25個ある

all_data["Neighborhood"] = all_data.Neighborhood.replace(Neighborhood_map)
# Exterior1st_map = {} # 16個ある

plt.figure(figsize=(16, 10));

sns.violinplot(x=all_data['Exterior1st'], y=all_data['SalePrice']);

x=plt.xticks(rotation=90) # X軸のラベル回転
Exterior1st_map = {

    'AsbShng':7, 'AsphShn':6, 'BrkComm':8, 'BrkFace':7, 'CBlock':6,

    'CemntBd':5, 'HdBoard':3, 'ImStucc':0, 'MetalSd':4, 'Plywood':3,

    'Stone':2, 'Stucco':4, 'VinylSd':3, 'Wd Sdng':3, 'WdShing':3,

    'MISSING':1,

} # 16個ある

all_data["Exterior1st"] = all_data.Exterior1st.replace(Exterior1st_map)
# Exterior2nd_map = {} # 17個ある

plt.figure(figsize=(16, 10));

sns.violinplot(x=all_data['Exterior2nd'], y=all_data['SalePrice']);

x=plt.xticks(rotation=90) # X軸のラベル回転
Exterior2nd_map = {

    'AsbShng':7, 'AsphShn':8, 'Brk Cmm':7, 'BrkFace':7, 'CBlock':6,

    'CmentBd':5, 'HdBoard':4, 'ImStucc':5, 'MetalSd':4, 'Other':0, 'Plywood':4,

    'Stone':3, 'Stucco':3, 'VinylSd':2, 'Wd Sdng':2, 'Wd Shng':2,

    'MISSING':1,

} # 17個ある

all_data["Exterior2nd"] = all_data.Exterior2nd.replace(Exterior2nd_map)
MISSING_KitchenQual = all_data.loc[all_data['KitchenQual'] == 5] # ラベルエンコーディング[MISSING]=5



MISSING_Exterior1st_0 = all_data.loc[all_data['Exterior1st'] == 0]

MISSING_Exterior1st_1 = all_data.loc[all_data['Exterior1st'] == 1]



MISSING_Exterior2nd_0 = all_data.loc[all_data['Exterior2nd'] == 0]

MISSING_Exterior2nd_1 = all_data.loc[all_data['Exterior2nd'] == 1]



MISSING_KitchenQual #95

MISSING_Exterior1st_0 #1187

MISSING_Exterior1st_1 # 691

MISSING_Exterior2nd_0 #595

MISSING_Exterior2nd_1 #691
all_data.shape
# 削除するレコード　カテゴリ変数のラベルエンコーディング後

all_data.drop([95, 1187, 691, 595, 691], inplace=True)

test['Id'].drop([95, 1187, 691, 595, 691], inplace=True) # テストの数も合わせる
all_data.shape
# log transform the target:

train['SalePrice'] = np.log1p(train['SalePrice'])



# log transform skewed numeric features: 歪んだ数値変数

numeric_feats = train.dtypes[train.dtypes != "object"].index



# 歪度計算

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))



skewed_feats = skewed_feats[skewed_feats > 0.75] # 絞る

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
#impute NA's, numerical features with the median, categorical values with most common value:



# objectを除外 = 数値変数→欠損を中央値で

for feature in all_data.select_dtypes(exclude=['object', 'category']).columns:

        all_data[feature].fillna(all_data[feature].median(), inplace = True)

        

# # objectを抽出 = カテゴリ変数→欠損を最頻値で

# for feature in all_data.select_dtypes(include=['object']).columns: 

#         all_data[feature].fillna(all_data[feature].value_counts().idxmax(), inplace = True)



# objectを抽出 = カテゴリ変数→make_MISSINGで新しい特徴量[MISSING]で埋める

# for feature in all_data.select_dtypes(include=['object']).columns: 

#         make_MISSING(all_data)
all_data
all_data.drop('SalePrice', axis=1, inplace=True)

# all_data.drop(f_drop_num, axis=1, inplace=True) # yとの相関0.○以下の数値変数を削除
all_data
all_data = pd.get_dummies(all_data)

print(all_data.shape)
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
train_x, test_x, train_y, test_y = train_test_split(X_train, y, test_size=0.3, random_state=0)
# Define Model

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=3))
ENet.fit(train_x, train_y)

ENet.score(test_x,test_y)*100
# モデル学習

ENet.fit(X_train.values, y)

ENet.score(test_x,test_y)*100
def predict_cv(model, train_x, train_y, test_x):

    preds = []

    preds_test = []

    va_indexes = []

    kf = KFold(n_splits=4, shuffle=True, random_state=6785)

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する

    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)): # enumerate 配列要素とインデックスも同時に取得する

        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]

        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        model.fit(tr_x, tr_y)

        tr_pred = model.predict(tr_x)

        pred = model.predict(va_x)

        preds.append(pred)

        pred_test = model.predict(test_x)

        preds_test.append(pred_test)

        va_indexes.append(va_idx)

        print('  score Train : {:.6f}' .format(np.sqrt(mean_squared_error(tr_y, tr_pred))), 

              '  score Valid : {:.6f}' .format(np.sqrt(mean_squared_error(va_y, pred))),

              '  Correct answer rate Train: {:.6f}' .format(model.score(tr_x, tr_y)),

              '  Correct answer rate Valid: {:.6f}' .format(model.score(va_x, va_y)),

             ) 

    # バリデーションデータに対する予測値を連結し、その後元の順番に並べなおす

    va_indexes = np.concatenate(va_indexes)

    preds = np.concatenate(preds, axis=0)

    order = np.argsort(va_indexes)

    pred_train = pd.DataFrame(preds[order])

    # テストデータに対する予測値の平均をとる

    preds_test = pd.DataFrame(np.mean(preds_test, axis=0))

    print('Score : {:.6f}' .format(np.sqrt(mean_squared_error(train_y, pred_train))))

    return pred_train, preds_test, model
# Cross Validation

predict_cv(ENet, train_x, train_y, test_x)
# モデル学習

ENet.fit(X_train.values, y)

ENet.score(test_x,test_y)*100
# テストデータ 推測

ENet_pred = np.expm1(ENet.predict(X_test.values))
sub = pd.DataFrame()

sub['Id'] = test['Id']

sub['SalePrice'] = ENet_pred

sub.to_csv('submission.csv',index=False)