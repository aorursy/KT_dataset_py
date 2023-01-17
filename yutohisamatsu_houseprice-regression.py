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
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats # QQ-plot

from scipy.stats import norm # 正規化
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.head()
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
train.shape
test.head()
test.shape
submission.head()
submission.shape
# 目的変数の大まかな統計量

train['SalePrice'].describe()
# 目的変数の現分布をプロットする

plt.figure(figsize=(18,8))



# distplot

plt.subplot(221);

sns.distplot(train['SalePrice'], fit=norm);



plt.legend(['Normal dist. \n skewness: {:.2f} \n kurtosis: {:.2f}'.format(train['SalePrice'].skew(), train['SalePrice'].kurt())], loc='best');

plt.ylabel('Frequency'); # 頻度

plt.title('SalePrice density distribution');



# QQ-plot

plt.subplot(222);

stats.probplot(train['SalePrice'], sparams=(2.5,), plot=plt, rvalue=True);

plt.title('SalePrice Probability Plot');
# 特徴量（変数）の数を知る ('Id', 'SalePrice'を含む)

len(train.columns)
# 数値変数

f_numeric = [f for f in train.columns if train.dtypes[f] != 'object']



f_numeric.remove('Id')

f_numeric.remove('SalePrice')



# カテゴリ変数

f_categorical = [f for f in train.columns if train.dtypes[f] == 'object']
print('数値変数： ' + str(len(f_numeric)) + ', カテゴリ変数: ' + str(len(f_categorical))) # 合計79
# 欠損値をみる

missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True, ascending=False)

plt.figure(figsize=(18,8))

missing.plot.bar();
# 全体のうち、欠損値は何割か

percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([missing, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# この段階(EDA中)では、「欠損値」を削除するか、補うかの判断ができないので、欠損値は欠損値のカテゴリを作ってボックスプロットで見てみる。



# 新カテゴリを作成し、そこに欠損値を補う



for c in f_categorical:

    # astypeで型をキャスト(object→category)

    train[c] = train[c].astype('category')

    

    if train[c].isnull().any(): # 列に一個以上、欠損値が含まれるならTrue

        # カテゴリ追加

        train[c] = train[c].cat.add_categories(['MISSING'])

        train[c] = train[c].fillna('MISSING') # 「MISSING」の文字列で埋める



# ボックスプロットを描画する関数

def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90) # X軸のラベル回転
# 横長のデータを、縦長に変換する。 カラムは、「SalePrice」 「variable(これまでのカラムたち)」 「value(それぞれのカラムの値)」

categorical_data = pd.melt(train, id_vars=['SalePrice'], value_vars=f_categorical);

categorical_data.head()
# 図化

# カテゴリ別にグラフ化

FacetGrid = sns.FacetGrid(categorical_data, col='variable', col_wrap=4, sharex=False, sharey=False, size=5);



# map、関数とリストを引数として実行

FacetGrid = FacetGrid.map(boxplot, 'value', 'SalePrice');
# 分散をp値の順番にソートした配列を返す関数。

def anova(frame):

    anova = pd.DataFrame()

    anova['feature'] = f_categorical

    pvals = [] # p値の配列

    

    for c in f_categorical:

        samples = [] # 各カテゴリ要素の値がappendされている

    

        for cls in frame[c].unique(): # 各カテゴリの要素 (MSZoningなら[RL, RM, C (all), FV, RH]それぞれ)

            

            # SalePrice_Valueは、カテゴリ内の各要素の値 (MSZoningなら、RLのvalues、RMのvalues...)

            SalePrice_Value = frame[frame[c] == cls]['SalePrice'].values

            samples.append(SalePrice_Value)

        

        # scipyのstats.f_oneway() 一元配置分散 [statistic= , pvalue= ]

        pval = stats.f_oneway(*samples).pvalue

        pvals.append(pval)

    

    # p値を格納　カテゴリ変数カラムごと

    anova['pval'] = pvals

    

    return anova.sort_values('pval')
# 分散によってソートされたカテゴリ変数の配列を取得

anova = anova(train)



# log=対数をとる p値(a['pval']の値)の対数をとっている　[1./]の意味は、マイナスにならないように、とった対数がプラスの値になるように(グラフで可視化しやすく)

anova['disparity'] = np.log(1./anova['pval'].values)



# グラフ化

plt.figure(figsize=(18,8))

sns.barplot(data=anova, x='feature', y='disparity')

x=plt.xticks(rotation=90)
anova
for c in f_numeric:

    # astypeで型をキャスト(object→category)

    print(train[c].dtype)



# float64 小数点

# int64 整数
# 中央値で欠損値を補う



for n in f_numeric:

    

    if train[n].isnull().any(): # 列に一個以上、欠損値が含まれるならTrue

        # カテゴリ追加

#         train[n] = train[n].cat.add_categories(['MISSING'])

        train[n] = train[n].fillna(train[n].median()) # 中央値で埋める



# 散布図を描画する関数

def regplot(x, y, **kwargs):

    sns.regplot(x=x, y=y, data=train[f_numeric])

    x=plt.xticks(rotation=90) # X軸のラベル回転
numeric_data = pd.melt(train, id_vars=['SalePrice'], value_vars=f_numeric);

numeric_data.head()
# 図化

# カテゴリ別にグラフ化

FacetGrid = sns.FacetGrid(numeric_data, col='variable', col_wrap=4, sharex=False, sharey=False, size=5);



# map、関数とリストを引数として実行

FacetGrid = FacetGrid.map(regplot, 'value', 'SalePrice');