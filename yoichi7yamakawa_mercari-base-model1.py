# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

import seaborn as sns

from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression



%matplotlib inline

pd.set_option("display.max_colwidth", 2000)
train = pd.read_table('../input/train/train.tsv')

test =  pd.read_table('../input/train/test.tsv')
gr_train = train.groupby(['category_name']).count()

gr_train['volume_rate'] = (gr_train['train_id']* 100 / train['train_id'].count()).astype(float)

mass_categories = gr_train.sort_values(by ='train_id', ascending = False).head(20)

mass_categories = mass_categories.reset_index()



train_mass = pd.merge(train, mass_categories, on='category_name')
#category ごとに平均値を取る（分布が正規分布に近いかどうかも調べること）

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(15.7, 12.27)

violinplot = sns.violinplot(data = train_mass, x = 'category_name', y= 'price_x',inner="quart")

plt.xticks(rotation = -90)

axes = violinplot.axes

plt.ylim(-10, 100)



#平均ではなく中央値を取るのが良さそう。外れ値はcondition, shipping, brand_nameに依存していそう

# 実際に外れ値を見てみるとガチブランドばっかりだった　train_mass.sort_values(by = ['price_x'], ascending = False)
# category_name がnullのものはothersに分類

train.loc[train['category_name'].isnull(), 'category_name'] = 'others'

# category_name ごとの中央値を取るモデルの作成

base_price = train.groupby(['category_name']).median()['price']

base_price = base_price.reset_index()



train.loc[train['category_name'].isnull(), 'category_name'] = 'others'

test_base = pd.merge(test, base_price, on = 'category_name', how = 'left')

test_base.loc[test_base['price'].isnull(),'price'] = 15



test_base_submission = pd.DataFrame()

test_base_submission['test_id'] = test_base['test_id']

test_base_submission['price'] = test_base['price'].astype(int)

test_base_submission.to_csv('test_base_submission.csv',index=False)