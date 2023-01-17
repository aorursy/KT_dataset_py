import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
housing_df = pd.read_csv('../input/kaggle/train.csv')

housing_df.info()

#共有10个属性，总数据量为16512 

#除total_bedrooms有缺失外其他都是完整的 

#ocean_proximity是object型，其他都是float64
#ocean_proximity各个取值的分布

housing_df['ocean_proximity'].value_counts()
#各属性除去id的描述性统计

housing_df.drop(['id'],axis=1,inplace=True)

housing_df.info()

housing_df.describe()
#各属性的数值分布

housing_df.hist(bins=50, figsize=(20,15))
#各样本点的位置分布，颜色越深样本点数量越多

#可以看出大致是加州的形状,但取值有明显的几个聚集

housing_df.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)

plt.show()
#房价在地理分布上的可视化数据 颜色趋近蓝色代表价格低 趋近红色代表价格高

#圆圈半径代表该地区population的大小

#可以猜测房价与离海的距离有密切关系（对照加州地图）

housing_df.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,

             s=housing_df["population"]/100,label="population",

             c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True,sharex=False)

plt.legend()

plt.show()
#使用pearson相关查看相关性矩阵

corr = housing_df.corr(method='pearson')

corr
#热力图 可以更直观地看出各属性之间的相关性

import seaborn as sns

k = 10 

corrmat = housing_df.corr()

cols = corrmat.nlargest(k, 'median_house_value')['median_house_value'].index

cm = np.corrcoef(housing_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,cmap='YlGnBu')

plt.show()
# 只查看各属性与房价的相关系数

#可以看出房价与income存在比较强的正相关

corr['median_house_value'].sort_values(ascending=False)
#组合字段

tmp_df = housing_df.copy()

tmp_df['population_per_household'] = tmp_df['population'] / tmp_df['households']

tmp_df['rooms_per_household'] = tmp_df['total_rooms'] / tmp_df['households']

tmp_df['bedrooms_per_room'] = tmp_df['total_bedrooms'] / tmp_df['total_rooms']

tmp_df.head(10)
#查看组合字段后的相关系数 

#bedrooms_per_room有相对大的负相关 

#population_per_household有相对非常小的负相关

corr = tmp_df.corr(method='pearson')

corr['median_house_value'].sort_values(ascending=False)
#使用imputer处理缺失值示例

from sklearn.impute import SimpleImputer

tmp_df = housing_df.copy()

imputer = SimpleImputer(strategy='median')

tmp_df['total_bedrooms'] = imputer.fit_transform(tmp_df[['total_bedrooms']])

tmp_df.info()
#处理文本信息进行编码

from sklearn.preprocessing import LabelBinarizer

encode = LabelBinarizer()

encode.fit_transform(housing_df['ocean_proximity'])
#数据转化流水线

#转化为numpy数组并输出

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names=attribute_names

    def fit(self, x, y=None):

        return self

    def transform(self, x):

        return x[self.attribute_names].values
def get_columns_index(df, columns):

    return [list(df.columns).index(column) for column in list(columns)]
#补充组合字段函数

def add_extra_features(x, rooms_ix, bedrooms_ix, population_ix, household_ix):    

    population_per_household = x[:, population_ix] / x[:, household_ix]

    rooms_per_household = x[:, rooms_ix] / x[:, household_ix]

    bedrooms_per_room = x[:, bedrooms_ix] / x[:, rooms_ix]

    return np.c_[x, population_per_household, rooms_per_household, bedrooms_per_room]
# 转化流水线  使用pipeline形成机器学习工作流 

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder



# 数值类型流水线

num_attribute_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',

              'households', 'median_income']

rooms_ix, bedrooms_ix, population_ix, household_ix = get_columns_index(housing_df, ['total_rooms', 'total_bedrooms', 'population', 'households'])

num_pipline = Pipeline([

    ('selector', DataFrameSelector(num_attribute_names)),#numpy数组

    ('imputer', SimpleImputer(strategy='median')),# 缺失值处理

    ('attribs_adder', FunctionTransformer(add_extra_features, kw_args={'rooms_ix':rooms_ix, 'bedrooms_ix':bedrooms_ix, 

                                                                       'population_ix':population_ix, 'household_ix':household_ix})),

    ('std_scaler', StandardScaler())#组合字段

])



# 文本类型流水线

text_pipline = Pipeline([

    ('selector', DataFrameSelector(['ocean_proximity'])),

    ('text_encoder', OneHotEncoder(sparse=False)),

])



# 合并

union_pipplines = FeatureUnion(transformer_list=[

    ('num_pipline', num_pipline),

    ('text_pipline', text_pipline),

])

#结果

housing_prepares = union_pipplines.fit_transform(housing_df)

print('shape: ', housing_prepares.shape)

print('data head 5: \n', housing_prepares[0:5, :])
#将数据集分为训练集和测试集

from sklearn.model_selection import train_test_split

housing_label = housing_df['median_house_value'].values

x_train, x_test, y_train, y_test = train_test_split(housing_prepares, housing_label)
#线性回归

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



estimator = LinearRegression()

estimator.fit(x_train, y_train)



estimator = GridSearchCV(estimator, param_grid={}, cv=None)

estimator.fit(x_train, y_train)



y_test_predict = estimator.predict(x_test)

print('R-square:',r2_score(y_test,y_test_predict))

print(u'估计器: \n', estimator.best_estimator_)

print(u'交叉验证结果: \n', estimator.cv_results_)
#决策树

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



estimator = DecisionTreeRegressor()

estimator.fit(x_train, y_train)



estimator = GridSearchCV(estimator, param_grid={}, cv=None)

estimator.fit(x_train, y_train)



y_test_predict = estimator.predict(x_test)



print('R-square:',r2_score(y_test,y_test_predict))

print(u'估计器: \n', estimator.best_estimator_)

print(u'交叉验证结果: \n', estimator.cv_results_)
#随机森林 相对好一点 采用随机森林

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score





estimator = RandomForestRegressor()

estimator.fit(x_train, y_train)



param_grid=[{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},

            {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},]

estimator = GridSearchCV(estimator, param_grid, cv=None)

estimator.fit(x_train, y_train)



y_test_predict = estimator.predict(x_test)



print('R-square:',r2_score(y_test,y_test_predict))

print(u'估计器: \n', estimator.best_estimator_)

print(u'交叉验证结果: \n', estimator.cv_results_)
test_df = pd.read_csv('../input/kaggle/test.csv')

test_df.drop(['id'],axis=1,inplace=True)

test_df.info()
#流水线处理数据

_num_attribute_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',

              'households', 'median_income']

_rooms_ix, _bedrooms_ix, _population_ix, _household_ix = get_columns_index(test_df, ['total_rooms', 'total_bedrooms', 'population', 'households'])

_num_pipline = Pipeline([

    ('selector', DataFrameSelector(_num_attribute_names)),

    ('imputer', SimpleImputer(strategy='median')),

    ('attribs_adder', FunctionTransformer(add_extra_features, kw_args={'rooms_ix':_rooms_ix, 'bedrooms_ix':_bedrooms_ix, 

                                                                       'population_ix':_population_ix, 'household_ix':_household_ix})),

    ('std_scaler', StandardScaler())

])





_text_pipline = Pipeline([

    ('selector', DataFrameSelector(['ocean_proximity'])),

    ('text_encoder', OneHotEncoder(sparse=False)),

])





_union_pipplines = FeatureUnion(transformer_list=[

    ('num_pipline', _num_pipline),

    ('text_pipline', _text_pipline),

])

_housing_prepares = _union_pipplines.fit_transform(test_df)



#随机森林预测

estimator = RandomForestRegressor()

estimator.fit(housing_prepares, housing_label)
param_grid=[{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},

            {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},]



estimator = GridSearchCV(estimator, param_grid, cv=None)

estimator.fit(housing_prepares, housing_label)
#模型相关内容

print("模型的最优参数：\n",estimator.best_params_)

print("最优模型分数：\n",estimator.best_score_)

print("最优模型对象：\n",estimator.best_estimator_)

#输出网格搜索每组超参数的cv数据

for p, s in zip(estimator.cv_results_['params'],estimator.cv_results_['mean_test_score']):

    print(p, s)
#预测

_y_test_predict = estimator.predict(_housing_prepares)
#写文件

data = pd.DataFrame(_y_test_predict)

data.to_csv('result.csv')