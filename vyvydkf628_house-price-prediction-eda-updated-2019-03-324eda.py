# Loading packages

import pandas as pd #Analysis 

import matplotlib.pyplot as plt #Visulization

import seaborn as sns #Visulization

import numpy as np #Analysis 

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler #Analysis 

from scipy import stats #Analysis 

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import gc
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
print("train.csv. Shape: ",df_train.shape)

print("test.csv. Shape: ",df_test.shape)
df_train.head()
#descriptive statistics summary

df_train['price'].describe()
#histogram

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['price'])
#skewness and kurtosis

print("Skewness: %f" % df_train['price'].skew())

print("Kurtosis: %f" % df_train['price'].kurt())
fig = plt.figure(figsize = (15,10))



fig.add_subplot(1,2,1)

res = stats.probplot(df_train['price'], plot=plt)



fig.add_subplot(1,2,2)

res = stats.probplot(np.log1p(df_train['price']), plot=plt)
df_train['price'] = np.log1p(df_train['price'])

#histogram

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['price'])
# correlation이 높은 상위 10개의 heatmap

# continuous + sequential variables --> spearman

# abs는 반비례관계도 고려하기 위함

# https://www.kaggle.com/junoindatascience/let-s-eda-it 준호님이 수정해 준 코드로 사용하였습니다. 

import scipy as sp



cor_abs = abs(df_train.corr(method='spearman')) 

cor_cols = cor_abs.nlargest(n=10, columns='price').index # price과 correlation이 높은 column 10개 뽑기(내림차순)

# spearman coefficient matrix

cor = np.array(sp.stats.spearmanr(df_train[cor_cols].values))[0] # 10 x 10

print(cor_cols.values)

plt.figure(figsize=(10,10))

sns.set(font_scale=1.25)

sns.heatmap(cor, fmt='.2f', annot=True, square=True , annot_kws={'size' : 8} ,xticklabels=cor_cols.values, yticklabels=cor_cols.values)
data = pd.concat([df_train['price'], df_train['grade']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='grade', y="price", data=data)
data = pd.concat([df_train['price'], df_train['sqft_living']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_living', y="price", data=data)
data = pd.concat([df_train['price'], df_train['sqft_living15']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_living15', y="price", data=data)
data = pd.concat([df_train['price'], df_train['sqft_above']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_above', y="price", data=data)
data = pd.concat([df_train['price'], df_train['bathrooms']], axis=1)

f, ax = plt.subplots(figsize=(18, 6))

fig = sns.boxplot(x='bathrooms', y="price", data=data)
data = pd.concat([df_train['price'], df_train['bedrooms']], axis=1)

f, ax = plt.subplots(figsize=(18, 6))

fig = sns.boxplot(x='bedrooms', y="price", data=data)
from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



import plotly.graph_objs as go



import time

import random



#https://www.kaggle.com/ashishpatel26/bird-eye-view-of-two-sigma-nn-approach

def mis_value_graph(data):  

    data = [

    go.Bar(

        x = data.columns,

        y = data.isnull().sum(),

        name = 'Counts of Missing value',

        textfont=dict(size=20),

        marker=dict(

        line=dict(

            color= generate_color(),

            #width= 2,

        ), opacity = 0.45

    )

    ),

    ]

    layout= go.Layout(

        title= '"Total Missing Value By Column"',

        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),

        yaxis= dict(title='Value Count', ticklen=5, gridwidth=2),

        showlegend=True

    )

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='skin')

    

def generate_color():

    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))

    return color



df_all = pd.concat([df_train,df_test])

del df_all['price']

mis_value_graph(df_all)
### 유니크 갯수 계산

train_unique = []

columns = ['bedrooms','bathrooms','floors','waterfront','view','condition','grade']



for i in columns:

    train_unique.append(len(df_train[i].unique()))

unique_train = pd.DataFrame()

unique_train['Columns'] = columns

unique_train['Unique_value'] = train_unique



data = [

    go.Bar(

        x = unique_train['Columns'],

        y = unique_train['Unique_value'],

        name = 'Unique value in features',

        textfont=dict(size=20),

        marker=dict(

        line=dict(

            color= generate_color(),

            #width= 2,

        ), opacity = 0.45

    )

    ),

    ]

layout= go.Layout(

        title= "Unique Value By Column",

        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),

        yaxis= dict(title='Value Count', ticklen=5, gridwidth=2),

        showlegend=True

    )

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='skin')
df_train['floors'].unique()
data = pd.concat([df_train['price'], df_train['sqft_living']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_living', y="price", data=data)
df_train.loc[df_train['sqft_living'] > 13000]
df_train = df_train.loc[df_train['id']!=8990]
data = pd.concat([df_train['price'], df_train['grade']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='grade', y="price", data=data)
df_train.loc[(df_train['price']>12) & (df_train['grade'] == 3)]
df_train.loc[(df_train['price']>14.7) & (df_train['grade'] == 8)]
df_train.loc[(df_train['price']>15.5) & (df_train['grade'] == 11)]
df_train = df_train.loc[df_train['id']!=456]

df_train = df_train.loc[df_train['id']!=2302]

df_train = df_train.loc[df_train['id']!=4123]

df_train = df_train.loc[df_train['id']!=7259]

df_train = df_train.loc[df_train['id']!=2777]
data = pd.concat([df_train['price'], df_train['bedrooms']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='bedrooms', y="price", data=data)
skew_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for c in skew_columns:

    df_train[c] = np.log1p(df_train[c].values)

    df_test[c] = np.log1p(df_test[c].values)
for df in [df_train,df_test]:

    df['date'] = df['date'].apply(lambda x: x[0:8])

    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])
df_train.head()
for df in [df_train,df_test]:

    # 방의 전체 갯수 

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    

    # 거실의 비율 

    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

    

    df['sqft_total_size'] = df['sqft_above'] + df['sqft_basement']

    

    # 면적 대비 거실의 비율 

    df['sqft_ratio_1'] = df['sqft_living'] / df['sqft_total_size']

    

    df['sqft_ratio15'] = df['sqft_living15'] / df['sqft_lot15'] 

    

    # 재건축 여부 

    df['is_renovated'] = df['yr_renovated'] - df['yr_built']

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)

    df['date'] = df['date'].astype('int')
df_train['per_price'] = df_train['price']/df_train['sqft_total_size']

zipcode_price = df_train.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()

df_train = pd.merge(df_train,zipcode_price,how='left',on='zipcode')

df_test = pd.merge(df_test,zipcode_price,how='left',on='zipcode')



for df in [df_train,df_test]:

    df['zipcode_mean'] = df['mean'] * df['sqft_total_size']

    df['zipcode_var'] = df['var'] * df['sqft_total_size']

    del df['mean']; del df['var']
df_train.head()




train_columns = [c for c in df_train.columns if c not in ['id','price','per_price']]





from sklearn.model_selection import train_test_split

df_train1, df_train2 = train_test_split(df_train, train_size = 0.8, random_state=3)



from sklearn.ensemble import RandomForestRegressor

B = RandomForestRegressor(n_estimators=28,random_state=0)

B.fit(df_train1[train_columns], df_train1['price'])

score = B.score(df_train2[train_columns], df_train2['price'])

print(format(score,'.3f'))