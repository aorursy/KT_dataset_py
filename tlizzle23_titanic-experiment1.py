#!kaggle competitions download --force titanic

import pandas as pd

import numpy as np

import os

import sys

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

%matplotlib inline  


"""

dependent var = survived: 0/1 for no/yes

independent vars = 

    pcalss: ticket class 

    sex: 

    age: age in years

    sibsp: num of sibling aboard the titanic

    parch: num of parents aboard the titanic

    ticket: ticket number

    fare: passenger fare

    cabin: cabin number

    embarked: port of embarkation

"""
_data_path = '../input/titanic/'



_train_data = os.path.join(_data_path, 'train.csv') 

_test_data =  os.path.join(_data_path, 'test.csv') 



train_df = pd.read_csv(_train_data)

test_df = pd.read_csv(_test_data)

train_df.columns = train_df.columns.str.lower()

test_df.columns = test_df.columns.str.lower()



display(train_df.head(10))
print('data shape: {}'.format(train_df.shape))

# train_df

print('\ndata columns: {}'.format(list(train_df.columns)))

print(train_df.survived.value_counts() / len(train_df))



# # check no duplicates

train_df.groupby('survived').size().describe()



# # check for null values

print('whether having a null value: {}\n'.format(train_df.isnull().values.any()))

print(train_df.isnull().sum())

print('\n')

print(train_df.info())
sns.countplot(x= 'survived', data= train_df)

sns.catplot(x= "pclass", y= "fare", hue= 'survived', kind= 'box', data= train_df)

train_df[['pclass', 'survived']].groupby(['pclass', 'survived']).size().unstack().plot(kind = 'bar', stacked = True, title = 'Pclass distribution')
import re



combined_dfs = [train_df, test_df]

for df in combined_dfs:

    df['title'] = df['name'].apply(lambda x: re.search('[A-Za-z]+\.', x).group())

# train_df['title'].value_counts()

def encode_title(data):

    if data == 'Mr.':

        data = 0

    elif data == 'Miss.':

        data = 1

    elif data == 'Mrs.':

        data = 2

    else:

        data = 3

    return data



for df in combined_dfs:

    df['title'] = df['title'].apply(encode_title)



df['title'].value_counts()
train_df.groupby(['title', 'survived']).size().unstack().plot(kind = 'bar')



# delete unneeded variable

for df in combined_dfs:

    df.drop('name', axis= 1, inplace = True)



for df in combined_dfs:

    df['age'].fillna(df.groupby('title')['age'].transform('median'), inplace= True)



display(train_df.head())

display(test_df.head())



sns.FacetGrid(train_df, hue= 'survived', height=8).map(sns.kdeplot, 'age', shade = True).set_axis_labels('age','survived').add_legend()

sns.FacetGrid(train_df, hue= 'survived', height=8).map(sns.distplot, 'age').set_axis_labels('age','survived').add_legend()

plt.show()





for df in combined_dfs:

    df.loc[df['age'] <= 18, 'age'] = 0

    df.loc[(df['age'] > 18) & (df['age'] <= 30), 'age'] = 1

    df.loc[(df['age'] > 30) & (df['age'] <= 45), 'age'] = 2

    df.loc[(df['age'] > 45) & (df['age'] <= 60), 'age'] = 3

    df.loc[df['age'] > 60, 'age'] = 4

train_df.head()



for df in combined_dfs:

    df['sex'] = df['sex'].apply(lambda x:1 if x =='male' else 0)

train_df
train_df[['sex', 'survived']].groupby(['sex', 'survived']).size().unstack().plot(kind = 'bar', stacked = True, title = 'Sex distribution')

train_df[['embarked', 'survived']].groupby(['embarked', 'survived']).size().unstack().plot(kind = 'bar', stacked = True, title = 'embarked distribution')

print('{} missing value for embark\n'.format(train_df['embarked'].isnull().sum()))



# to see the distribution of embarked within different pclass

for i in range(1, 4):

    print(train_df[train_df['pclass'] == i]['embarked'].value_counts())
# fill out the Nan value with S cat 

for df in combined_dfs:

    df['embarked'] = df['embarked'].fillna('S')

print('{} Nan value of Embarked'.format(df['embarked'].isnull().sum()))
def embarked_encode(data):

    if data == 'C':

        data = 0

    elif data == 'Q':

        data = 1

    else:

        data = 2

    return data 



for df in combined_dfs:

    df['embarked'] = df['embarked'].astype(str).apply(embarked_encode)

    df.drop(['ticket', 'cabin'], axis = 1, inplace = True)

# train_df['embarked'] = train_df['embarked'].astype(str).apply(embarked_encode)

# train_df.drop(['ticket', 'cabin'], axis = 1, inplace = True)

# train_df



display(train_df)

display(test_df)

test_df.isnull().sum()

# fill out fare null by mean



test_df['fare'] = test_df['fare'].fillna(test_df['fare'].mean())

print('{} fare null value'.format(test_df['fare'].isnull().sum()))
train_df.isnull().sum()
X_train = train_df.loc[:, train_df.columns != 'survived']

y_train= train_df.survived.values

X_test = test_df



# display(X_train)

# display(y_train)

from xgboost.sklearn import XGBClassifier



xfb_clf = XGBClassifier(

        # 樹的個數

        n_estimators=100,

        # 學習率

        learning_rate= 0.3, 

        # 構建樹的深度，越大越容易過擬合    

        max_depth=6, 

        # 隨機取樣訓練樣本 訓練例項的子取樣比

        subsample=1, 

        # 用於控制是否後剪枝的引數,越大越保守，一般0.1、0.2這樣子

        gamma=0, 

        # 控制模型複雜度的權重值的L2正則化項引數，引數越大，模型越不容易過擬合。

        reg_lambda=1,  

        #最大增量步長，我們允許每個樹的權重估計。

        max_delta_step=0,

        # 生成樹時進行的列取樣 

        colsample_bytree=1, 



        # 這個引數預設是 1，是每個葉子裡面 h 的和至少是多少，對正負樣本不均衡時的 0-1 分類而言

        # 假設 h 在 0.01 附近，min_child_weight 為 1 意味著葉子節點中最少需要包含 100 個樣本。

        #這個引數非常影響結果，控制葉子節點中二階導的和的最小值，該引數值越小，越容易 overfitting。

        min_child_weight=1, 



        #隨機種子

        seed=1000)



# 模型 訓練

xfb_clf.fit(X_train, y_train, eval_metric='auc')

# 預測值

y_pred = xfb_clf.predict(X_test)
X_test['Survived'] = y_pred

X_test = X_test[['passengerid', 'Survived']]

X_test.rename(columns = {'passengerid': 'PassengerId'}, inplace = True)

X_test
X_test.to_csv('Titanic_output.csv', index = False)