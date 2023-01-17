# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import explained_variance_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split  # 함수
train_df = pd.read_csv('../input/train.csv') #df = data frame

train_df.head(10)
print(train_df.columns)
train_df.info()
for column in train_df:

    print(column,':', train_df[column].nunique())
# 지정한 컬럼들 간의 관계를 그래프로 그림. 

plt.figure(figsize=(10,6))

sns.plotting_context('notebook',font_scale=1.2)

cols = ['sqft_lot','sqft_above','sqft_living', 'bedrooms','grade','price']

g = sns.pairplot(train_df[cols], hue='bedrooms',size=2)

g.set(xticklabels=[])
import warnings

warnings.filterwarnings('ignore')
# sqft_living과 price간의 관계를 표시하되 등급(grade)을 다른 색으로 출력하다. 

sns.lmplot(x='sqft_living', y='price', hue='grade', data=train_df, fit_reg=False)

#CHECK THE PPT SLIDE
sns.lmplot(x='sqft_living', y='price', hue='condition', data=train_df, fit_reg=False)
sns.lmplot(x='sqft_living', y='price', hue='waterfront', data=train_df, fit_reg=False)
plt.figure(figsize=(15,10))

columns =['sqft_lot','sqft_above','sqft_living', 'bedrooms','grade','price']

sns.heatmap(train_df[columns].corr(),annot=True)

#CHECK THE PPT SLIDE
f, sub = plt.subplots(1, 1,figsize=(12.18,5))

sns.boxplot(x=train_df['bedrooms'],y=train_df['price'], ax=sub)

sub.set(xlabel='Bedrooms', ylabel='Price');
f, sub = plt.subplots(1, 2,figsize=(12,4))

sns.boxplot(x=train_df['bedrooms'],y=train_df['price'], ax=sub[0])

sns.boxplot(x=train_df['floors'],y=train_df['price'], ax=sub[1])

sub[0].set(xlabel='Bedrooms', ylabel='Price')

sub[1].yaxis.set_label_position("right")

sub[1].yaxis.tick_right()

sub[1].set(xlabel='Floors', ylabel='Price')
f, axe = plt.subplots(1, 1,figsize=(12,5))

sns.boxplot(x=train_df['grade'],y=train_df['price'], ax=axe)

axe.set(xlabel='Grade', ylabel='Price');
from mpl_toolkits.mplot3d import Axes3D



fig=plt.figure(figsize=(12,8))



ax=fig.add_subplot(1,1,1, projection="3d")

ax.scatter(train_df['floors'],train_df['bedrooms'],train_df['sqft_living'],c="darkred",alpha=.5)

ax.set(xlabel='\nFloors',ylabel='\nBedrooms',zlabel='\nsqft Living')

ax.set(ylim=[0,12])
fig=plt.figure(figsize=(12,8))

ax=fig.add_subplot(1,1,1, projection="3d")

ax.scatter(train_df['sqft_living'],train_df['sqft_lot'],train_df['bedrooms'],c="darkgreen",alpha=.5)

ax.set(xlabel='\nSqft Living',ylabel='\nsqft Lot',zlabel='Bedrooms')

ax.set(ylim=[0,250000]);
train_df = train_df.drop(['id', 'date'], axis=1) 
train_df_part1, train_df_part2 = train_test_split(train_df, train_size = 0.8, random_state=3)  # 3=seed
print(train_df.shape, train_df_part1.shape, train_df_part2.shape)
from sklearn import linear_model



gildong = LinearRegression()

gildong.fit(train_df_part1[['sqft_living']], train_df_part1[['price']])
score = gildong.score(train_df_part2[['sqft_living']], train_df_part2['price'])

print(format(score,'.3f'))
predicted = gildong.predict(train_df_part2[['sqft_living']])

print(predicted, '\n', predicted.shape)
print(train_df_part2['price'], '\n', predicted.shape)
print('Intercept: {}'.format(gildong.intercept_))

print('Coefficient: {}'.format(gildong.coef_))
features = ['sqft_living','bedrooms','bathrooms']
gildong = LinearRegression()

gildong.fit(train_df_part1[features], train_df_part1['price'])

score = gildong.score(train_df_part2[features], train_df_part2['price'])

print(format(score,'.3f'))
features = ['sqft_living','bedrooms','bathrooms','sqft_lot','floors','zipcode']
gildong = LinearRegression()

gildong.fit(train_df_part1[features], train_df_part1['price'])

score = gildong.score(train_df_part2[features], train_df_part2['price'])

print(format(score,'.3f'))
features = ['sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'view',

             'grade','yr_built','zipcode']
gildong = LinearRegression()

gildong.fit(train_df_part1[features], train_df_part1['price'])

score = gildong.score(train_df_part2[features], train_df_part2['price'])

print(format(score,'.3f'))
from sklearn.neighbors import KNeighborsRegressor



babo = KNeighborsRegressor(n_neighbors=10)

babo.fit(train_df_part1[features], train_df_part1['price'])

score = babo.score(train_df_part2[features], train_df_part2['price'])

print(format(score,'.3f'))
youngja = DecisionTreeRegressor(random_state = 0)

youngja.fit(train_df_part1[features], train_df_part1['price'])

score = youngja.score(train_df_part2[features], train_df_part2['price'])

print(format(score,'.3f'))



predicted = youngja.predict(train_df_part2[features])

print(predicted, '\n', predicted.shape)
cheolsu = RandomForestRegressor(n_estimators=28,random_state=0)

cheolsu.fit(train_df_part1[features], train_df_part1['price'])

score = cheolsu.score(train_df_part2[features], train_df_part2['price'])

print(format(score,'.3f'))



predicted = youngja.predict(train_df_part2[features])

print(predicted, '\n', predicted.shape)
test_df = pd.read_csv('../input/test.csv')

print(test_df.shape)

predicted = cheolsu.predict(test_df[features])

print(predicted.shape)

print(predicted)
sub = pd.read_csv('../input/sample_submission.csv')

sub['price'] = predicted 

sub.to_csv('my_submission.csv', index=False)

print('Submission file created!')