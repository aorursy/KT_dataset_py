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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

print('row:',df.shape[0],'column:',df.shape[1])

print('-'*20)

print(df.sample(7))

print('-'*20)

print(df.isna().sum())
##salary 有遺失值的那些資料尚未找到適當的工作,所以沒薪水

print(df[df['salary'].isna()].status.unique())

df['salary'] =df['salary'].fillna(0)
print(df.nunique())

cate_var = [i for i in df if df[i].nunique()<10]

print(cate_var)

continuous_var = [i for i in df if (df[i].nunique()>10)]

print(continuous_var)

y = df['salary']
print('salary_median: %.2f'%df['salary'].median())

print('salary_mean: %.2f'%df['salary'].mean())

##check the outlier of salary and its distribution

sns.boxplot(data=df['salary'])

plt.show()

box_length = np.percentile(df['salary'],75)-np.percentile(df['salary'],25)

max_ = 1.5*box_length + np.percentile(df['salary'],75)

print(df[df['salary']>max_])



sns.distplot(df['salary'])

plt.show()
#check the relationship between continuous varibles

sns.pairplot(df[continuous_var])

plt.show()
##Let's start to do some data visualization

##the relationship between specialisation and salary

##by the following plot, we can understand which specialisation get more money than fifty percent of people



print('salary > %.2f'% df['salary'].median())

for i in df['specialisation'].unique():

    for gender in df['gender'].unique():

        print(i,gender,len(df[(df['specialisation']==i)&(df['salary']>df['salary'].median())&(df['gender']==gender)]))

print('-'*15)

print('Not Placed (salary=0)')

for i in df['specialisation'].unique():

    for gender in df['gender'].unique():

        print(i,gender,len(df[(df['specialisation']==i)&(df['salary']==0)&(df['gender']==gender)]))

sns.catplot(x='specialisation',y='salary',data=df,hue='gender')

plt.show()
#the population in every category

plt.figure(figsize=(16,10))

for index,crv in enumerate(cate_var):

    plt.subplot(2,4,index+1)

    plt.title('%s'%crv)

    plt.bar(x=df[crv].unique(),height=df[crv].value_counts())
fig = plt.figure(figsize=(20,12))

for index,crv in enumerate(cate_var):

    ax = fig.add_subplot(2,4,index+1)

    sns.boxplot(x=crv,y='salary',data=df,ax=ax)

plt.show()
print('salary > %.2f'% df['salary'].median())

print()

for crv in cate_var:

    print(crv)

    for i in df[crv].unique():

        print(i+':',len(df[(df[crv]==i)&(df['salary']>df['salary'].median())])/len(df[df[crv]==i]),

              end=' | ' if df[crv].unique()[-1]!=i else '\n')

    print('-'*15 if crv!=cate_var[-1] else '-'*70)

print('Not Placed (salary=0)')

print()

for crv in cate_var:

    print(crv)

    for i in df[crv].unique():

        print(i+':',len(df[(df[crv]==i)&(df['salary']==0)])/len(df[df[crv]==i]),

             end=' | ' if df[crv].unique()[-1]!=i else '\n')

    print('-'*15)
# According the proportion of salary which surpass the median of salary ,choose some features to explore 

fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(131)

df.groupby(by='gender').salary.mean().plot(kind='bar',ax=ax1)

ax2 = fig.add_subplot(132)

df.groupby(by='workex').salary.mean().plot(kind='bar',ax=ax2)

ax3 = fig.add_subplot(133)

df.groupby(by='specialisation').salary.mean().plot(kind='bar',ax=ax3)

plt.show()

print(df.groupby(by='gender').salary.mean())

print('-'*10)

print(df.groupby(by='workex').salary.mean())

print('-'*10)

print(df.groupby(by='specialisation').salary.mean())
from sklearn.preprocessing import StandardScaler

continuous_var.remove('salary') ##remove y(target)

scaler = StandardScaler()

scaler.fit(df[continuous_var])

df[continuous_var] = scaler.transform(df[continuous_var])

df = pd.get_dummies(df,columns=cate_var) #one-hot encoding
df = df.drop(['sl_no','salary'],axis=1)

df.head()
# split the data into two pieces

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.2)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
# import Alogrithm and evaluate the efficacy (predict salary)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,mean_squared_error



def relu(a):

    '''

    relu

    '''

    for index,item in enumerate(a):

        if item<=0:

            a[index] =0

        else:

            a[index] =item

    return(a)



lr = LinearRegression()

lr.fit(X_train,y_train)

y_pre = lr.predict(X_test)

#print(y_pre)

y_pre = relu(y_pre)

#print(y_pre)

r2 = r2_score(y_test,y_pre)

mse = mean_squared_error(y_test,y_pre)

print('r2:%.2f'%r2)

print('mse:%.2f'%mse)
#cross validation

from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(estimator=lr,X=df,y=y,cv=5)

print('r2:',cv_score.mean())