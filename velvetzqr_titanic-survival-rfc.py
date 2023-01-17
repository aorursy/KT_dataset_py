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
train_path='../input/titanic/train.csv'

test_path='../input/titanic/test.csv'

traindf=pd.read_csv(train_path)

testdf=pd.read_csv(test_path)
print(traindf.shape[0],testdf.shape[0])
traindf.head()
# 合并数据集

alldf=traindf.append(testdf,ignore_index=True)

alldf.shape
alldf.describe()
alldf.info()
# 缺失值填充

alldf['Embarked']=alldf['Embarked'].fillna('S')

alldf['Cabin'].fillna('U',inplace=True)

alldf['Fare']=alldf['Fare'].fillna(alldf['Fare'].mean())

alldf['Age']=alldf['Age'].fillna(alldf['Age'].mean())
alldf.info()
# Sex映射

gender_map={'male':1,'female':0}

alldf['Sex']=alldf['Sex'].map(gender_map)
# Embarked提取one-hot编码

embarkeddf=pd.DataFrame()

embarkeddf=pd.get_dummies(alldf['Embarked'],prefix='Embarked')

alldf=pd.concat([alldf,embarkeddf],axis=1)

alldf.drop('Embarked',axis=1,inplace=True)
# Pclass提取one-hot编码

pclassdf=pd.DataFrame()

pclassdf=pd.get_dummies(alldf['Pclass'],prefix='Pclass')

alldf=pd.concat([alldf,pclassdf],axis=1)

alldf.drop('Pclass',axis=1,inplace=True)
# 从Name提取Title

def pop_title(ser):

    title_str=ser.split(',')[1].split('.')[0].strip()    # .strip()用于去除前后的空格

    return title_str



titledf=pd.DataFrame()

titledf['Title']=alldf['Name'].apply(pop_title)



title_mapDict = {"Capt":"Officer","Col":"Officer","Major":"Officer","Dr":"Officer","Rev":"Officer", 

                 "Jonkheer":"Royalty","Don":"Royalty","Sir" :"Royalty","the Countess":"Royalty","Dona":"Royalty",

                 "Mme":"Mrs","Mlle":"Miss","Ms":"Mrs","Mr" :"Mr","Mrs" :"Mrs","Miss" :"Miss",

                 "Master":"Master","Lady":"Royalty"}

titledf['Title']=titledf['Title'].map(title_mapDict)



titleDf=pd.get_dummies(titledf['Title'])

alldf=pd.concat([alldf,titleDf],axis=1)

alldf.drop('Name',axis=1,inplace=True)
# 从Cabin提取船舱等级

alldf['Cabin']=alldf['Cabin'].map(lambda x:x[0])

cabindf=pd.DataFrame

cabindf=pd.get_dummies(alldf['Cabin'],prefix='Cabin')
# 设计家庭大小评估

familysize=pd.DataFrame()

familysize['f_size']=alldf['SibSp']+alldf['Parch']+1



familysize['single_f']=familysize['f_size'].map(lambda x: 1 if x==1 else 0)

familysize['norm_f']=familysize['f_size'].map(lambda x: 1 if 2<=x<=4 else 0)

familysize['big_f']=familysize['f_size'].map(lambda x: 1 if x>4 else 0)



alldf=pd.concat([alldf,familysize],axis=1)
# 寻找特征

corrDF=alldf.corr()

corrDF['Survived'].sort_values(ascending=False)
all_x=pd.concat([embarkeddf, pclassdf,titleDf,cabindf,familysize,alldf['Fare'],alldf['Sex']],axis=1)
# 原始训练数据集行数

source_row=traindf.shape[0]



# 前891行数据为原始数据，需要拆分为训练数据和测试数据

source_x=all_x.loc[0:source_row-1,:]

source_y=alldf.loc[0:source_row-1,'Survived']



# 892行之后的数据数据需要提交给kaggle

pred_x=all_x.loc[source_row:,:]
from sklearn.model_selection import train_test_split



train_x,test_x,train_y,test_y=train_test_split(source_x,source_y,train_size=0.8)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score,GridSearchCV
# 选择最优控制随机状态参数random_state

l = []

for i in range(1,100,5):

    rfc = RandomForestClassifier(random_state=i)

    score = cross_val_score(rfc,source_x,source_y,cv=5).mean()

    l.append(score)



print(max(l),[*range(1,100,5)][l.index(max(l))])
# 创建初始模型

model=RandomForestClassifier(random_state=26)



# 交叉验证

score_pre = cross_val_score(model,source_x,source_y,cv=5)

score_pre.mean()
model.get_params
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# n_estimators 学习曲线

l = []

for i in range(1,301,20):

    rfc = RandomForestClassifier(n_estimators=i,random_state=26)

    score = cross_val_score(rfc,source_x,source_y,cv=5).mean()

    l.append(score)



print(max(l),[*range(1,301,20)][l.index(max(l))])



plt.figure(figsize=(20,5))

plt.plot(range(1,301,20),l)

plt.show()
# n_estimators 学习曲线：缩小范围

l = []

for i in range(1,101,2):

    rfc = RandomForestClassifier(n_estimators=i,random_state=26)

    score = cross_val_score(rfc,source_x,source_y,cv=5).mean()

    l.append(score)



print(max(l),[*range(1,101,2)][l.index(max(l))])



plt.figure(figsize=(20,5))

plt.plot(range(1,101,2),l)

plt.show()
# max_depth

param_grid = {'max_depth':range(3,30,2)}

rfc_grid = RandomForestClassifier(n_estimators=3,random_state=26)

GS = GridSearchCV(rfc_grid,param_grid,cv=5)

GS.fit(source_x,source_y)



print(GS.best_params_,GS.best_score_)
# min_samples_leaf

param_grid = {'min_samples_leaf':range(1,201,10)}

rfc_grid = RandomForestClassifier(n_estimators=3,max_depth=5,random_state=26)

GS = GridSearchCV(rfc_grid,param_grid,cv=5)

GS.fit(source_x,source_y)



print(GS.best_params_,GS.best_score_)
# min_samples_split

param_grid = {'min_samples_split':range(2,201,10)}

rfc_grid = RandomForestClassifier(n_estimators=3,max_depth=5,random_state=26)

GS = GridSearchCV(rfc_grid,param_grid,cv=5)

GS.fit(source_x,source_y)



print(GS.best_params_,GS.best_score_)
# max_features

param_grid = {'max_features':range(2,28)}

rfc_grid = RandomForestClassifier(n_estimators=3,max_depth=5,random_state=26)

GS = GridSearchCV(rfc_grid,param_grid,cv=5)

GS.fit(source_x,source_y)



print(GS.best_params_,GS.best_score_)
# criterion

param_grid = {'criterion':['gini','entropy']}

rfc_grid = RandomForestClassifier(n_estimators=3,max_depth=5,max_features=9,random_state=26)

GS = GridSearchCV(rfc_grid,param_grid,cv=5)

GS.fit(source_x,source_y)



print(GS.best_params_,GS.best_score_)
# oob_score

param_grid = {'oob_score':['False','True']}

rfc_grid = RandomForestClassifier(n_estimators=3,max_depth=5,max_features=9,random_state=26)

GS = GridSearchCV(rfc_grid,param_grid,cv=5)

GS.fit(source_x,source_y)



print(GS.best_params_,GS.best_score_)
# 最佳模型

rfc = RandomForestClassifier(n_estimators=3,max_depth=5,max_features=9,random_state=26)
rfc.fit(train_x,train_y)
pred_y=rfc.predict(pred_x)

pred_y=pred_y.astype(int)
passenger_id=alldf.loc[source_row:,'PassengerId']

pred_df=pd.DataFrame({'PassengerId':passenger_id,'Survived':pred_y})
pred_df.head()
pred_df.shape
pred_df.to_csv('titanic_pred_RFC.csv',index=False)