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

import numpy as np

import pandas as pd

data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")

df = data_train.append(data_test)

print("合并后的数据一共有{0}条".format(str(df.shape[0])))

print(pd.isnull(df).sum())

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



plt.style.use("fast")

plt.rc('font', family='SimHei', size=13)

cat_list = ['Pclass','Name','Sex','SibSp','Embarked','Parch','Ticket','Cabin']

for n,i in enumerate(cat_list):  

    Cabin_cat_num = df[i].value_counts().shape[0]

    print('{0}. {1}特征的类型数量是: {2}'.format(n+1,i,Cabin_cat_num))

f, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(20,5))

sns.countplot(x='Sex', hue='Survived', data=data_train, ax=ax1)

sns.countplot(x='Pclass', hue='Survived', data=data_train, ax=ax2)

sns.countplot(x='Embarked', hue='Survived', data=data_train, ax=ax3)

ax1.set_title('Sex特征分析')

ax2.set_title('Pclass特征分析')

ax3.set_title('Embarked特征分析')

f.suptitle('定类/定序数据类型特征分析',size=20,y=1.1)



f, [ax1,ax2] = plt.subplots(1,2,figsize=(20,5))

sns.countplot(x='SibSp', hue='Survived', data=data_train, ax=ax1)

sns.countplot(x='Parch', hue='Survived', data=data_train, ax=ax2)

ax1.set_title('SibSp特征分析')

ax2.set_title('Parch特征分析')



plt.show()
grid = sns.FacetGrid

grid = sns.FacetGrid(df, col='Pclass', hue='Sex',size=5)

grid.map(sns.countplot, 'Embarked', alpha=1)

grid.add_legend()
grid = sns.FacetGrid(data_train, col='Pclass', row='Sex',hue='Survived', palette='seismic', size=4)

grid.map(sns.countplot, 'Embarked', alpha=1)

grid.add_legend()
# kde分布

f,ax = plt.subplots(figsize=(10,5))

sns.kdeplot(data_train.loc[(data_train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')

sns.kdeplot(data_train.loc[(data_train['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')

plt.title('Age特征分布 - Surviver V.S. Not Survivors', fontsize = 15)

plt.xlabel("Age", fontsize = 15)

plt.ylabel('Frequency', fontsize = 15)







fig,[ax1,ax2] = plt.subplots(1,2,figsize=(20,6))



sns.boxplot(x="Pclass",y="Age",data=data_train, ax = ax1)

sns.swarmplot(x="Pclass",y="Age",data=data_train,ax = ax1)

sns.kdeplot(data_train.loc[(data_train['Pclass'] == 3),'Age'],color='b',shade=True,label='Pclass3',ax=ax2)

sns.kdeplot(data_train.loc[(data_train['Pclass'] == 2),'Age'],color = 'g',shade=True,label='Pclass2',ax=ax2)

sns.kdeplot(data_train.loc[(data_train['Pclass']==1),'Age'],color = 'r',shade=True,label='Pclass1',ax=ax2)

ax1.set_title('kkkkk')

ax2.set_title('xxxx')

fig.show()

# # Sex，Pclass分类条件下的 Age年龄对Survived的散点图

# grid = sns.FacetGrid(data_train, row='Sex', col='Pclass', hue='Survived', palette='seismic', size=3.5)

# grid.map(plt.scatter, 'PassengerId', 'Age')

# grid.add_legend()
fig,[ax1,ax2] = plt.subplots(1,2,figsize=(20,6))

sns.boxplot(x='Pclass',y='Fare',data=data_train,ax=ax1)

sns.swarmplot(x='Pclass',y='Fare',data=data_train,ax=ax1)

sns.kdeplot(data_train.loc[(data_train['Pclass']==1),'Fare'],color='b',shade=True,label='Pclass1',ax=ax2)

sns.kdeplot(data_train.loc[(data_train['Pclass']==2),'Fare'],color='g',shade=True,label='Pclass2',ax=ax2)

sns.kdeplot(data_train.loc[(data_train['Pclass']==3),'Fare'],color='r',shade=True,label='Pclass3',ax=ax2)

ax1.set_title('111')

ax2.set_title('222')


fit_mean=df.loc[(df['Pclass']==3)&(df['Age']>60)&(df['Sex']=='male')].Fare.mean()

df.loc[df.PassengerId==1044,'Fare']=fit_mean

df[df.PassengerId==1044]

df.head()
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor

import re

import operator

from sklearn.feature_selection import SelectKBest, f_classif

df[df.Embarked.isnull()]

df['Embarked']=df['Embarked'].fillna('C')

df['CabinCat'] = pd.Categorical(df.Cabin.fillna('0').apply(lambda x: x[0])).codes

df['CabinCat']
fig, ax = plt.subplots(figsize=(10,5))

sns.countplot(x='CabinCat', hue='Survived',data=df)

plt.show()
#比较有趣的一个深度特征，就是CabinCat的奇偶性，解释的话我觉得应该是走廊两边的住户因为穿的结构影响，逃生上存在着差异。

#网上找的对这个的处理是直接做除法分3类，因为没有房间号的必须也要列出来，不然归入偶数就不准确了

def get_type_cabine(cabine):

    # Use a regular expression to search for a title. 

    cabine_search = re.search('\d+', cabine)

    # If the title exists, extract and return it.

    if cabine_search:

        num = cabine_search.group(0)

        if np.float64(num) % 2 == 0:

            return '2'

        else:

            return '1'

    return '0'

df["Cabin"] = df["Cabin"].fillna(" ")



df["CabinType"] = df["Cabin"].apply(get_type_cabine)

print(pd.value_counts(df["CabinType"]))

df['surname'] = df["Name"].apply(lambda x: x.split(',')[0].lower())

df.loc[df['surname']=='abbott']

df.loc[(df['surname']=='abbott')&(df['Age']==35),'SibSp'] = 0

df.loc[(df['surname']=='abbott')&(df['Age']==35),'Parch'] = 2

df.loc[(df['surname']=='abbott')&(df['Age']==13),'SibSp'] = 1

df.loc[(df['surname']=='abbott')&(df['Age']==13),'Parch'] = 1

df.loc[(df['surname']=='ford')&(df['Age']==16),'SibSp'] = 3

df.loc[(df['surname']=='ford')&(df['Age']==16),'Parch'] = 1

df.loc[(df['surname']=='ford')&(df['Age']==9),'SibSp'] = 3

df.loc[(df['surname']=='ford')&(df['Age']==9),'Parch'] = 1

df.loc[(df['surname']=='ford')&(df['Age']==21),'SibSp'] = 3

df.loc[(df['surname']=='ford')&(df['Age']==21),'Parch'] = 1

df.loc[(df['surname']=='ford')&(df['Age']==48),'SibSp'] = 0

df.loc[(df['surname']=='ford')&(df['Age']==48),'Parch'] = 4

df.loc[(df['surname']=='ford')&(df['Age']==18),'SibSp'] = 3

df.loc[(df['surname']=='ford')&(df['Age']==18),'Parch'] = 1
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 根据FamilySize分布进行分箱

df['FamilySize'].value_counts()

df["FamilySize"] = pd.cut(df["FamilySize"], bins=[0,1,2,3,20], labels=[0,1,2,3])

df['FamilySize']
#我观察到有很多的人票号都相同，通过数数发现有最多一个Ticket对应11个人的，当然一大部分是对应一个人

#

bucket=df['Ticket'].value_counts()

df['SameTicket'] = df["Ticket"].apply(lambda x:bucket[x])

# #因为有些票价很高，已经可以说是异常值了，但是可能是因为这个人买了全家的车票，其他家人票价相对就变低了，这里很难去确定那些人是一家人，只能暂且将surname相同的进行一并处理

# new_df = data_train.append(data_test)

# new_df['surname'] = df["Name"].apply(lambda x: x.split(',')[0].lower())

# new_df.groupby('surname').mean()

# new_df.head(10)

# # new_df.rename(columns={'Fare':'MeanFare'}, inplace = True)

# # df=pd.concat([df,new_df['MeanFare']])

# # df.head(10)

# #因为比较菜，不知道有没有什么更好的方法，只能优先解决问题

# df.describe()

# df.loc[df['Fare']>500]

#上面的经过实践发现效果不好，舍弃
df.head()

df['Embarked']=pd.Categorical(df.Embarked).codes

# 对Sex特征进行独热编码分组

df = pd.concat([df,pd.get_dummies(df['Sex'])],axis=1)

df.head()


f,ax = plt.subplots(figsize=(10,5))

sns.kdeplot(df.loc[(df['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')

sns.kdeplot(df.loc[(df['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')

plt.title('Age特征分布 - Surviver V.S. Not Survivors', fontsize = 15)

plt.xlabel("Age", fontsize = 15)

plt.ylabel('Frequency', fontsize = 15)

df.head()
child = 18

baby = 6

def get_person(passenger):

    age, sex = passenger

    if (age < baby):

        return 'baby'

    elif(age<child):

        return 'child'

    elif (sex == 'female'):

        return 'female_adult'

    else:

        return 'male_adult'

df = pd.concat([df, pd.DataFrame(df[['Age','Sex']].apply(get_person,axis=1),columns=['person'])],axis=1)

df.head()
df = pd.concat([df,pd.get_dummies(df['person'])],axis=1)

df.head()
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor



classers = ['Fare','Parch','Pclass','SibSp',

            'female','male', 'Embarked', 'FamilySize']

etr = ExtraTreesRegressor(n_estimators=200,random_state=0)

X_train = df[classers][df['Age'].notnull()]

Y_train = df['Age'][df['Age'].notnull()]

X_test = df[classers][df['Age'].isnull()]



etr.fit(X_train.as_matrix(),np.ravel(Y_train))

age_preds = etr.predict(X_test.as_matrix())

df['Age'][df['Age'].isnull()] = age_preds
X_test['Age'] = pd.Series(age_preds)

f,ax=plt.subplots(figsize=(10,5))

sns.swarmplot(x='Pclass',y='Age',data=X_test)

plt.show()
#根据mr，ms等title信息提取一个头衔特征

df["Title"] = df["Name"].apply(lambda x: re.search(' ([A-Za-z]+)\.',x).group(1))

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 2, "Mme": 3,"Don": 9,"Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

df["TitleCat"] = df.loc[:,'Title'].map(title_mapping)

#名字的长短对于死亡率也有影响，试着加进去

df["NameLength"] = df["Name"].apply(lambda x: len(x))



from sklearn.feature_selection import SelectKBest, f_classif,chi2

from sklearn.model_selection import train_test_split



target = data_train["Survived"].values

features= ['female','male','Age','male_adult','female_adult', 'child','TitleCat',

           'Pclass','NameLength', 'SibSp', 'Parch','CabinCat','CabinType','SameTicket',

           'Fare','Embarked','FamilySize','baby']

train = df[0:891].copy()

test = df[891:].copy()

rfc = RandomForestClassifier(n_estimators=3000, min_samples_split=4, class_weight={0:0.745,1:0.255})

rfc.fit(train[features], target)

predictions = rfc.predict(test[features])

PassengerId =np.array(test["PassengerId"]).astype(int)

my_prediction = pd.DataFrame(predictions, PassengerId, columns = ["Survived"])



my_prediction.to_csv("my_prediction.csv", index_label = ["PassengerId"])




