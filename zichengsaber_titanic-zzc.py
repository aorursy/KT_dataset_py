# import lib
import pandas as pd 
from pandas import Series,DataFrame
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
# sns.set_style('whitegrid')
%matplotlib inline
from  sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split
import xgboost
sns.set()
# read_file
train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')

train.head()
# 除去非必要的数据
train=train.drop(['PassengerId','Name','Ticket'],axis=1)
test=test.drop(['Name','Ticket'],axis=1)
# 检查是否有缺失的数据
train.info()
test.info()
# embark 登船地点
# 找出出现频率最高的
train.Embarked.value_counts().idxmax()

# 补齐缺失数据
train.Embarked=train.Embarked.fillna('S')
# Categorical plots
sns.catplot('Embarked','Survived',data=train,height=4,aspect=3,kind="point")

fig,(axis1,axis2,axis3)=plt.subplots(1,3,figsize=(15,5))
sns.countplot(x='Embarked',data=train,ax=axis1)
# 下面的表现了
#  加入hue表明我想根据Embarked来分别统计存活率
sns.countplot(x='Survived',hue='Embarked',data=train,order=[0,1],ax=axis2)
# 根据embarked属性来计算存活率的均值
embark_perc=train[["Embarked","Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
embark_dummies_titanic  = pd.get_dummies(train['Embarked'])
embark_dummies_titanic.drop(['S'],axis=1,inplace=True)

embark_dummies_test=pd.get_dummies(test.Embarked)
embark_dummies_test.drop(['S'],axis=1,inplace=True)

train=train.join(embark_dummies_titanic)
test=test.join(embark_dummies_test)

train.drop(['Embarked'],axis=1,inplace=True)
test.drop(['Embarked'],axis=1,inplace=True)

# Fare
# 测试集上费用有缺省,用中位数补全
test["Fare"].fillna(test.Fare.median(),inplace=True)

# get fare for survived && didn't survive passebgers
fare_not_survived=train["Fare"][train["Survived"]==0]
fare_survived=train["Fare"][train["Survived"]==1]
# 分别求出不存活的人的fare均值，和存活人的均值
avgerage_fare=pd.DataFrame([fare_not_survived.mean(),fare_survived.mean()])
# 
std_fare=pd.DataFrame([fare_not_survived.std(),fare_survived.std()])


# 画出fare的分布图
sns.distplot(train["Fare"],kde=False,color='blue')
# 调用pandas的画图库
avgerage_fare.index.names = std_fare.index.names = ["Survived"]
# 这个绘图操作。。。
avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
print(avgerage_fare)
print(std_fare)
fig1,(axis1,axis2)=plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values-Titanic')
axis2.set_title('New Age with random values-Titanic')

# get average,std,and number of NAN values in train
average_age_titanic=train["Age"].mean()
std_age_titanic=train["Age"].std()
count_nan_age_titanic = train["Age"].isnull().sum()
# get average,std,and number of NAN values in test_data
average_age_test=test["Age"].mean()
std_age_test=test["Age"].std()
count_nan_age_test=test["Age"].isnull().sum()
# 绘制出original Age values 分布
sns.distplot(train['Age'].dropna().astype(int),kde=False,bins=70,color='blue',ax=axis1)
# fill nan values in Age colunm with random
randnum1=np.random.randint(average_age_titanic-std_age_titanic,
                           average_age_titanic+std_age_titanic,
                           size=count_nan_age_titanic
                          )
randnum2=np.random.randint(average_age_test-std_age_test,
                           average_age_test+std_age_test,
                           size=count_nan_age_test,
)

train.Age[np.isnan(train.Age)]=randnum1
test.Age[np.isnan(test.Age)]=randnum2

# plot new distribution
sns.distplot(train.Age,bins=70,kde=False,color='blue',ax=axis2)
# .... continue with plot Age column

# peaks for survived/not survived passengers by their age
facet=sns.FacetGrid(train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train.Age.max()))
facet.add_legend()
train.Age=train.Age.astype(int)
test.Age=test.Age.astype(int)
fig,axis1=plt.subplots(1,1,figsize=(20,4))
# 单独提取出Age和Survived其中groupby表示自变量
# mean 相当于计算出了存活率的期望
average_age=train[["Age","Survived"]].groupby(["Age"],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
print(average_age)
# Cabin
# Cabin属性缺失值非常多
train.drop("Cabin",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)
# Family 
# Instead of having two columns Parch & Sibsp
# 我们可以只用一个属性来表示乘客是否有家属
# 这意味着我们假设 if having  any family member will increase chances of Survival
# 那么我们要如何验证我们的假设呢

# 想法有Parch的应该和有Sibsp的存活率分布近似
fig,(axis1,axis2)=plt.subplots(1,2,figsize=(10,4))
Sibsp_Survived=train[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean()
sns.barplot(x="SibSp",y="Survived",data=Sibsp_Survived,ax=axis1)
Parch_Survived=train[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean()
sns.barplot(x="Parch",y="Survived",data=Parch_Survived,ax=axis2)
train["Family"]=train["Parch"]+train["SibSp"]
train["Family"].loc[train["Family"]>0]=1
train["Family"].loc[train["Family"]==0]=0

test["Family"]=test.Parch+test.SibSp
test["Family"].loc[test["Family"]>0]=1
test["Family"].loc[test["Family"]==0]=0

# drop
train=train.drop(["SibSp","Parch"],axis=1)
test=test.drop(["SibSp","Parch"],axis=1)

#plot
fig,(axis1,axis2)=plt.subplots(1,2,sharex=True,figsize=(10,5))
# 人群中有多少是带家人的
sns.countplot(x='Family',data=train,order=[1,0],ax=axis1)
family_prec=train[["Family",'Survived']].groupby(['Family'],as_index=False).mean()
sns.barplot(x='Family',y='Survived',data=family_prec,order=[1,0],ax=axis2)
axis1.set_xticklabels(["with Family","Alone"],rotation=0)
# Sex 
# As we see,children(age<16)有可能有更高的概率存活
# 所以我们将乘客们分为男，女，小孩
def get_person(passenger):
    age,sex=passenger
    return 'child' if age<16 else sex
train['Person']=train[['Age','Sex']].apply(get_person,axis=1)
test['Person']=test[['Age','Sex']].apply(get_person,axis=1)

train.drop(['Sex'],axis=1,inplace=True)
test.drop(['Sex'],axis=1,inplace=True)

#将train.Person重新编码
person_titanic=pd.get_dummies(train["Person"])
person_titanic.columns=['Child','Female','Male']
person_titanic.drop(['Male'], axis=1, inplace=True)

person_test=pd.get_dummies(test["Person"])
person_test.columns=['Child','Female','Male']
person_test.drop(['Male'],axis=1,inplace=True)

train=train.join(person_titanic)
test=test.join(person_test)

fig,(axis1,axis2)=plt.subplots(1,2,figsize=(10,5))
sns.countplot(x='Person',data=train,ax=axis1)

person_prec=train[["Person","Survived"]].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person',y='Survived',data=person_prec,ax=axis2,order=[
    'male','female','child'
])

train.drop(['Person'],axis=1,inplace=True)
test.drop(['Person'],axis=1,inplace=True)

sns.catplot('Pclass','Survived',order=[1,2,3],data=train,height=5,kind='point')
# drop class3 消除虚拟变量陷阱
Pclass_titanic=pd.get_dummies(train['Pclass'])
Pclass_titanic.columns=['Class1','Class2','Class3']
Pclass_titanic.drop(['Class3'],axis=1,inplace=True)

Pclass_test=pd.get_dummies(test['Pclass'])
Pclass_test.columns=['Class1','Class2','Class3']
Pclass_test.drop(['Class3'],axis=1,inplace=True)

train.drop(['Pclass'],axis=1,inplace=True)
test.drop(['Pclass'],axis=1,inplace=True)


train=train.join(Pclass_titanic)
test=test.join(Pclass_test)
y=train.Survived
X=train.drop(["Survived"],axis=1)
# training set and validation set
train_X,val_X,train_y,val_y=train_test_split(X,y,train_size=0.8,random_state=0)
for n in [50,100,150,200,250,300,350]:
    rf_model=RandomForestClassifier(n_estimators=n,random_state=0)
    rf_model.fit(train_X,train_y)
    rf_pred=rf_model.predict(val_X)
    rf_val_acc=accuracy_score(rf_pred,val_y)
    print('n={},loss={}'.format(n,rf_val_acc))
#选择n=200
# X_test=test.drop("PassengerId",axis=1).copy()
# model=RandomForestClassifier(n_estimators=200,random_state=0)
# model.fit(X,y)
# Y_pred=model.predict(X_test)
from xgboost import XGBClassifier
for i in [50,100,150,200,250,300]:
    gb_model=XGBClassifier(max_depth=4,n_estimators=i)
    gb_model.fit(train_X,train_y)
    gb_pred=gb_model.predict(val_X)
    print(i,'=>',accuracy_score(gb_pred,val_y))
X_test=test.drop("PassengerId",axis=1).copy()
model=XGBClassifier(max_depth=4,n_estimators=150)
model.fit(X,y)
Y_pred=model.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)