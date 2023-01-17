# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pandas import Series, DataFrame

data_train=pd.read_csv('/kaggle/input/train.csv')

data_train
data_train.info()
data_train.describe()
# -*- coding: utf-8 -*

import matplotlib

import matplotlib.pyplot as plt

from pylab import * 

mpl.rcParams['font.sans-serif']=['SimHei']

mpl.rcParams['axes.unicode_minus']=False

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图

data_train.Survived.value_counts().plot(kind='bar')# 柱状图 

plt.title("Survival Situation (1 is survival)") # 标题

plt.ylabel(u"num of passager")  



plt.subplot2grid((2,3),(0,1))

data_train.Pclass.value_counts().plot(kind="bar")

plt.ylabel(u"num of passager")

plt.title(u"Pclass")



plt.subplot2grid((2,3),(0,2))

plt.scatter(data_train.Survived, data_train.Age)

plt.ylabel(u"Age")                         # 设定纵坐标名称

plt.grid(b=True, which='major', axis='y') 

plt.title(u"Age with survial (1 is survival)")





plt.subplot2grid((2,3),(1,0), colspan=2)

data_train.Age[data_train.Pclass == 1].plot(kind='kde')   

data_train.Age[data_train.Pclass == 2].plot(kind='kde')

data_train.Age[data_train.Pclass == 3].plot(kind='kde')

plt.xlabel(u"age")# plots an axis lable

plt.ylabel(u"density") 

plt.title(u"Age distribution of passengers of all levels")

plt.legend((u'First class', u'second class',u'third class'),loc='best') # sets our legend for our graph.





plt.subplot2grid((2,3),(1,2))

data_train.Embarked.value_counts().plot(kind='bar')

plt.title(u"Number of people boarding at each boarding port")

plt.ylabel(u"num of passager")  

plt.show()

#看看各乘客等级的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()

df=pd.DataFrame({u'S':Survived_1, u'NS':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"Pclass and S")

plt.xlabel(u"PC") 

plt.ylabel(u"NUM") 

plt.show()

#看看各性别的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()

Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()

df=pd.DataFrame({u'M':Survived_m, u'W':Survived_f})

df.plot(kind='bar', stacked=True)

plt.title(u"Seeing the situation of salvation by sex")

plt.xlabel(u"Gender") 

plt.ylabel(u"Num")

plt.show()



 #然后我们再来看看各种舱级别情况下各性别的获救情况

fig=plt.figure()

fig.set(alpha=0.65) # 设置图像透明度，无所谓

plt.title(u"Rescue according to class and gender")



ax1=fig.add_subplot(141)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')

ax1.set_xticklabels([u"S", u"NS"], rotation=0)

ax1.legend([u"F/HC"], loc='best')



ax2=fig.add_subplot(142, sharey=ax1)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')

ax2.set_xticklabels([u"NS", u"S"], rotation=0)

plt.legend([u"F/LC"], loc='best')



ax3=fig.add_subplot(143, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')

ax3.set_xticklabels([u"NS", u"S"], rotation=0)

plt.legend([u"M/HC"], loc='best')



ax4=fig.add_subplot(144, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')

ax4.set_xticklabels([u"NS", u"S"], rotation=0)

plt.legend([u"M/LC"], loc='best')



plt.show()



fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()

df=pd.DataFrame({u'S':Survived_1, u'NS':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"Rescue of passengers in each registered port")

plt.xlabel(u"registered port") 

plt.ylabel(u"NUM") 



plt.show()



g = data_train.groupby(['SibSp','Survived'])

df = pd.DataFrame(g.count()['PassengerId'])

print (df)



g = data_train.groupby(['Parch','Survived'])

df = pd.DataFrame(g.count()['PassengerId'])

print (df)





#ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴把

#cabin只有204个乘客有值，我们先看看它的一个分布

data_train.Cabin.value_counts()





fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()

Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()

df=pd.DataFrame({u'WC':Survived_cabin, u'WNC':Survived_nocabin}).transpose()

df.plot(kind='bar', stacked=True)

plt.title(u"According to Cabin, see if you are rescued.")

plt.xlabel(u"With Carbin or not") 

plt.ylabel(u"NUM")

plt.show()





from sklearn.ensemble import RandomForestRegressor

 

### 使用 RandomForestClassifier 填补缺失的年龄属性

def set_missing_ages(df):

    

    # 把已有的数值型特征取出来丢进Random Forest Regressor中

    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    age_df.info()

    # 乘客分成已知年龄和未知年龄两部分

    known_age = age_df[age_df.Age.notnull()].values

    unknown_age = age_df[age_df.Age.isnull()].values

    print(known_age)

    print(unknown_age)

    # y即目标年龄

    y = known_age[:, 0]



    # X即特征属性值

    X = known_age[:, 1:]



    # fit到RandomForestRegressor之中

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)

    rfr.fit(X, y)

    

    # 用得到的模型进行未知年龄结果预测

    predictedAges = rfr.predict(unknown_age[:, 1::])

    

    # 用得到的预测结果填补原缺失数据

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df, rfr



def set_Cabin_type(df):

    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"

    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"

    return df



data_train, rfr = set_missing_ages(data_train)

data_train = set_Cabin_type(data_train)





dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')



dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')



dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')



dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')



df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df



import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()

age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))

df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)

fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))

df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)

df.head()

from sklearn import linear_model



# 用正则取出我们要的属性值

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

train_np = train_df.values



# y即Survival结果

y = train_np[:, 0]



# X即特征属性值

X = train_np[:, 1:]



# fit到RandomForestRegressor之中

clf = linear_model.LogisticRegression(solver='liblinear',C=1.0, penalty='l1', tol=1e-6)

clf.fit(X, y)

    

clf

data_test = pd.read_csv("/kaggle/input/test.csv")

data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0

# 接着我们对test_data做和train_data中一致的特征变换

# 首先用同样的RandomForestRegressor模型填上丢失的年龄

tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

null_age = tmp_df[data_test.Age.isnull()].values

# 根据特征属性X预测年龄并补上

X = null_age[:, 1:]

predictedAges = rfr.predict(X)

data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges



data_test = set_Cabin_type(data_test)

dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')





df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)

df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)

df_test.head()
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions = clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})

result.to_csv("logistic_regression_predictions.csv", index=False)

pd.read_csv("logistic_regression_predictions.csv").head()
pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})

# from sklearn import cross_validation

# 参考https://blog.csdn.net/cheneyshark/article/details/78640887 ， 0.18版本中，cross_validation被废弃

# 改为下面的从model_selection直接import cross_val_score 和 train_test_split

from sklearn.model_selection import cross_val_score, train_test_split



 #简单看看打分情况

clf = linear_model.LogisticRegression(solver='liblinear',C=1.0, penalty='l1', tol=1e-6)

all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

X = all_data.values[:,1:]

y = all_data.values[:,0]

# print(cross_validation.cross_val_score(clf, X, y, cv=5))

print(cross_val_score(clf, X, y, cv=5))
# 分割数据，按照 训练数据:cv数据 = 7:3的比例

# split_train, split_cv = cross_validation.train_test_split(df, test_size=0.3, random_state=0)

split_train, split_cv = train_test_split(df, test_size=0.3, random_state=42)



train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

# 生成模型

clf = linear_model.LogisticRegression(solver='liblinear',C=1.0, penalty='l1', tol=1e-6)

clf.fit(train_df.values[:,1:], train_df.values[:,0])



# 对cross validation数据进行预测



cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions = clf.predict(cv_df.values[:,1:])



origin_data_train = pd.read_csv("/kaggle/input/train.csv")

bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.values[:,0]]['PassengerId'].values)]

bad_cases.head(10)
import numpy as np

import matplotlib.pyplot as plt

# from sklearn.learning_curve import learning_curve  修改以fix learning_curve DeprecationWarning

from sklearn.model_selection import learning_curve



# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 

                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):

    """

    画出data在某模型上的learning curve.

    参数解释

    ----------

    estimator : 你用的分类器。

    title : 表格的标题。

    X : 输入的feature，numpy类型

    y : 输入的target vector

    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点

    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)

    n_jobs : 并行的的任务数(默认1)

    """

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    

    if plot:

        plt.figure()

        plt.title(title)

        if ylim is not None:

            plt.ylim(*ylim)

        plt.xlabel(u"训练样本数")

        plt.ylabel(u"得分")

        plt.gca().invert_yaxis()

        plt.grid()

    

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 

                         alpha=0.1, color="b")

        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 

                         alpha=0.1, color="r")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")

        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

    

        plt.legend(loc="best")

        

        plt.draw()

        plt.gca().invert_yaxis()

        plt.show()

    

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2

    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])

    return midpoint, diff



plot_learning_curve(clf, u"学习曲线", X, y)

from sklearn.ensemble import BaggingRegressor



train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')

train_np = train_df.values



# y即Survival结果

y = train_np[:, 0]



# X即特征属性值

X = train_np[:, 1:]



# fit到BaggingRegressor之中

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)

bagging_clf.fit(X, y)



test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')

predictions = bagging_clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})

result.to_csv("logistic_regression_bagging_predictions.csv", index=False)
pd.read_csv("logistic_regression_bagging_predictions.csv").head(10)