train.describe()
train.columns
#for analysis of data, dataframe

import numpy as np

import pandas as pd



#for plotting and stuffs

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

#the above line of code is known as a magic function, helps to display our plots just below our code in the notebook.



#for model training & prediction

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
#read training data into 'train_df' dataframe

train_df=pd.read_csv('../input/train.csv')



#read testing data into 'test_df' dataframe

test_df=pd.read_csv('../input/test.csv')



#combined dataset, will be handy in wrangling steps.

combined_df=[train_df,test_df]
train_df.shape
test_df.shape

test_df.columns
len(combined_df) # 只是2个数据集放在1个文件里，1个文件里有2条记录，每条记录是原来的1个数据集。

# 但是combine_df没有用到。
train_df_copy = train_df

train_df_copy.shape
train_df_copy = train_df

train_df_copy.combine_first(test_df)

# 用test填充train，训练拟合模型会导致数据泄露？因为test数据已经进入训练环节了。

# 用test_df的数据填充train_df_copy的缺失值，是一种填补缺失值的方法。
train_df.columns
test_df.columns
#to know what type of data columns hold ; 'object' type means they hold string values

train_df.dtypes
test_df.dtypes
train_df.info()
test_df.info()
#train_df.info(verbose=False) will give a compact version of the above output, it set to True by default(in above case).

train_df.info(verbose=False)
train_df.head() #by default it prints first 5 rows, any other integer can also be given inside parenthesis.
test_df.head()
train_df.describe()
ax=train_df['Sex'].value_counts().plot.bar(title='Sex Distribution aboard Titanic',figsize=(8,4))



#below loop is to print numeric value above the bars

for p in ax.patches:

    ax.annotate(str(p.get_height()),(p.get_x(),p.get_height()*1.005))



sns.despine()  #to remove borders (by default : from top & right side)
sns.set(style='whitegrid')

ax=sns.kdeplot(train_df['Age'])

ax.set_title('Age Distribution aboard the Titanic')

ax.set_xlabel('<---AGE--->')
print(train_df['Survived'].value_counts())

l=['Not Survived','Survived']

ax=train_df['Survived'].value_counts().plot.pie(autopct='%.2f%%',figsize=(6,6),labels=l)

#autopct='%.2f%%' is to show the percentage text on the plot

ax.set_ylabel('')
sns.countplot(train_df['Pclass'])

sns.despine()
sns.countplot(train_df['Embarked'])
train_df[['Sex','Survived']].groupby('Sex').mean()
train_df[['Pclass','Survived']].groupby('Pclass').mean()
train_df.groupby(['Pclass','Survived'])['Pclass'].count()
sns.countplot(x='Pclass',hue='Survived',data=train_df)
train_df[['Embarked','Survived']].groupby('Embarked').mean()
train_df[['Parch','Survived']].groupby('Parch').mean()
train_df[['SibSp','Survived']].groupby('SibSp').mean()
ax=train_df[['Parch','Survived']].groupby('Parch').mean().plot.line(figsize=(8,4))

ax.set_ylabel('Survival')

sns.despine()
ax=train_df[['SibSp','Survived']].groupby('SibSp').mean().plot.line(figsize=(8,4))

ax.set_ylabel('Survival')

sns.despine()
a=sns.FacetGrid(train_df,col='Survived')

a.map(sns.distplot, 'Age')
a=sns.FacetGrid(train_df,col='Pclass',row='Survived')

a.map(plt.hist,'Age')
train_df['Embarked'].value_counts()
a=sns.FacetGrid(train_df,col='Embarked')

a.map(sns.distplot,'Survived')
train_df.groupby(['Embarked','Survived'])['Embarked'].count()
a=sns.FacetGrid(train_df,col='Embarked')

a.map(sns.pointplot, 'Pclass','Survived','Sex') #colum order is x='Pclass', y='Survived', hue='Sex'

a.add_legend()
train_df.groupby(['Embarked','Sex'])['Embarked'].count()
a=sns.FacetGrid(train_df,col='Survived')

a.map(sns.barplot,'Sex', 'Fare')
combined_df[0].head(3) #[0] is train_df
combined_df[1].head(3)  #[1] is test_df
print('training data dimensions :',train_df.shape)

print('testing data dimensions :', test_df.shape)

print('combined data\'s dimension are :\n',combined_df[0].shape,'\n',combined_df[1].shape)
train_df[['PassengerId','Name','Ticket','Cabin']].head()
#removing mentioned columns from dataset

train_df=train_df.drop(['Name','Ticket','Cabin','SibSp','Parch','PassengerId'],axis=1)

test_df=test_df.drop(['Name','Ticket','Cabin','SibSp','Parch'],axis=1)
# the combined data

combined_df=[train_df, test_df]
#lets check the new dimensions

print('new training data dimensions :',train_df.shape)

print('new testing data dimensions :', test_df.shape)

print('new combined data\'s dimension are :\n',combined_df[0].shape,'\n',combined_df[1].shape)
train_df.head(3)
#checking for any null values

train_df.isnull().any() #True means null present
test_df.isnull().any()
# age columns

print('mean age in train data :',train_df['Age'].mean())

print('mean age in test data :',test_df['Age'].mean())
#replacing null values with 30 in age column

for df in combined_df:

    df['Age']=df['Age'].replace(np.nan,30).astype(int)
train_df['Embarked'].value_counts()
#most people embarked from 'S'. So, we'll replace the missing missing Embarked value by 'S'.

train_df['Embarked']=train_df['Embarked'].replace(np.nan,'S')

# 年龄用平均值、上船用最多的替换缺失值，用最可能的值替换缺失值。
#finding mean fare in test data

test_df['Fare'].mean()
#replace missing fare values in test data by mean

test_df['Fare']=test_df['Fare'].replace(np.nan,36).astype(int)

# 票价的缺失值用平均值替换。
combined_df=[train_df,test_df]

for df in combined_df:

    print(df.isnull().any()) #bool value = False means that there are no nulls in the column.

# 处理完了缺失值以后检查是否全部处理完成了，再没有缺失值了。
#will code female as 1 and male as 0

for df in combined_df:

    df['Sex']=df['Sex'].map({'female':1,'male':0}).astype(int)
train_df.head(3)
#coding Embarked column as: S=2, C=1, Q=0

for df in combined_df:

    df['Embarked']=df['Embarked'].map({'S':2,'C':1,'Q':0}).astype(int)
train_df.head(3)
#binning or making bands of age into intervals and then assigning labels to them(encoding the bands as 0,1,2,3,4)

#将年龄段分为多个区间，然后为其分配标签（将年龄段编码为0、1、2、3、4）

for df in combined_df:

    df['Age']=pd.cut(df['Age'],5,labels=[0,1,2,3,4]).astype(int) #pandas cut will help us divide age in bins
train_df.head(3)
#binning fares and assigning label 0,1,2,3 to their respective bins

for df in combined_df:

    df['Fare']=pd.qcut(df['Fare'],4,labels=[0,1,2,3]).astype(int)
train_df.head(3)
test_df.head(3)
X_train=train_df.drop('Survived',axis=1)

Y_train=train_df['Survived']



#X_train is the entire training data except the Survived column, which is separately stored in Y_train. We will use these to train our MODEL !



X_test=test_df.drop('PassengerId',axis=1).copy()

#X_test is the test data, for on which we will apply model and predict the "SURVIVED" column for its entries.
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
#first applying Logistic Regression



lg = LogisticRegression()

lg.fit(X_train, Y_train)

Y_pred1 = lg.predict(X_test)

# accu_lg = (lg.score(X_train, Y_train))

accu_lg = lg.score(X_test, Y_pred1)

round(accu_lg*100,2)
#applying decision tree



dtree = DecisionTreeClassifier()

dtree.fit(X_train, Y_train)

Y_pred2 = dtree.predict(X_test)

accu_dtree = (dtree.score(X_train, Y_train))

round(accu_dtree*100,2)
#applying random forest



rafo = RandomForestClassifier(n_estimators=100)

rafo.fit(X_train, Y_train)

Y_pred3 = rafo.predict(X_test)

accu_rafo = rafo.score(X_train, Y_train)

round(accu_rafo*100,2)
rafo = RandomForestClassifier(n_estimators=1000, random_state=0)

rafo.fit(X_train, Y_train)

Y_pred4 = rafo.predict(X_test)

accu_rafo = rafo.score(X_train, Y_train)

round(accu_rafo*100,2)
# Y_pred4 == Y_pred3



from xgboost import XGBRegressor

rafo = XGBRegressor(n_estimators=1000, random_state=0)

rafo.fit(X_train, Y_train)

Y_pred5 = rafo.predict(X_test)

accu_rafo = rafo.score(X_train, Y_train)

round(accu_rafo*100,2)
# 模型使用神经网络深度学习多层感知机，数据使用之前的预处理。

from sklearn.neural_network import MLPClassifier



# mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, Y_train)

rafo = MLPClassifier(solver='lbfgs', max_iter=1000, random_state=0)

rafo.fit(X_train, Y_train)

Y_pred6 = rafo.predict(X_test)

accu_rafo = rafo.score(X_train, Y_train)

round(accu_rafo*100, 2)

# 已经达到了其它模型的最高得分。没有用上预测值！系统不给你test的目标值，上传数据后系统检查结果。2019.11.3
#our goal was to predict survived column for test data, and were asked to submit a dataframe with 'PassengerId' and 'Survived' columns



submission=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':Y_pred6})
submission.shape
submission.head(10)
submission.to_csv('submission.csv', index=False)
# 模型使用神经网络深度学习多层感知机，数据使用之前的预处理。

from sklearn.neural_network import MLPClassifier



# mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, Y_train)

mlp = MLPClassifier(solver='lbfgs', max_iter=1000, random_state=0).fit(X_train, Y_train)

Y_pred6 = mlp.predict(X_test)

accu_mlp = mlp.score(X_train, Y_train)

round(accu_mlp*100, 2)

# 已经达到了其它模型的最高得分。没有用上预测值！系统不给你test的目标值，上传数据后系统检查结果。2019.11.3
c = Y_pred6 == Y_pred5

# 神经网络与其它模型的预测结果对比的百分数

# 1.LogiticReg_84.69, 2.decitionTree_98.56, 3.RanomForest_95.53, 

# 4.RandomForest_estimator=1000_96.41, 5.xgboost_0

c[0]

j = 0

for i in c:

    if i ==True:

        j = j+1

print(len(c), j, round(j/len(c)*100,2))