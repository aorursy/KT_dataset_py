##使いそうなやつをimport

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate,train_test_split

from sklearn import svm, neighbors, datasets

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier

from sklearn.metrics import accuracy_score

#値を読み込む

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

#楽するためにtrainとtestのデータを連結

test['Survived'] = np.nan

df = pd.concat([train,test], ignore_index=True,sort=False)

##データの確認Ϟໂ‧͡‧̫ໃ

print(df.info())

##欠損値の確認Ϟໂ‧͡‧̫ໃ

print(df.isnull().sum())

##データの形を確認Ϟໂ‧͡‧̫ໃ

print(df.head())
#---------------- Pclass ----------------

##確認Ϟໂ‧͡‧̫ໃ

sns.countplot(df.Pclass , hue = df.Survived, palette='spring')
#---------------- Sex ----------------

##確認Ϟໂ‧͡‧̫ໃ

sns.countplot(df.Sex , hue = df.Survived, palette='spring')
#---------------- Embarked ----------------

#Embarkedの欠損値を最頻値へ変換する

df.Embarked = df.Embarked.fillna('S')

##確認Ϟໂ‧͡‧̫ໃ

sns.countplot(df.Embarked , hue = df.Survived, palette='spring')
#---------------- Fare ----------------

#Fareの欠損値を中間値へ変換する

df.Fare = df.Fare.fillna(df.Fare.median())

##確認Ϟໂ‧͡‧̫ໃ

fig = sns.FacetGrid(df[0:890], hue = 'Survived' ,aspect=2)

fig.map(sns.kdeplot, 'Fare' ,shade= True)

fig.set(xlim=(0,df.Fare.max()))

fig.add_legend()
#---------------- Name ----------------

#敬称(Title)と家族の構成人数(Group)を抽出

df['Title'] = df.Name.map(lambda x: x.split(', ')[1].split('. ')[0])

#df['Group'] = df.Name.map(lambda x: x.split(', ')[0])

#df.Group = df.Group.map(df.Group.value_counts()) 

##確認Ϟໂ‧͡‧̫ໃ

#cross = pd.crosstab(df.Title, df.Survived, normalize='index')

#cross.plot.bar(stacked=True, figsize=(12, 2), color=['maroon', 'pink'])

##敬称の男女比を確認Ϟໂ‧͡‧̫ໃ

#cross = pd.crosstab(df.Title, df.Sex, normalize='index')

#cross.plot.bar(stacked=True, figsize=(12, 2), color=['skyblue', 'pink'])
#SibSpとParchの足し合わせる

df['Family'] = pd.Series(df.Parch + df.SibSp + 1)

##確認Ϟໂ‧͡‧̫ໃ

#sns.countplot(df.Family , hue = df.Survived, palette='spring')
# 年齢の欠損値を他のデータから推測して補完する

age_df = df[['Age','Pclass','Sex','Title','Embarked','Fare']]

# ラベル特徴量をワンホットエンコーディング

age_df=pd.get_dummies(age_df)

# 学習データとテストデータに分離し、numpyに変換

known_age = age_df[age_df.Age.notnull()].values  

unknown_age = age_df[age_df.Age.isnull()].values

# 学習データをX, yに分離

X = known_age[:, 1:]  

y = known_age[:, 0]

# ランダムフォレストで推定モデルを構築

rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)

rfr.fit(X, y)

# 推定モデルを使って、テストデータのAgeを予測し、補完

predictedAges = rfr.predict(unknown_age[:, 1::])

df.loc[(df.Age.isnull()), 'Age'] = predictedAges 

# 年齢別生存曲線と死亡曲線

facet = sns.FacetGrid(df[0:890], hue="Survived",aspect=2)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, df.loc[0:890,'Age'].max()))

facet.add_legend()

plt.show()
#Cabinの欠損値が多すぎるから、扱いを考えるためにまずは頭文字ごとに仕分けをする

df.Cabin = df.Cabin.map(lambda x:str(x)[0])

#次にCabinのデータを全て数値に変換する（ここ綺麗に描きたい）(欠損値をそのまま扱う）

#df.Cabin = df.Cabin.replace(['A','B','C','D','E','F','G','T','n'], [1,2,3,4,5,6,7,8,0])

##確認Ϟໂ‧͡‧̫ໃ

#sns.countplot(df.Cabin , hue = df.Survived,palette='spring')
#同じチケット番号の人が何人いるかで生存率を確認Ϟໂ‧͡‧̫ໃ

Ticket_count = dict(df.Ticket.value_counts())

df['TicketGroup'] = df.Ticket.map(Ticket_count)

sns.barplot(x='TicketGroup', y='Survived', data=df, palette='spring')

plt.show()
# ------------- 前処理 ---------------

#要素をまとめる

df.Title.replace(['Col', 'Dr'], 'Officer', inplace=True)

df.Title.replace(['the Countess', 'Dona'], 'Royalty', inplace=True)

df.Title.replace(['Ms'], 'Mrs', inplace=True)

df.Title.replace(['Jonkheer','Capt','Lady','Don','Mlle','Mme','Major','Sir'],'others',inplace=True)

df.Cabin.replace('T','n',inplace=True)

# 推定に使用する項目を指定

df = df[['Survived','Title','Family','Embarked','Fare','Sex','Age','Cabin','Pclass','TicketGroup']]

# ラベル特徴量をワンホットエンコーディングしておく

df = pd.get_dummies(df)

df.drop('Title_others',axis=1)

# dfをtrainとtestに分割する

train = df[df['Survived'].notnull()]

test_x = df[df['Survived'].isnull()].drop('Survived',axis=1)

print(df.columns)

# データフレームをnumpyに変換

X = train.values[:,1:]  

y = train.values[:,0] 

test_x = test_x.values
Acc=[]

for i in range(4):

    acc=[]

    for j in range(3,7):

        clf = RandomForestClassifier(random_state = 5, 

                                     warm_start = True,  # 既にフィットしたモデルに学習を追加 

                                     n_estimators = 4**(i+2),

                                     max_depth = j, 

                                     max_features = 'sqrt')

        clf.fit(X, y)



        # フィット結果の表示

        cv_result = cross_validate(clf, X, y, cv= 10)

        print('n_estimators = ', 4**(i+2), 'max_depth = ', j)

        print('mean_score = ', np.mean(cv_result['test_score']))

        print('mean_std = ', np.std(cv_result['test_score']))

        

        acc.append(np.mean(cv_result['test_score']))

        

    Acc.append(acc)

    

    
x_datas = range(1,5)

plt.plot(x_datas, Acc[0], marker = 'o')

plt.plot(x_datas, Acc[1], marker = 'x')

plt.plot(x_datas, Acc[2], marker = '*')

plt.plot(x_datas, Acc[3], marker = 'D')



plt.show()
#使うパラメータを決定する

i = 2

j = 5

#予測データとPassengerIdをデータフレームへ落とし込む

pred = clf.predict(test_x)

PassengerId = test.PassengerId

submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": pred.astype(np.int32)})

submission.to_csv("submmision.csv", index=False)
#決定木モデルの作成

clf1 = DecisionTreeClassifier(max_depth=8)

clf1 = clf1.fit(X, y)

#交差検証

score1 = cross_validate(clf1, X, y, cv= 10)

print('mean_score1 = ', np.mean(score1['test_score']))



#次にランダムフォレスト

forest = RandomForestClassifier(n_estimators = 200, max_depth=6)

forest = forest.fit(X, y)

#交差検証

score2 = cross_validate(forest, X, y, cv= 10)

print('mean_score2 = ', np.mean(score2['test_score']))



#勾配ブースティング

gb = GradientBoostingClassifier(n_estimators=1000, random_state=9)

gb = gb.fit(X, y)

#交差検証

score3 = cross_validate(gb, X, y, cv= 10)

print('mean_score3 = ', np.mean(score3['test_score']))
#予測

pred1 = clf1.predict(test_x)

pred2 = forest.predict(test_x)

pred3 = gb.predict(test_x)

#予測データとPassengerIdをデータフレームへ落とし込む

PassengerId = test.PassengerId

submission1 = pd.DataFrame({"PassengerId": PassengerId, "Survived": pred1.astype(np.int32)})

submission1.to_csv("submission1.csv", index=False)

submission2 = pd.DataFrame({"PassengerId": PassengerId, "Survived": pred2.astype(np.int32)})

submission2.to_csv("submission2.csv", index=False)

submission3 = pd.DataFrame({"PassengerId": PassengerId, "Survived": pred3.astype(np.int32)})

submission3.to_csv("submission3.csv", index=False)
print(submission)