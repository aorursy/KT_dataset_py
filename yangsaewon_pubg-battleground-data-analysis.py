import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
# 데이터 불러오기

train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
train.info()
# 상위 5개 row들

train.head()
print("유저는 평균적으로 {:.4f}명의 킬을 하며, 99%의 유저들은 {} 이하의 킬을 한다. 반면 최대 많이 킬을 한 횟수는 {}이다.".format(train['kills'].mean(),train['kills'].quantile(0.99), train['kills'].max()))
# 킬 카운트를 시각화 해보자.

data = train.copy()

data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'

plt.figure(figsize=(15,10))

sns.countplot(data['kills'].astype('str').sort_values())

plt.title("Kill Count",fontsize=15)

plt.show()
data = train.copy()

data = data[data['kills']==0]

plt.figure(figsize=(15,10))

plt.title("Damage Dealt by 0 killers",fontsize=15)

sns.distplot(data['damageDealt'])

plt.show()
print("{}의 유저({:.4f}%)는 킬을 하지도 않고도 우승했다".format(len(data[data['winPlacePerc']==1]), 100*len(data[data['winPlacePerc']==1])/len(train)))



data1 = train[train['damageDealt'] == 0].copy()

print("{}의 유저({:.4f}%)는 데미지를 가하지 않고 우승했다".format(len(data1[data1['winPlacePerc']==1]), 100*len(data1[data1['winPlacePerc']==1])/len(train)))
sns.jointplot(x="winPlacePerc", y="kills", data=train, height=10, ratio=3, color="r")

plt.show()
kills = train.copy()



kills['killsCategories'] = pd.cut(kills['kills'], [-1, 0, 2, 5, 10, 60], labels=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills'])



plt.figure(figsize=(15,8))

sns.boxplot(x="killsCategories", y="winPlacePerc", data=kills)

plt.show()
print("평균적으로 유저는 {:.1f}m를 뛰며, 99%의 유저는 {}m 이하를 뛰었다. 반면 마라톤 챔피언은 {}m를 뛰었다.".format(train['walkDistance'].mean(), train['walkDistance'].quantile(0.99), train['walkDistance'].max()))
data = train.copy()

data = data[data['walkDistance'] < train['walkDistance'].quantile(0.99)]

plt.figure(figsize=(15,10))

plt.title("Walking Distance Distribution",fontsize=15)

sns.distplot(data['walkDistance'])

plt.show()
print("{}의 유저({:.4f}%)는 0미터를 걸었다. 이것이 의미하는 것은 유저가 발을 딛기 전에도 죽었다는 것이다.".format(len(data[data['walkDistance'] == 0]), 100*len(data1[data1['walkDistance']==0])/len(train)))
sns.jointplot(x="winPlacePerc", y="walkDistance",  data=train, height=10, ratio=3, color="lime")

plt.show()
print("평균적으로 유저는 {:.1f}m를 운전하며, 99%의 유저는 {}m 이하를 운전했다. 반면 최장거리 운전자는 {}m를 운전했다.".format(train['rideDistance'].mean(), train['rideDistance'].quantile(0.99), train['rideDistance'].max()))
data = train.copy()

data = data[data['rideDistance'] < train['rideDistance'].quantile(0.9)]

plt.figure(figsize=(15,10))

plt.title("Ride Distance Distribution",fontsize=15)

sns.distplot(data['rideDistance'])

plt.show()
print("{}의 유저({:.4f}%)는 0미터를 운전했다. 이 뜻은 그들이 아직 운전면허가 없다는 말이다.".format(len(data[data['rideDistance'] == 0]), 100*len(data1[data1['rideDistance']==0])/len(train)))
sns.jointplot(x="winPlacePerc", y="rideDistance", data=train, height=10, ratio=3, color="y")

plt.show()
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='vehicleDestroys',y='winPlacePerc',data=data,color='#606060',alpha=0.8)

plt.xlabel('Number of Vehicle Destroys',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.title('Vehicle Destroys/ Win Ratio',fontsize = 20,color='blue')

plt.grid()

plt.show()
print("유저는 평균적으로 {:.1f}m를 수영하며, 99%의 유저는 {}m이하를 수영했다. 반면 수영 챔피언은 {}m를 수영했다.".format(train['swimDistance'].mean(), train['swimDistance'].quantile(0.99), train['swimDistance'].max()))
data = train.copy()

data = data[data['swimDistance'] < train['swimDistance'].quantile(0.95)]

plt.figure(figsize=(15,10))

plt.title("Swim Distance Distribution",fontsize=15)

sns.distplot(data['swimDistance'])

plt.show()
swim = train.copy()



swim['swimDistance'] = pd.cut(swim['swimDistance'], [-1, 0, 5, 20, 5286], labels=['0m','1-5m', '6-20m', '20m+'])



plt.figure(figsize=(15,8))

sns.boxplot(x="swimDistance", y="winPlacePerc", data=swim)

plt.show()
print("평균적으로 유저는 {:.1f}개의 치료 아이템을 사용하며, 99%의 유저는 {}개 이하를 사용한다. 반면 가장 많이 치료 아이템을 사용한 유저는 {}개를 사용했다.".format(train['heals'].mean(), train['heals'].quantile(0.99), train['heals'].max()))

print("평균적으로 유저는 {:.1f}개의 부스트 아이템을 사용하며, 99%의 유저는 {}개 이하를 사용한다. 반면 가장 많이 부스트 아이템을 사용한 유저는 {}개를 사용했다.".format(train['boosts'].mean(), train['boosts'].quantile(0.99), train['boosts'].max()))
data = train.copy()

data = data[data['heals'] < data['heals'].quantile(0.99)]

data = data[data['boosts'] < data['boosts'].quantile(0.99)]



f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='heals',y='winPlacePerc',data=data,color='lime',alpha=0.8)

sns.pointplot(x='boosts',y='winPlacePerc',data=data,color='blue',alpha=0.8)

plt.text(4,0.6,'Heals',color='lime',fontsize = 17,style = 'italic')

plt.text(4,0.55,'Boosts',color='blue',fontsize = 17,style = 'italic')

plt.xlabel('Number of heal/boost items',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.title('Heals vs Boosts',fontsize = 20,color='blue')

plt.grid()

plt.show()
sns.jointplot(x="winPlacePerc", y="heals", data=train, height=10, ratio=3, color="lime")

plt.show()
sns.jointplot(x="winPlacePerc", y="boosts", data=train, height=10, ratio=3, color="blue")

plt.show()
solos = train[train['numGroups']>50]

duos = train[(train['numGroups']>25) & (train['numGroups']<=50)]

squads = train[train['numGroups']<=25]

print("{} ({:.2f}%) 솔로 게임이 존재하며, {} ({:.2f}%) 듀오 게임이, 그리고 {} ({:.2f}%) 스쿼드 게임이 존재한다.".format(len(solos), 100*len(solos)/len(train), len(duos), 100*len(duos)/len(train), len(squads), 100*len(squads)/len(train),))
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='kills',y='winPlacePerc',data=solos,color='black',alpha=0.8)

sns.pointplot(x='kills',y='winPlacePerc',data=duos,color='#CC0000',alpha=0.8)

sns.pointplot(x='kills',y='winPlacePerc',data=squads,color='#3399FF',alpha=0.8)

plt.text(37,0.6,'Solos',color='black',fontsize = 17,style = 'italic')

plt.text(37,0.55,'Duos',color='#CC0000',fontsize = 17,style = 'italic')

plt.text(37,0.5,'Squads',color='#3399FF',fontsize = 17,style = 'italic')

plt.xlabel('Number of kills',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.title('Solo vs Duo vs Squad Kills',fontsize = 20,color='blue')

plt.grid()

plt.show()
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='DBNOs',y='winPlacePerc',data=duos,color='#CC0000',alpha=0.8)

sns.pointplot(x='DBNOs',y='winPlacePerc',data=squads,color='#3399FF',alpha=0.8)

sns.pointplot(x='assists',y='winPlacePerc',data=duos,color='#FF6666',alpha=0.8)

sns.pointplot(x='assists',y='winPlacePerc',data=squads,color='#CCE5FF',alpha=0.8)

sns.pointplot(x='revives',y='winPlacePerc',data=duos,color='#660000',alpha=0.8)

sns.pointplot(x='revives',y='winPlacePerc',data=squads,color='#000066',alpha=0.8)

plt.text(14,0.5,'Duos - Assists',color='#FF6666',fontsize = 17,style = 'italic')

plt.text(14,0.45,'Duos - DBNOs',color='#CC0000',fontsize = 17,style = 'italic')

plt.text(14,0.4,'Duos - Revives',color='#660000',fontsize = 17,style = 'italic')

plt.text(14,0.35,'Squads - Assists',color='#CCE5FF',fontsize = 17,style = 'italic')

plt.text(14,0.3,'Squads - DBNOs',color='#3399FF',fontsize = 17,style = 'italic')

plt.text(14,0.25,'Squads - Revives',color='#000066',fontsize = 17,style = 'italic')

plt.xlabel('Number of DBNOs/Assits/Revives',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.title('Duo vs Squad DBNOs, Assists, and Revives',fontsize = 20,color='blue')

plt.grid()

plt.show()
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
k = 5 # 히트맵을 위한 변수의 개수

f,ax = plt.subplots(figsize=(11, 11))

cols = train.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = ['winPlacePerc', 'walkDistance', 'boosts', 'weaponsAcquired', 'damageDealt', 'killPlace']

sns.pairplot(train[cols], size = 2.5)

plt.show()