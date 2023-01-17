import pandas as pd
# 爬取科比常规赛数据

# df = pd.read_html('http://www.stat-nba.com/query.php?page=0&QueryType=game&GameType=season&Player_id=195&crtcol=season&order=1#label_show_result',encoding='utf-8',index_col=0)[0]

# for page in range(67):

#     url = f'http://www.stat-nba.com/query.php?page={page+1}&QueryType=game&GameType=season&Player_id=195&crtcol=season&order=1#label_show_result'

#     temp = pd.read_html(url,encoding='utf-8',index_col=0)[0]

#     df = pd.concat([df,temp])
# df.to_csv('kobe_regular.csv',index=None)
# 爬取科比季后赛数据

# df_2 = pd.read_html('http://www.stat-nba.com/query.php?page=0&QueryType=game&GameType=playoff&Player_id=195&crtcol=season&order=1#label_show_result',encoding='utf-8',index_col=0)[0]

# for page in range(10):

#     url = f'http://www.stat-nba.com/query.php?page={page+1}&QueryType=game&GameType=playoff&Player_id=195&crtcol=season&order=1#label_show_result'

#     temp = pd.read_html(url,encoding='utf-8',index_col=0)[0]

#     df_2 = pd.concat([df_2,temp])
# df_2.to_csv('kobe_playoff.csv',index=None)
# 读取数据

import pandas as pd

df_r = pd.read_csv('../input/kobedata/kobe_regular.csv')

df_o = pd.read_csv('../input/kobedata/kobe_playoff.csv')
# 替换"结果"中的字符

df_r.replace({'胜':1,'负':0},inplace=True)

df_o.replace({'胜':1,'负':0},inplace=True)
# def convert_percent(x):

#     new = x.str.replace('%','')

#     return float(new)/100

# df['投篮'].apply(convert_percent)

# 不知道为什么这种替换方式行不通
# 将字符串转换为float

df_r[['投篮','三分','罚球']]=df_r[['投篮','三分','罚球']].apply(lambda x:x.str.replace('%', '').astype('float')/100)

df_o[['投篮','三分','罚球']]=df_o[['投篮','三分','罚球']].apply(lambda x:x.str.replace('%', '').astype('float')/100)
#查看缺失数据

df_r.isnull().values.sum() 

df_o.isnull().values.sum()
df_r[df_r.isnull().any(axis=1)].iloc[:,7:].head(10) 
import re 



# 将“比赛”列拆分为对阵双方

df_r_score= df_r['比赛'].str.split('-',1,expand=True).rename(columns={0:'对手',1:'湖人'})

df_o_score= df_o['比赛'].str.split('-',1,expand=True).rename(columns={0:'对手',1:'湖人'})

# 提取湖人比分

df_r_score['湖人得分']=df_r_score['湖人'].str.extract(r'(\d+)',expand=True)

df_o_score['湖人得分']=df_o_score['湖人'].str.extract(r'(\d+)',expand=True)

# 将对手比分和队名拆分，其中队名中“76人”既有数字又有字符，因此使用正则表达式

rival = []

for i in range(len(df_r)):

    tem = re.split(r'(\d+\D|\D+)',df_r_score['对手'][i],maxsplit=1)

    rival.append(tem)

# 合并两队数据，删除无用列

df_r_score = df_r_score.join(pd.DataFrame(rival)).drop(['对手','湖人',0],axis=1).rename(columns={1:'对手',2:'对手得分'})



rival = []

for i in range(len(df_o)):

    tem = re.split(r'(\d+\D|\D+)',df_o_score['对手'][i],maxsplit=1)

    rival.append(tem)

# 合并两队数据，删除无用列

df_o_score = df_o_score.join(pd.DataFrame(rival)).drop(['对手','湖人',0],axis=1).rename(columns={1:'对手',2:'对手得分'})
df_r_score.head()
df_r.columns
df_o.columns
# 删除无用列，重新命名列名

df_r.drop(['球员','比赛'],axis=1,inplace=True)

df_o.drop(['球员','比赛'],axis=1,inplace=True)

df_r.columns = ['赛季', '结果', '首发', '出场时间', '投篮命中率', '投篮命中', '投篮出手', '三分命中率', '三分命中', '三分出手',

       '罚球命中率', '罚球命中', '罚球出手', '篮板', '前场', '后场', '助攻', '抢断', '盖帽', '失误', '犯规',

       '得分']

df_o.columns = ['赛季', '结果', '出场时间', '投篮命中率', '投篮命中', '投篮出手', '三分命中率', '三分命中', '三分出手',

       '罚球命中率', '罚球命中', '罚球出手', '篮板', '前场', '后场', '助攻', '抢断', '盖帽', '失误', '犯规',

       '得分']

df_r.head()
# 倒序，重置索引

df_r.sort_index(ascending=False,inplace=True)

df_o.sort_index(ascending=False,inplace=True)

df_r.reset_index(drop=True,inplace=True)

df_o.reset_index(drop=True,inplace=True)
# 将赛季信息按原来的顺序组成含有唯一值的列表

df_index = list(df_r['赛季'])

t = list(set(df_index))

t.sort(key=df_index.index)
# 生成场次的序列，并添加至原数据

import  numpy as np

game = []

for session in t:

    length = len(df_r[df_r['赛季']==session])

    tem = np.array([i+1 for i in range(length)])

    game = np.concatenate((game,tem))

df_r['场次'] = pd.DataFrame(game.astype(int))



game = []

for session in t:

    length = len(df_o[df_o['赛季']==session])

    tem = np.array([i+1 for i in range(length)])

    game = np.concatenate((game,tem))



df_o['场次'] = pd.DataFrame(game.astype(int))
df_r[df_r['赛季']=='98-99'].head(10)

# df_o[df_o['赛季']=='98-99'].head(10)
df_r.to_csv('kobe_regular_.csv',index=None)

df_o.to_csv('kobe_playoff_.csv',index=None)
import pandas as pd
# 读取数据

df_r = pd.read_csv('kobe_regular_.csv')

df_o = pd.read_csv('kobe_playoff_.csv')

df_r.head()
import warnings

warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt

%matplotlib inline

plt.rcParams['font.sans-serif']=['SimHei']   #加载本地字体以显示中文(kaggle中没显示中文，因此之后涉及到文字的修改为英文)
# 按序排列的赛季序列

df_index = list(df_r['赛季'])

session_list = list(set(df_index))

session_list.sort(key=df_index.index)
# 计算科比的常规赛和季后赛胜率

win_r_rate = len(df_r[df_r['结果']==1])/len(df_r)

win_o_rate = len(df_o[df_o['结果']==1])/len(df_o)

print(win_r_rate,win_o_rate)
# 计算各赛季的胜率

win_r_single_rate = []

for session in session_list:

    rate = df_r.groupby(by=['赛季','结果'],sort=False).count().loc[session,1][0]/df_r.groupby(by=['赛季'],sort=False).count().loc[session,:][0]

    win_r_single_rate.append(rate)

# win_r_single_rate



win_o_single_rate = []

for session in session_list:

    if session in ['04-05','12-13','13-14','14-15','15-16']:    #该赛季未进季后赛

        win_o_single_rate.append(0)

    else:

        rate = df_o.groupby(by=['赛季','结果'],sort=False).count().loc[session,1][0]/df_o.groupby(by=['赛季'],sort=False).count().loc[session,:][0]

        win_o_single_rate.append(rate)

# win_o_single_rate
# 绘图查看科比职业生涯的胜率变化

fig, ax = plt.subplots(figsize=(12,6))

# ax.plot(session_list,win_r_single_rate,'.-',label='季后赛胜率')

# ax.plot(session_list,len(session_list)*[win_r_rate],'r',label='生涯胜率_常')

# ax.plot(session_list,win_o_single_rate,'.-',label='季后赛胜率')

# ax.plot(session_list,len(session_list)*[win_o_rate],label='生涯胜率_后')

# ax.set_xlabel('赛季')

# ax.set_ylabel('胜率')



ax.plot(session_list,win_r_single_rate,'.-',label='regular')

ax.plot(session_list,len(session_list)*[win_r_rate],'r',label='career_regular')

ax.plot(session_list,win_o_single_rate,'.-',label='playoff')

ax.plot(session_list,len(session_list)*[win_o_rate],label='career_playoff')

ax.set_xlabel('session')

ax.set_ylabel('winrate')

plt.legend()
# 计算出勤率

attendance = []

for session in session_list:

    if session=='98-99':

        rate = df_r.groupby(by=['赛季'],sort=False).count().loc[session,:][0]/50

    elif session == '11-12':

        rate = df_r.groupby(by=['赛季'],sort=False).count().loc[session,:][0]/66

    else:

        rate = df_r.groupby(by=['赛季'],sort=False).count().loc[session,:][0]/82

    attendance.append(rate)

attendance_mean = np.array(attendance).mean()

# attendance

attendance_mean
fig,ax = plt.subplots(figsize=(12,6))

# ax.plot(session_list,attendance,'-o',label='赛季出勤')

# ax.plot(session_list,len(session_list)*[attendance_mean],label='生涯出勤')



ax.plot(session_list,attendance,'-o',label='single')

ax.plot(session_list,len(session_list)*[attendance_mean],label='career')

ax.set_xlabel('Attendance')

ax.set_ylabel('session')

plt.legend()
# 合并常规赛和季后赛的数据，方便可视化展示

df_all = pd.concat([df_r.drop(['首发'],axis=1),df_o],keys=['常规赛','季后赛'])

df_all.index.names = ['类型','x']

df_all = df_all.reset_index(['类型','x']).drop('x',axis=1)

df_all.columns
df_all = df_all.rename(columns={'类型':'type','赛季':'session','投篮命中率':'field_goal','三分命中率':'three_goal','罚球命中率':'free_goal'})
df_all.replace({'常规赛':'regular','季后赛':'playoff'},inplace=True)
import seaborn as sns



fig,ax = plt.subplots(3,1,figsize=(12,18))

pos = list(range(len(session_list)))



sns.boxplot(x='session',y='field_goal',hue='type',data=df_all,ax=ax[0],palette="Set3")

sns.boxplot(x='session',y='three_goal',hue='type',data=df_all,ax=ax[1],palette="Set3")

sns.boxplot(x='session',y='free_goal',hue='type',data=df_all,ax=ax[2],palette="Set3")



plt.show()
games = pd.DataFrame(df_r.groupby(by=['赛季'],sort=False).count().iloc[:,0])

games.columns=['场次']

games.sort_values('场次',ascending=False)
# 插入数据

import random

miss = df_r.loc[0]

miss['得分'] = np.nan   # 需要插入的缺失数据

df_r_insert = df_r.copy()

for session in session_list:

    miss['赛季'] = session

    k = 82- games.loc[session][0]

    insert_index = random.choices(df_r_insert[df_r_insert['赛季']==session].index,k=k)

    df_values = np.insert(df_r_insert.values,insert_index,miss,axis=0)

    df_r_insert = pd.DataFrame(df_values,columns=df_r_insert.columns)



#df_r_insert.groupby(by=['赛季']).count()[0]   #查看是否插入正确
df_r_insert.shape
date = pd.date_range('1996','2016',periods=82*20)  # 添加时间索引

df_r_insert.set_index(date,inplace=True)
session_drop = ['98-99','11-12','13-14','14-15','15-16']  #需要删除的赛季数据

df_r_drop = df_r_insert[-df_r_insert.赛季.isin(session_drop)]



df_r_drop.shape
# 生成标准数据

df_r_drop = df_r_drop.reset_index().rename(columns={'index':'时间'})

data = df_r_drop[['时间','得分']]

data.columns=['ds','y']

data
365/82
import fbprophet

from fbprophet.plot import add_changepoints_to_plot

model = fbprophet.Prophet()



model.fit(data)

future = model.make_future_dataframe(periods=82*3,freq='4.45D')

forcast = model.predict(future)
fig = model.plot(forcast)

a = add_changepoints_to_plot(fig.gca(),model,forcast)
# 对比13-14，14-15，15-16赛季的预测数据与真实数据

score_f = []

str = ['2013','2014','2015','2016']

for i in range(3):

    score_f.append(forcast[(forcast['ds']>str[i])&(forcast['ds']<str[i+1])]['yhat'].mean())



score_a = []

periods = ['13-14','14-15','15-16']

for i in range(3):

    score_a.append(df_r[df_r['赛季']==periods[i]]['得分'].mean())

print(score_a)

print(score_f)
from matplotlib import pyplot as plt

%matplotlib inline

fig,ax = plt.subplots(figsize=(12,6))

ax.plot(periods,score_f,'o-',label='forcast')

ax.plot(periods,score_a,'*-',label='actual')

ax.set_xlabel('session')

ax.set_ylabel('score')

plt.show()
# 归一化

def max_min_scaler(x): return (x-np.min(x))/(np.max(x)-np.min(x))



df_min_max = df_r[['篮板','助攻','抢断','盖帽','失误','犯规','得分','出场时间','投篮命中率','三分命中率','罚球命中率']].apply(max_min_scaler)

target = pd.DataFrame(df_r['结果'])



df_min_max = df_min_max.join(target).dropna()  #删除缺失数据
# 数据集分类

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df_min_max.iloc[:,:-1],df_min_max.iloc[:,-1],test_size=0.3,random_state=12)
# 建立SVM模型

import warnings

warnings.filterwarnings('ignore')

from sklearn.svm import SVC

model_svc = SVC(C=300)

model_svc.fit(X_train,y_train)
from sklearn.metrics import accuracy_score



# 训练集的正确率

y_train_a = model_svc.predict(X_train)

accuracy_score(y_train,y_train_a)
# 测试集的正确率

model_svc.score(X_test,y_test)    #相同的random_state，在本地测试集正确率为68%，但是kaggle上不到60%。
# 测试集的预测结果

model_svc.predict(X_test)