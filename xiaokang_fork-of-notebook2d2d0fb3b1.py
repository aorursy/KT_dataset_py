import numpy as np #调用numpy库并命名为 np

import pandas as pd #.....



from subprocess import check_output  #check 调用是否正确的模块

print(check_output(["ls", "../input"]).decode("utf8"))#支持utf-8格式输出所调用的CSV文件名，检查输出。

#强调地二个CSV文件只包括了两个id信息，所以只调用，后面不做分析。
import random as rad#调用随机数模块，命名为rad

import seaborn as sns#seaborn模块，本质是为了Python可视化跟好看，更柔和。优化了matplotlib模块可视化，

import matplotlib.pyplot as plt#调用matplotlib可视化模块
Gamexx= pd.read_csv('../input/gameinfo.csv')

print(Gamexx.columns.values)
print(Gamexx.head())#输出frist csv的前五行info
Gamexx.tail()#输出尾部，这是直接输出，没通过print输出模块输出。
Gamexx.head()
Gamexx.info()#info（表基本信息）。
Gamexx.describe()#describe()函数对于数据的快速统计汇总：只对int or double。。。等数值类型统计
#<===

#这里我们可以初步分析得知  @gameid总数为10000个，@平均游戏id为5.738569e+07，@在录黑棋方平均ELO为1257.953400，

#...@在录红色方平均ELO为1261.086800，@std为标准方差，反应对应列的数据离散程度， 。。。。。。。。略去一大截。。- -。
Gamexx.describe(include=['object'])#仅对是对象类型的数据做一个数学统计描述。此处统计没实际性作用。
print(Gamexx.columns)

Gamexx["blackELO"].hist()

plt.title("ELO-Winner_conect") # add a title

plt.xlabel("blackELO") # label the x axes 

plt.ylabel("winner") # label the y a xes

plt.show()
Gamexx["redELO"].hist()

plt.title("ELO-Winner_conect") # add a title

plt.xlabel("redELO") # label the x axes 

plt.ylabel("winner") # label the y a xes

plt.show()
#其实由上面两个图就已经初步知道ELO（象棋一个等级经验评定制度的评定数）在ELO=1200时获胜者最多，
Gamexx["blackELO"].hist()

Gamexx["redELO"].hist()#两个图重叠式画出，有一点冒出的地方，是应为两图数据基本持平
#箱型图，操作对象是数据（int,double....）

fig = plt.figure()

ax = fig.add_subplot(111)

ax.boxplot(Gamexx['gameID'])

plt.show()
fig= plt.figure()

ax= fig.add_subplot(111)

ax.scatter(Gamexx['redELO'], Gamexx['winner'])

plt.show()
var = Gamexx.groupby(['redELO']).sum().stack()

temp = var.unstack()

type(temp)

x_list = temp['winner']

label_list = temp.index

plt.axis('equal')

plt.pie(x_list, labels=label_list, autopct='%1.1f%%')

plt.title('expense')

plt.show()