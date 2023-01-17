import os
import pandas as pd
import numpy as np

NUM = 51
data_path = r'../input'
trans = np.full([NUM,NUM], np.inf)
path = pd.read_table(os.path.join(data_path,'task-must-done-update/path.txt'),sep = ' ')

for _,b,e,d in path.itertuples():
    trans[b][e] = d+1  #初始化邻接表
for ind in range(trans.shape[0]):
    trans[ind,ind] = 0

# Dijkstra

begin = 1 #起始点
end = 10 #终止点

def Dijkstra(begin, end, trans):
    LEN = trans.shape[0]
    visited = np.full(LEN,False)#用于记录访问
    distance = np.full(LEN,np.Inf)#用于记录聚类
    path = np.full(LEN,-1)#用于记录路径
    
    #初始化起点
    visited[begin] = True
    distance[begin] = 0
    for ind in range(LEN):
        if trans[begin,ind] < distance[ind]:
            distance[ind] = trans[begin,ind] #若可达则替换
            path[ind] = begin #可由begin到达
    
    # 选择一个顶点进行访问
    for v in range(LEN):
        # 如果访问过或者不可达，则跳过该点
        if (visited[v]==True) | (distance[v]==np.inf):
            continue
            
        # 经过这个定点给到其他未访问点的距离变化情况
        for ind in range(LEN):
            if distance[v]+trans[v,ind] < distance[ind]:#如果经由v点距离缩短
                distance[ind] = distance[v]+trans[v,ind]#更改距离向量
                path[ind] = v#更改路径向量
                
    #得到最短路径长度            
    dist = distance[end]
    
    #求最短路径
    inv_path = [end]
    while True:
        if path[end] == -1:
            break
        inv_path.append(path[end])
        end = path[end]
        
    inv_path.reverse()
    return {
        'path':inv_path,
        'dist':dist-len(inv_path)+1 #减去每个有向边权值上加上的1
        }

Dijkstra(1, 10, trans)
import pandas as pd 
import numpy as np
import os
data_path = r'../input'
# 加载数据
locations = pd.read_table(os.path.join(data_path,'task-must-done/tsp.txt'),sep=' ',nrows=16)
locations = locations[['Position_X','Position_Y']]
LEN = locations.shape[0]
dis = np.zeros((LEN,LEN))

# 初始化邻接表
for i in range(LEN):
    for j in range(i):
        p1 = np.array(locations.iloc[i])
        p2 = np.array(locations.iloc[j])
        dis[j,i]=np.linalg.norm(p1-p2)
        dis[i,j]=dis[j,i]
    dis[i,i] = np.inf
# 贪婪算法 未使用
import pandas as pd 
import numpy as np
import copy
# 加载数据
locations = pd.read_table(os.path.join(data_path,'task-must-done/tsp.txt'),sep=' ',nrows=16)
locations = locations[['Position_X','Position_Y']]
LEN = locations.shape[0]
dis = np.zeros((LEN,LEN))

# 初始化邻接表
for i in range(LEN):
    for j in range(i):
        p1 = np.array(locations.iloc[i])
        p2 = np.array(locations.iloc[j])
        dis[j,i]=np.linalg.norm(p1-p2)
        dis[i,j]=dis[j,i]
    dis[i,i] = np.inf

def Greedy(dis,i):
    visited = np.full(len(dis),False)
    # 访问第一个点
    d = copy.deepcopy(dis[i,])
    cost = []
    path = [i]
    visited[i]=True
    
    while True:
        nextpoint = d.argmin()
        if visited[nextpoint]:
            d[nextpoint]=np.inf
            if all(visited):
                path.append(i)
                cost.append(d[0])
                return{
                    'cost':cost,
                    'path':path
                }
            continue
        visited[nextpoint] = True
        path.append(nextpoint)
        cost.append( d.min())
        d = copy.deepcopy(dis[nextpoint,])

# 从不同点出发的贪心
for i in range(len(dis)):
    path = np.full(len(dis)+1,np.nan)
    cost = np.inf
    res = Greedy(dis,i)
    if sum(res['cost']) < cost:
        cost = sum(res['cost'])
        path = res['path']
cost,path
# -*- coding: utf-8 -*-

import os

#返回当前工作目录
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)


loc = pd.read_table(os.path.join(data_path,'task-must-done/tsp.txt'),sep=' ',nrows=16)
loc = np.array(loc[['Position_X','Position_Y']])
 

def getdistmat(loc):
    '''返回距离阵'''
    num = loc.shape[0]
    distmat = np.zeros((len(loc), len(loc)))
    # 初始化生成距离矩阵
    for i in range(num):
        for j in range(i, num):
            distmat[i][j] = distmat[j][i] = np.linalg.norm(loc[i] - loc[j])
    return distmat
 
 
distmat = getdistmat(loc)
 
numant = 100  # 蚂蚁个数
numcity = loc.shape[0]
alpha = 1  # 信息素重要程度因子
beta = 5   # 启发函数重要程度因子
rho = 0.1  # 信息素的挥发速度
Q = 1      # 完成率
 
iter = 0       #迭代初始
itermax = 150  #迭代总数
 
etatable = 1.0 / (distmat + np.diag([1e10] * numcity))
pheromonetable = np.ones((numcity, numcity))# 信息素矩阵 16*16
pathtable = np.zeros((numant, numcity)).astype(int)# 路径记录表，转化成整型 100*16
distmat = getdistmat(loc)# 城市的距离矩阵 16*16
 
lengthaver = np.zeros(itermax)  # 迭代，存放每次迭代后，路径的平均长度  
lengthbest = np.zeros(itermax)  # 迭代，存放每次迭代后，最佳路径长度  
pathbest = np.zeros((itermax, numcity))  # 迭代，存放每次迭代后，最佳路径城市的坐标



###########################################################
while iter < itermax:#迭代总数
 
    #60个蚂蚁随机放置于16个城市中
    if numant <= numcity:  # 城市数比蚂蚁数多，不用管
        pathtable[:, 0] = np.random.permutation(range(numcity))[:numant]
    else:  # 蚂蚁数比城市数多，需要有城市放多个蚂蚁
        pathtable[:numcity, 0] = np.random.permutation(range(numcity))[:]
        pathtable[numcity:, 0] = np.random.randint(numcity,size= numant - numcity)
    length = np.zeros(numant)  
 
    #本段程序算出每只/第i只蚂蚁转移到下一个城市的概率
    for i in range(numant):
 
        visiting = pathtable[i, 0]  # 当前所在的城市
        unvisited = set(range(numcity))#未访问的城市集合
        unvisited.remove(visiting)  # 删除已经访问过的城市元素
 
        for j in range(1, numcity):  # 循环numcity-1次，访问剩余的所有numcity-1个城市
            # j=1
            # 每次用轮盘法选择下一个要访问的城市
            listunvisited = list(unvisited)
            #未访问城市数
            probtrans = np.zeros(len(listunvisited))
            #每次循环都初始化转移概率矩阵
 
 
            #以下是计算转移概率
            for k in range(len(listunvisited)):
                probtrans[k] = np.power(pheromonetable[visiting][listunvisited[k]], alpha) \
                               * np.power(etatable[visiting][listunvisited[k]], alpha)
            #eta-从城市i到城市j的启发因子 这是概率公式的分母   其中[visiting][listunvis[k]]是从本城市到k城市的信息素
            cumsumprobtrans = (probtrans / sum(probtrans)).cumsum()
            #求出本只蚂蚁的转移到各个城市的概率斐波衲挈数列
 
            cumsumprobtrans -= np.random.rand()# 随机生成下个城市的转移概率，再用区间比较
            k = listunvisited[list(cumsumprobtrans > 0).index(True)]# 函数选出符合cumsumprobtans>0的数
            
            pathtable[i, j] = k# 下一个要访问的城市
            unvisited.remove(k)#将未访问城市列表中的K城市删去，增加到已访问城市列表中
 
            length[i] += distmat[visiting][k]#计算本城市到K城市的距离
            visiting = k
 
        length[i] += distmat[visiting][pathtable[i, 0]]# 计算本只蚂蚁的总的路径距离，包括最后一个城市和第一个城市的距离
 
    # 包含所有蚂蚁的一个迭代结束后，统计本次迭代的若干统计参数
    
    #本轮的平均路径
    lengthaver[iter] = length.mean()
    
 
    #####本部分是为了求出最佳路径
 
    if iter == 0:
        lengthbest[iter] = length.min()
        pathbest[iter] = pathtable[length.argmin()].copy()#如果是第一轮路径，则选择本轮最短的路径,并返回索引值下标，并将其记录
    else:
    #后面几轮的情况，更新最佳路径
        if length.min() > lengthbest[iter - 1]:
            lengthbest[iter] = lengthbest[iter - 1]
            pathbest[iter] = pathbest[iter - 1].copy()
        else:
            lengthbest[iter] = length.min()
            pathbest[iter] = pathtable[length.argmin()].copy()
 
 
    #########此部分是为了更新信息素
    changepheromonetable = np.zeros((numcity, numcity))
    for i in range(numant):#更新所有的蚂蚁
        for j in range(numcity - 1):
            changepheromonetable[pathtable[i, j]][pathtable[i, j + 1]] += Q / distmat[pathtable[i, j]][pathtable[i, j + 1]]
            #根据公式更新本只蚂蚁改变的城市间的信息素      Q/d   其中d是从第j个城市到第j+1个城市的距离
        changepheromonetable[pathtable[i, j + 1]][pathtable[i, 0]] += Q / distmat[pathtable[i, j + 1]][pathtable[i, 0]]
        #首城市到最后一个城市 所有蚂蚁改变的信息素总和
 
    #信息素更新公式p=(1-挥发速率)*现有信息素+改变的信息素
    pheromonetable = (1 - rho) * pheromonetable + changepheromonetable
 
    iter += 1  # 迭代次数指示器+1
    
#迭代完成
 

 
# 作出找到的最优路径图
lengthbest[-1],pathbest[-1]

import os
import pandas as pd
import numpy as np
def Load_data():
    data_path = r'../input'
    #点
    V = pd.read_excel(os.path.join(data_path,'task-opt-done/map.xls'),sheet_name=0,header=1)
    V = V[['X坐标','Y坐标']]
    #边
    E = pd.read_excel(os.path.join(data_path,'task-opt-done/map.xls'),sheet_name=1,header=1)
    NUM = len(V)+len(E)#原有的点以及加的点
    Adj = np.full((NUM+1,NUM+1),np.inf)#以节点编号为标号
    Addpoint=[]
    #为边上插入点
    for i,e in enumerate(E.itertuples()):
        d = np.linalg.norm(V.iloc[e[1]-1]-V.iloc[e[2]-1])
        Adj[ e[1],i+len(V)+1 ] = d/2
        Adj[ i+len(V)+1,e[1] ] = d/2
        Adj[ e[2],i+len(V)+1 ] = d/2
        Adj[ i+len(V)+1,e[2] ] = d/2
        Addpoint.append([i+len(V)+1,*list((V.iloc[e[1]-1]+V.iloc[e[2]-1])/2),e[1],e[2]])
    P_add = np.vstack((np.array(V),np.array(Addpoint)[:,1:3]))

    nearest = []
    for p in [(5112,4806),(9126,4266),(7434 ,1332)]:
        nearest.append(np.sqrt(((P_add-p)**2).sum(1)).argmin())

    return Adj,Addpoint,nearest,P_add

Adj,addpoint,nearest,P_add = Load_data()

## Floyd
def Floyd(Adj):
    NUM=len(Adj)
    P = np.full((NUM,NUM),-1)
    Move = Adj.copy()
    for k in range(NUM):
        for i in range(NUM):
            for j in range(NUM):

                if(Move[i,j] > Move[i,k]+Move[k,j]):#两个顶点直接较小的间接路径替换较大的直接路径
                    P[i,j] = k                 #记录新路径的前驱
                    Move[i,j] = Move[i,k]+Move[k,j]
    return Move,P

Move,P = Floyd(Adj)
def Greedy(Move,nearest,n,v_car=40):
    Point =list(range(50))
    for k in range(n):
        point_set=[]
        #记录是否可以两分钟到达
        tag = np.full(3,False)
        visited = []
        visiting = list(range(len(Move)))
        visiting.pop(0)
        while len(visiting):
            p = random.choice(visiting)
            #访问节点p
            visiting.remove(p)
            visited.append(p)
            point_set.append(p)

            adj_p = [visiting[ind] for ind,Len in enumerate(Move[p,visiting]) if Len<v_car*3*1000/60]
            visited.extend(adj_p)
            for p in adj_p:
                visiting.remove(p)

            tag |= Move[p,nearest]<v_car*2*1000/60
            if (len(visited)/len(Move)>=0.9) & all(tag):
                break
        for i,t in enumerate(tag):
            if not t:
                point_set.append(nearest[i])
        if len(point_set)<len(Point):
            Point = point_set

    return Point
def tsp(point_set):
    distmat = Move[point_set,:][:,point_set]# 节点的距离矩阵
    
    numant = 100  # 蚂蚁个数
    numcity = len(distmat)
    alpha = 1  # 信息素重要程度因子
    beta = 5   # 启发函数重要程度因子
    rho = 0.1  # 信息素的挥发速度
    Q = 1      # 完成率

    iter = 0       #迭代初始
    itermax = 150  #迭代总数
    
    
    etatable = 1.0 / (distmat + np.diag([1e10] * numcity))
    pheromonetable = np.ones((numcity, numcity))# 信息素矩阵 16*16
    pathtable = np.zeros((numant, numcity)).astype(int)# 路径记录表，转化成整型 100*16


    degree = [sum(Adj[p]!=np.inf) for p in point_set]#节点的度
    MAX_degree = max(degree)
    lengthaver = np.zeros(itermax)  # 迭代，存放每次迭代后，路径的平均长度  
    lengthbest = np.zeros(itermax)  # 迭代，存放每次迭代后，最佳路径长度  
    pathbest = np.zeros((itermax, numcity))  # 迭代，存放每次迭代后，最佳路径城市的坐标



    ###########################################################
    while iter < itermax:#迭代总数

        #60个蚂蚁随机放置于16个城市中
        if numant <= numcity:  # 城市数比蚂蚁数多，不用管
            pathtable[:, 0] = np.random.permutation(range(numcity))[:numant]
        else:  # 蚂蚁数比城市数多，需要有城市放多个蚂蚁
            pathtable[:numcity, 0] = np.random.permutation(range(numcity))[:]
            pathtable[numcity:, 0] = np.random.randint(numcity,size= numant - numcity)
        length = np.zeros(numant)  

        #本段程序算出每只/第i只蚂蚁转移到下一个城市的概率
        for i in range(numant):

            visiting = pathtable[i, 0]  # 当前所在的城市
            unvisited = set(range(numcity))#未访问的城市集合
            unvisited.remove(visiting)  # 删除已经访问过的城市元素

            for j in range(1, numcity):  # 循环numcity-1次，访问剩余的所有numcity-1个城市
                # j=1
                # 每次用轮盘法选择下一个要访问的城市
                listunvisited = list(unvisited)
                #未访问城市数
                probtrans = np.zeros(len(listunvisited))
                #每次循环都初始化转移概率矩阵


                #以下是计算转移概率
                for k in range(len(listunvisited)):
                    probtrans[k] = np.power(pheromonetable[visiting][listunvisited[k]], alpha) \
                                   * np.power(etatable[visiting][listunvisited[k]], alpha)
                #eta-从城市i到城市j的启发因子 这是概率公式的分母   其中[visiting][listunvis[k]]是从本城市到k城市的信息素
                cumsumprobtrans = (probtrans / sum(probtrans)).cumsum()
                #求出本只蚂蚁的转移到各个城市的概率斐波衲挈数列

                cumsumprobtrans -= np.random.rand()# 随机生成下个城市的转移概率，再用区间比较
                k = listunvisited[list(cumsumprobtrans > 0).index(True)]# 函数选出符合cumsumprobtans>0的数

                pathtable[i, j] = k# 下一个要访问的城市
                unvisited.remove(k)#将未访问城市列表中的K城市删去，增加到已访问城市列表中

                length[i] += distmat[visiting][k]#计算本城市到K城市的距离
                visiting = k

            length[i] += distmat[visiting][pathtable[i, 0]]# 计算本只蚂蚁的总的路径距离，包括最后一个城市和第一个城市的距离

        # 包含所有蚂蚁的一个迭代结束后，统计本次迭代的若干统计参数

        #本轮的平均路径
        lengthaver[iter] = length.mean()


        #####本部分是为了求出最佳路径

        if iter == 0:
            lengthbest[iter] = length.min()
            pathbest[iter] = pathtable[length.argmin()].copy()#如果是第一轮路径，则选择本轮最短的路径,并返回索引值下标，并将其记录
        else:
        #后面几轮的情况，更新最佳路径
            if length.min() > lengthbest[iter - 1]:
                lengthbest[iter] = lengthbest[iter - 1]
                pathbest[iter] = pathbest[iter - 1].copy()
            else:
                lengthbest[iter] = length.min()
                pathbest[iter] = pathtable[length.argmin()].copy()


        #########此部分是为了更新信息素
        changepheromonetable = np.zeros((numcity, numcity))
        for i in range(numant):#更新所有的蚂蚁
            for j in range(numcity - 1):#城市
    #             changepheromonetable[pathtable[i, j]][pathtable[i, j + 1]] += Q / distmat[pathtable[i, j]][pathtable[i, j + 1]]
                #根据公式更新本只蚂蚁改变的城市间的信息素   Q/d   其中d是从第j个城市到第j+1个城市的距离
                changepheromonetable[pathtable[i, j]][pathtable[i, j + 1]] +=(Q / distmat[pathtable[i, j]][pathtable[i, j + 1]])*(MAX_degree/(degree[j]+degree[j+1]))
            changepheromonetable[pathtable[i, j + 1]][pathtable[i, 0]] += (Q / distmat[pathtable[i, j + 1]][pathtable[i, 0]])*(MAX_degree/(degree[j]+degree[j+1]))
            #首城市到最后一个城市 所有蚂蚁改变的信息素总和

        #信息素更新公式p=(1-挥发速率)*现有信息素+改变的信息素
        pheromonetable = (1 - rho) * pheromonetable + changepheromonetable

        iter += 1  # 迭代次数指示器+1

    #迭代完成


    # 作出找到的最优路径图
#     print(lengthbest[-1],pathbest[-1])
    return lengthbest[-1],np.array(point_set)[pathbest[-1].astype(int)]
def broad_index(Adj,route_all):
    visited=np.full(len(Adj),False)
    for p in route_all:
        visited|=(Adj[p]!=np.inf)
    return sum(visited)/len(visited)
def freq_index(Adj,route_all):
    route_all_len=[]
    route_all_para=[]
    for ind,_ in enumerate(route_all):
        prior_ind = route_all[ind]
        next_ind = route_all[(ind+1)%len(route_all)]
        route_all_len.append(Adj[prior_ind,next_ind])
        vec = Adj[next_ind].copy()
        vec.sort()
        route_all_para.append(sum(vec[[0,1]])/2)#路的一半的和即为均值

    return sum(route_all_para)/sum(route_all_len)
# 两点之间的通路        
def getPath(i, j,path,p_):
    if i != j:
        if path[i][j] == -1:
            p_.append(j)
        else:
            getPath(i, path[i][j],P,p_)
            getPath(path[i][j], j,P,p_)
    return p_
#寻找经过关键节点的回路
def route(path,P):
    path_all=[]
    for ind,_ in enumerate(path):
        next_p=path[(ind+1)%len(path)]
        prior_p = path[ind]
        p_=[]
        path_all.extend(getPath(prior_p,next_p,P,p_))
    return path_all
import matplotlib.pyplot as plt
def print_map(P_add,path,point_set,l=True,p=True):
    data_path = r'../input'
    E = pd.read_excel(os.path.join(data_path,'task-opt-done/map.xls'),sheet_name=1,header=1)
    for e in E.itertuples():
        plt.plot( [P_add[e[1]-1][0],P_add[e[2]-1][0]],[P_add[e[1]-1][1],P_add[e[2]-1][1]],color='k',)
    if l == True:
        for ind,_ in enumerate(path):
            plt.plot( [P_add[path[ind]-1][0],P_add[path[(ind+1)%len(path)]-1][0]],[P_add[path[ind]-1][1],P_add[path[(ind+1)%len(path)]-1][1]],color='b')
    if p==True:
        for p in point_set:
            plt.scatter(P_add[p-1][0],P_add[p-1][1],linewidths=100,marker='x',color = 'r')
    pass
def print_table(path,vital_point,Adj,P_add):
    T=0
    change_car = False
    vital_point=[v-1 for v in  vital_point]
    v_car = 40#车速 km/h
    N_car = len(vital_point)+1#车的数量
    n_car = 0#正在运行车的编号
    ind = 0#道路节点标号
    gone_len = 0#某条路已经行走的路程
    recoding = np.zeros((4*60+1,N_car,2))
    way_len = Adj[path[(ind)%len(path)],path[(ind+1)%len(path)]]#路长
    recoding[0]=np.vstack((P_add[vital_point[0]],P_add[vital_point]))
    while T < 4*60:
        T += 1#时间流逝
        recoding[T]=recoding[T-1]#大多都是不变的

        gone_len += v_car*1000/60#每分钟都走一段距离

        while way_len-gone_len<=0:#判断是否已走路径超过路的长度路的个数大于等于1
            gone_len=gone_len-way_len
            ind += 1
            if path[(ind)%len(path)]-1 in vital_point:
                change_car = True
                n_car = (n_car+1)%N_car
            way_len = Adj[path[(ind)%len(path)],path[(ind+1)%len(path)]]
        #给出了路的起点和终点
        begin_p = P_add[path[(ind)%len(path)]-1]
        end_p = P_add[path[(ind+1)%len(path)]-1]
        #该分钟结束后车的位置
        point = begin_p+(begin_p-end_p)/way_len*gone_len
        if change_car:
            recoding[T,(n_car-1)%N_car]=recoding[T,n_car%N_car]
            change_car = False
        recoding[T,n_car] = point
    return recoding
            
# np.save('Move.npy',Move)
# np.save('P.npy',P)
Move=np.load('../input/tasktemp/Move.npy')
P = np.load('../input/tasktemp/P.npy')
import random
random.seed(1)
np.random.seed(1)
import copy
point_set = Greedy(Move,nearest,100)
show = copy.deepcopy(point_set)

print('重要节点：')
print(point_set)

#展示
for addp in addpoint:
    if addp[0] in show:
        
        show.remove(addp[0])
        show.append(addp[3:])
print('重要节点所在位置：')
print(show)
print('重要节点个数：')
print(len(show))
print('蚁群算法的tsp近似解：')
tsp_res = tsp(point_set)
print(tsp_res)
route_all = route(tsp_res[1],P)
print('所有路径:')
print(route_all)

print_map(P_add,route_all,point_set,l=True,p=True)
print('广度\频度指标:')
print(broad_index(Adj,route_all),freq_index(Adj,route_all))


#生成记录
recodings = print_table(route_all,point_set,Adj,P_add)

with open('2017357770121-Result3.csv','w') as f:
    f.write(str(recodings.shape[1]))
    f.write(',')
    f.write(str(broad_index(Adj,route_all)))
    f.write(',')
    f.write(str(freq_index(Adj,route_all)))
    for ind,Item in enumerate(recodings):
        f.write('\n')
        f.write(str(ind))
        for Cell in Item:
            f.write(',')
            f.write(str(tuple(Cell)))
print_map(P_add,route_all,point_set,l=0,p=True)

random.seed(6)
np.random.seed(6)
def Greedy_10(Adj,Move,P,nearest,n,v_car=40):
    visit_rate=0
    k=0
    while k < n:
        point_set=[]
        #记录是否可以两分钟到达
        tag = np.full(3,False)
        visited = []
        visiting = list(range(len(Move)))
        visiting.pop(0)

        while len(point_set)<9:#len(visiting):
            p = random.choice(visiting)
            #访问节点p
            visiting.remove(p)
            visited.append(p)
            point_set.append(p)

            adj_p = [visiting[ind] for ind,Len in enumerate(Move[p,visiting]) if Len<v_car*3*1000/60]
            visited.extend(adj_p)
            for p in adj_p:
                visiting.remove(p)

            tag |= Move[p,nearest]<v_car*2*1000/60
            if (len(visited)/len(Move)>=0.9) & all(tag):
                break

        if not all(tag):
            continue
        k+=1

        if 1-1.*(len(visiting))/len(Move)>visit_rate:
            visit_rate= 1-len(visiting)/len(Move)
            Point = point_set
            adj_=np.full(len(Adj),False)
            for p in point_set:
                adj_[p]=True
                adj_ |= Adj[p]!=np.inf
            broad = sum(adj_)/len(adj_)
    Point,visit_rate,broad
    return Point,visit_rate,broad

point_set,visit_rate,broad_ind = Greedy_10(Adj,Move,P,nearest,100,v_car=40)

tsp_res = tsp(point_set)
route_all = route(tsp_res[1],P)
freq_ind=freq_index(Adj,route_all)
print_map(P_add,path=route_all,point_set=point_set,l=1,p=1)
print(point_set,visit_rate,broad_ind,freq_ind)
#记录
recodings = print_table(route_all,point_set,Adj,P_add)

with open('2017357770121-Result5.csv','w') as f:
    f.write(str(recodings.shape[1]))
    f.write(',')
    f.write(str(broad_index(Adj,route_all)))
    f.write(',')
    f.write(str(freq_index(Adj,route_all)))
    for ind,Item in enumerate(recodings):
        f.write('\n')
        f.write(str(ind))
        for Cell in Item:
            f.write(',')
            f.write(str(tuple(Cell)))
random.seed(1)
np.random.seed(1)

point_set = Greedy(Move,nearest,100,v_car=50)
show = copy.deepcopy(point_set)

print('重要节点：')
print(point_set)

#展示
for addp in addpoint:
    if addp[0] in show:
        
        show.remove(addp[0])
        show.append(addp[3:])
print('重要节点所在位置：')
print(show)
print('重要节点个数：')
print(len(show))
print('蚁群算法的tsp近似解：')
tsp_res = tsp(point_set)
print(tsp_res)
route_all = route(tsp_res[1],P)
print('所有路径:')
print(route_all)

print_map(P_add,route_all,point_set,l=True,p=True)

print('广度\频度指标:')
print(broad_index(Adj,route_all),freq_index(Adj,route_all))

#生成记录
recodings = print_table(route_all,point_set,Adj,P_add)

with open('2017357770121-Result6.csv','w') as f:
    f.write(str(recodings.shape[1]))
    f.write(',')
    f.write(str(broad_index(Adj,route_all)))
    f.write(',')
    f.write(str(freq_index(Adj,route_all)))
    for ind,Item in enumerate(recodings):
        f.write('\n')
        f.write(str(ind))
        for Cell in Item:
            f.write(',')
            f.write(str(tuple(Cell)))