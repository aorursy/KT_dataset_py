# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
from numpy import mat

# 由于kaggle无预装gexf，因此涉及gexf操作的都进行了注释

# from gexf import Gexf

import networkx as nx
# 读取钓鱼节点的一阶和二阶交易数据，构成gexf二阶网络演示



# 获取path目录下的所有文件带后缀的文件名，保存到list_name中

def listDir(path, list_name):

    # os.listdir能获取路径下的所有文件名字以及所有文件夹名字

    for oneFile in os.listdir(path):

        list_name.append(oneFile)





# 获取path目录下的指定文件的不带后缀的文件名，保存到list_name中

def listDoc(path, list_name):

    # os.listdir能获取路径下的所有文件名字以及所有文件夹名字

    for file in os.listdir(path):

        # 这里筛选出所有.csv格式的文件

        # 例如os.path.splitext("E:\\lena.jpg")将得到"E:\lena"+".jpg"。

        if os.path.splitext(file)[1] == '.csv':

            # 只要.csv前面的部分

            list_name.append(os.path.splitext(file)[0])





def read():

    listAddress = []

    listDir('../input/secondorder-transaction-network-of-phishing-nodes/Phishing second-order nodes/', listAddress)



    # 获取初始钓鱼节点总数量

    number = len(listAddress)

    print(number)

    nowNumber = 0



    # 取出所有已知的初始钓鱼节点

    for address in listAddress:

        # address = '0xfdA941855Fb2E89b1b6d806d8a105C8a17947281'

        print('当前处理的原始节点为: ' + address)



        # 获取所有一阶节点，打开文件夹即可

        listFirstOrderAddress = []

        listDoc('../input/secondorder-transaction-network-of-phishing-nodes/Phishing second-order nodes/' + address, listFirstOrderAddress)

        # print('listFirstOrderAddress: ', listFirstOrderAddress)



        # 声明gexf，命名为当前的address，因为gexf存储的是当前address的交易网络

        gexf = Gexf(address, "A transaction network")

        # 为gexf添加图

        graph = gexf.addGraph("directed", "static", "A transaction network")

        # 为图添加节点属性

        atr1 = graph.addNodeAttribute('address', type='string', defaultValue="")



        # 该变量记录当前索引

        nowIndex = 0



        oneList = [address.upper()]

        # 该变量保存网格中所有的节点，用于查重，所以全部节点大写表示

        allNodes = set(oneList)

        # 该变量保存节点及在二维数组中对应的序号，序号也是gexf节点的序号

        oneDict = {address.upper(): nowIndex}

        # 添加gexf的节点

        tmp = graph.addNode(str(nowIndex), address)

        tmp.addAttribute(atr1, address)

        # 处理完索引加一

        nowIndex += 1



        # 把一阶节点添加进来

        for firstOrderAddress in listFirstOrderAddress:

            # 保存在文件夹里的肯定是去重过的，直接添加到set中

            allNodes.add(firstOrderAddress.upper())

            # 更新节点序号

            oneDict.update({firstOrderAddress.upper(): nowIndex})

            # 添加gexf的节点

            tmp = graph.addNode(str(nowIndex), firstOrderAddress)

            tmp.addAttribute(atr1, firstOrderAddress)

            # 处理完索引加一

            nowIndex += 1



        print('一阶节点添加完毕！')



        # 添加节点，节点即为当前address和其交易节点

        # 把一阶节点的一阶节点也就是二阶节点添加进来

        for firstOrderAddress in listFirstOrderAddress:

            # print('添加' + firstOrderAddress + '的二阶节点')

            # 获得该一阶节点的csv文件目录

            tmpFile = '../input/secondorder-transaction-network-of-phishing-nodes/Phishing second-order nodes/' + address + '/' + firstOrderAddress + '.csv'

            tmpF = open(tmpFile)



            # 将该csv文件打开，保存为dataframe格式

            df = pd.read_csv(tmpF, index_col=0)

            tmpF.close()



            # 获得其所有交易中的节点集合

            # 注意这里不能用加号合并，因为会把a和b合并成ab，而我要的是a,b……

            secondOrderNodes = df['From'].append(df['To'])



            # 筛选掉重复项，因为可能跟同一个一阶节点有多次交易,first表示保留第一次出现的重复项

            secondOrderNodes = secondOrderNodes.drop_duplicates(keep='first')



            # 把所有NAN去掉，注意节点中还有自身的节点没去掉，等下会进行一次最终过滤

            secondOrderNodes = secondOrderNodes.dropna(axis=0, how='any')



            # 转化为list形式，获得二阶节点的list

            listSecondOrderAddress = secondOrderNodes.tolist()



            # 对每个二阶节点进行处理

            for secondOrderAddress in listSecondOrderAddress:

                # 判断添加后的长度，防止节点重复

                prevLen = len(allNodes)

                allNodes.add(secondOrderAddress.upper())

                nowLen = len(allNodes)



                # allNodes为set，元素不能重复，如果插入后长度变长，表示成功插入，不与已有节点重复

                if nowLen > prevLen:

                    # 更新节点序号

                    oneDict.update({secondOrderAddress.upper(): nowIndex})

                    # 添加gexf的节点

                    tmp = graph.addNode(str(nowIndex), secondOrderAddress)

                    tmp.addAttribute(atr1, secondOrderAddress)

                    # 处理完索引加一

                    nowIndex += 1



        print('二阶节点添加完毕！')

        # print('oneDict:', oneDict)

        # print('allNodes:', allNodes)

        # print('nowIndex:', nowIndex)

        # print('listFirstOrderAddress:', listFirstOrderAddress)



        # 至此，gexf文件里总共要添加多少节点已经算明白了，就是nowIndex个，建一个nowIndex×nowIndex的矩阵，用于保存边的权值信息

        oneMatrix = mat(np.zeros((nowIndex, nowIndex)))



        # 着手收集边的权值信息

        for firstOrderAddress in listFirstOrderAddress:

            # print('收集' + firstOrderAddress + '的二阶节点交易数据')

            tmpFile = '../input/secondorder-transaction-network-of-phishing-nodes/Phishing second-order nodes/' + address + '/' + firstOrderAddress + '.csv'

            tmpF = open(tmpFile)

            # 将该csv文件打开，保存为dataframe格式

            df = pd.read_csv(tmpF, index_col=0)

            tmpF.close()



            # 逐行读取记录保存到矩阵中

            for oneIndex in df.index:

                # 只需要记录三列数据，从XX到XX交易了XX

                oneFrom = str(df.loc[oneIndex]['From']).upper()

                oneTo = str(df.loc[oneIndex]['To']).upper()

                oneValue = float(df.loc[oneIndex]['Value'])



                # 需要判断是否存在，因为有可能这个节点之前在爬虫的时候被过滤了，所以压根没对应的csv文件

                if oneFrom in oneDict and oneTo in oneDict:

                    # 找到节点序号

                    oneFromIndex = oneDict[oneFrom]

                    oneToIndex = oneDict[oneTo]

                    oneMatrix[oneFromIndex, oneToIndex] += oneValue



        print('二阶节点边权值收集完毕')



        # 终于可以将矩阵里的数据转化成gexf的边来添加进网络了

        edgeIndex = 0

        for i in range(nowIndex):

            for j in range(nowIndex):

                if oneMatrix[i, j] != 0.0:

                    graph.addEdge(str(edgeIndex), str(i), str(j), weight=str(oneMatrix[i, j]))

                    edgeIndex += 1



        print('边转换完毕，构建节点网络完毕！')



        # 写为gexf文件

        output_file = open("../input/secondorder-transaction-network-of-phishing-nodes/phiGexfData/" + address + ".gexf", "wb")

        gexf.write(output_file)



        # 计算还剩余多少节点网络未生成

        nowNumber += 1

        rest = number - nowNumber

        print('保存为gexf文件完毕，已完成: ' + str(nowNumber) + '个，还剩' + str(rest) + '个')

        print('已完成节点: ' + address)



    # 每一个节点所对应的网络将以gexf的形式存放在对应文件夹中

    # 非钓鱼节点部分的读取过程一致，只需把对应路径修改为非钓鱼前缀文件夹即可

    # 如果生成的gexf文件无法正常读取，可能是gexf版本所造成的gexf标签内容不一致问题
# read()

address = '0xfdA941855Fb2E89b1b6d806d8a105C8a17947281'

# graph = nx.read_gexf(filrdir + address + '.gexf')