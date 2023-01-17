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
# 读取数据，共有76条数据，34个属性列



fl = pd.read_csv('/kaggle/input/information-security-studentsgrades-at-university/Information security studentsgrades from freshman to junior.csv')



# 设置排名为索引



fl.set_index('排名',inplace=True)



fl.head()
# 检查缺失值



check_null = fl.isnull().sum()



check_null
# 删除有缺失值的属性列



drop_arr = ['大学英语二', '大学英语三', '大学英语四', '大学英语二.1', '大学英语三.1', '大学英语一']



fl.drop(drop_arr, axis=1, inplace=True)



fl.head()
# 删除不研究属性列



drop_arr = ['学号','大学语文','中国近现代史纲要','思想道德修养与法律基础','马原','毛泽东思想与中国特色社会主义理论体系概论','总分']



t = fl.drop(drop_arr, axis=1, inplace=False)



t.head()
# 分类统计数据类型



t.dtypes.value_counts() 
# 查看数据统计信息



t.describe()
# 按成绩高于75的成绩算作成绩有效



seventy_5 = [75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75]



ax = np.full_like(t,seventy_5)



ax[:5]
t = t > ax



t.head()
# 将初步预处理后的数据转化为csv



t.to_csv('grades_processed.csv',encoding='utf-8')
data = pd.read_csv('grades_processed.csv',encoding='utf-8')



data.set_index('排名',inplace=True)



data.head()
# 利用Apriori算法进行关联分析



def createC1(dataSet):

    C1=[]

    for transaction in dataSet:

        for item in transaction:

            if not [item] in C1:

                C1.append([item])

    C1.sort()

    return list(map(frozenset,C1))



def scanD(D,CK,minSupport):

    ssCnt = {}

    for tid in D:

        for can in CK:

            if can.issubset(tid):

                if not can in ssCnt:ssCnt[can]=1

                else:ssCnt[can]+=1

    numItems = float(len(D))

    retList = []

    supportData={}

    for key in ssCnt:

        support = ssCnt[key]/numItems

        if support>=minSupport:

            retList.insert(0,key)

        supportData[key]=support

    return retList,supportData



#频繁项集两两组合



def aprioriGen(Lk,k):

    retList=[]

    lenLk = len(Lk)

    for i in range(lenLk):

        for j in range(i+1,lenLk):

            L1=list(Lk[i])[:k-2];L2=list(Lk[j])[:k-2]

            L1.sort();L2.sort()

            if L1==L2:

                retList.append(Lk[i]|Lk[j])

    return retList





def apriori(dataSet,minSupport=0.5):

    C1=createC1(dataSet)

    D=list(map(set,dataSet))

    L1,supportData =scanD(D,C1,minSupport)

    L=[L1]

    k=2

    while(len(L[k-2])>0):

        CK = aprioriGen(L[k-2],k)

        Lk,supK = scanD(D,CK,minSupport)

        supportData.update(supK)

        L.append(Lk)

        k+=1

    return L,supportData



#规则计算的主函数

def generateRules(L,supportData,minConf):

    bigRuleList = []

    for i in range(1,len(L)):

        for freqSet in L[i]:

            H1 = [frozenset([item]) for item in freqSet]

            if(i>1):

                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)

            else:

                calcConf(freqSet,H1,supportData,bigRuleList,minConf)

    return bigRuleList





def calcConf(freqSet,H,supportData,brl,minConf):

    prunedH=[]

    for conseq in H:

        conf = supportData[freqSet]/supportData[freqSet-conseq]

        if conf>=minConf:

            print (freqSet-conseq,'--->',conseq,'conf:',conf)

            brl.append((freqSet-conseq,conseq,conf))

            prunedH.append(conseq)

    return prunedH

def rulesFromConseq(freqSet,H,supportData,brl,minConf):

    m = len(H[0])

    if (len(freqSet)>(m+1)):

        Hmp1 = aprioriGen(H,m+1)

        Hmp1 = calcConf(freqSet,Hmp1,supportData,brl,minConf)

        if(len(Hmp1)>1):

            rulesFromConseq(freqSet,Hmp1,supportData,brl,minConf)



            

# 构建数据集

def creat(da):

    daset=[]

    

    for i in range(0, len(da)):

        perdaset=[]

        t = 0

        for k in da.iloc[i]:

            if(k):

                perdaset.append(da.iloc[i].index[t])

            t = t+1

        daset.append(perdaset)

    return daset
one = pd.concat([data['面向对象程序设计'],data['数据结构'],data['数据库原理'],data['信息论与编码技术'],data['计算机网络/专业核心课程/3'],data['计算机组成原理/专业核心课程/3']],axis=1)



one.head()
# 进行分析数据



dataSet=creat(one)

L,supportData=apriori(dataSet)

rules = generateRules(L,supportData,minConf=0.7)
two = pd.concat([data['数据库原理'],data['计算机组成原理/专业核心课程/3'],data['密码学基础/专业核心课程/4'],data['信息安全概论/专业核心课程/2.5'],data['操作系统/专业核心课程/3']],axis=1)



two.head()
# 进行分析数据



dataSet=creat(two)

L,supportData=apriori(dataSet)

rules = generateRules(L,supportData,minConf=0.6)
three = pd.concat([data['数据结构'],data['数据库原理'],data['操作系统/专业核心课程/3'],data['计算机网络/专业核心课程/3'],data['密码学基础/专业核心课程/4'],data['计算机组成原理/专业核心课程/3']],axis=1)



three.head()
# 进行分析数据



dataSet=creat(three)

L,supportData=apriori(dataSet)

rules = generateRules(L,supportData,minConf=0.7)
one = pd.concat([data['面向对象程序设计'],data['数据结构'],data['数据库原理'],data['高等数学一'],data['高等数学（二）'],data['概率论与数理统计']],axis=1)



one.head()
# 进行分析数据



dataSet=creat(one)

L,supportData=apriori(dataSet)

rules = generateRules(L,supportData,minConf=0.7)
two = pd.concat([data['面向对象程序设计'],data['数据结构'],data['数据库原理'],data['线性代数'],data['信息安全数学基础'],data['概率论与数理统计']],axis=1)



two.head()
# 进行分析数据



dataSet=creat(two)

L,supportData=apriori(dataSet)

rules = generateRules(L,supportData,minConf=0.7)
three = pd.concat([data['信息论与编码技术'],data['密码学基础/专业核心课程/4'],data['高等数学一'],data['高等数学（二）'],data['线性代数'],data['概率论与数理统计']],axis=1)



three.head()
# 进行分析数据



dataSet=creat(three)

L,supportData=apriori(dataSet)

rules = generateRules(L,supportData,minConf=0.7)
one = pd.concat([data['面向对象程序设计'],data['数据结构'],data['大学物理'],data['大学物理（二）'],data['数字逻辑'],data['数据通信原理']],axis=1)



one.head()
# 进行分析数据



dataSet=creat(one)

L,supportData=apriori(dataSet)

rules = generateRules(L,supportData,minConf=0.7)
two = pd.concat([data['信息论与编码技术'],data['操作系统/专业核心课程/3'],data['密码学基础/专业核心课程/4'],data['计算机网络/专业核心课程/3'],data['数字逻辑'],data['数据通信原理']],axis=1)



two.head()
# 进行分析数据



dataSet=creat(two)

L,supportData=apriori(dataSet)

rules = generateRules(L,supportData,minConf=0.7)
three = pd.concat([data['数据库原理'],data['计算机组成原理/专业核心课程/3'],data['大学物理'],data['大学物理（二）'],data['数字逻辑'],data['数据通信原理']],axis=1)



three.head()
# 进行分析数据



dataSet=creat(three)

L,supportData=apriori(dataSet)

rules = generateRules(L,supportData,minConf=0.7)