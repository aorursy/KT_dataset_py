# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from statsmodels.tools.sm_exceptions import HessianInversionWarning,warnings

warnings.filterwarnings(action='ignore', category=HessianInversionWarning)
import numpy as np

import pandas as pd

import re

import time

import datetime

import pickle #把结构化数据存起来，读的时候也是结构化的读，而不是字符串（如存的字典，读出来也是字典而不是字符串）

from dateutil.relativedelta import relativedelta #求两个时间差，给一个时间和时间差求另一个时间的模块

from sklearn.model_selection import train_test_split #Split arrays or matrices into random train and test subsets

import matplotlib.pyplot as plt

import seaborn as sns #可视化

from statsmodels.stats.outliers_influence import variance_inflation_factor #方差膨胀因子，如果大于5则说明解释变量与其他解释变量高度共线性，因此参数估计将具有很大的标准误差

from sklearn.linear_model import LogisticRegressionCV,LogisticRegression

import statsmodels.api as sm #统计模型

from sklearn.ensemble import RandomForestClassifier

from numpy import log

from sklearn.metrics import roc_auc_score #接受者操作特征曲线是一种比较两个分类模型有用的可视化工具，ROC曲线以下的面积就是模型准确率的度量（AUC）



#find函数是检索输入字符串，如果有就返回该字符串，如果没有就返回-1

def CareerYear(x):

    #对工作年限进行转换

    if str(x).find('n/a') > -1:

        return -1

    elif str(x).find("10+")>-1:   #将"10＋years"转换成 11

        return 11

    elif str(x).find('< 1') > -1:  #将"< 1 year"转换成 0

        return 0

    else:

        

        if re.sub(r"\D",'', str(x)) == '':

            return -1

        else:

            return int(re.sub(r"\D",'', str(x)))

            

    

def DescExisting(x):

    #将desc变量转换成有记录和无记录两种

    if type(x).__name__ == 'float':

        return 'no desc'

    else:

        return 'desc'





def ConvertDateStr(x):#??????????为什么能解决issue_day这一列的时间问题

    mth_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,

                'Nov': 11, 'Dec': 12}

    if str(x) == 'nan':

        return datetime.datetime.fromtimestamp(time.mktime(time.strptime('9900-1','%Y-%m')))

        #time.mktime 不能读取1970年之前的日期

    else:

        yr = int(x[4:6])

        if yr <=17:

            yr = 2000+yr

        else:

            yr = 1900 + yr

        mth = mth_dict[x[:3]]

        return datetime.datetime(yr,mth,1)





def MonthGap(earlyDate, lateDate):

    if lateDate > earlyDate:

        gap = relativedelta(lateDate,earlyDate)

        yr = gap.years

        mth = gap.months

        return yr*12+mth

    else:

        return 0





def MakeupMissing(x):

    if np.isnan(x):

        return -1

    else:

        return x

    

file_add = '../input/'

allData = pd.read_csv(file_add + 'application.csv',header = 0,encoding = 'latin1')



allData['term'] = allData['term'].apply(lambda x: int(x.replace(' months',''))) 

#把借贷期限这一列的后缀都去掉只留下数字



#处理标签：Fully paid 是正常用户，charger off是违约用户

allData['y'] = allData['loan_status'].map(lambda x: int(x =='Charged Off'))

#对用户的还款状态进行编码，返回一列0,1   1是违约用户



#由于存在不同的贷款期限（term），申请评分卡模型评估的违约概率必须要在统一的期限中，

#且不宜太长，所以选取term＝36months的样本

allData1 = allData.loc[allData.term ==36]

trainData, testData = train_test_split(allData1,test_size=0.4)

#随机分割出0.6的训练集，0.4的测试集



#固化变量

#本地跑代码可用

# trainDataFile = open(file_add+'trainData.pkl','wb') #py3中要写wb

# pickle.dump(trainData, trainDataFile) #将对象以文件的形式存放在磁盘上

# trainDataFile.close()



# testDataFile = open(file_add+'testData.pkl','wb')

# pickle.dump(testData, testDataFile)

# testDataFile.close()



'''

第一步：数据预处理，包括

（1）数据清洗

（2）格式转换

（3）缺失值填补

'''

#将带%的字符串变成浮点数

trainData['int_rate_clean'] = trainData['int_rate'].map(lambda x: float(x.replace('%',''))/100)



# 将工作年限进行转化，否则影响排序

trainData['emp_length_clean'] = trainData['emp_length'].map(CareerYear)



# from collections import Counter

# Counter(trainData['emp_length_clean'])



# 将desc借款描述的缺失作为一种状态，非缺失作为另一种状态

trainData['desc_clean'] = trainData['desc'].map(DescExisting)



# 处理日期。earliest_cr_line的格式不统一，需要统一格式且转换成python的日期

trainData['app_date_clean'] = trainData['issue_d'].map(lambda x: ConvertDateStr(x))

trainData['earliest_cr_line_clean'] = trainData['earliest_cr_line'].map(lambda x: ConvertDateStr(x))



# 处理日期，需要先统一earliest_cr_line格式，再转换成python日期

# trainData['app_date_clean'] = trainData['issue_d'].map(lambda x: ConvertDateStr(x,'%y-%b'))

# trainData['earliest_cr_line_clean'] = trainData['earliest_cr_line'].map(lambda x: ConvertDateStr(x,'%b-%y'))



# 处理mths_since_last_delinq。注意原始值中有0，所以用－1代替缺失

trainData['mths_since_last_delinq_clean'] = trainData['mths_since_last_delinq'].map(lambda x:MakeupMissing(x))



trainData['mths_since_last_record_clean'] = trainData['mths_since_last_record'].map(lambda x:MakeupMissing(x))



trainData['pub_rec_bankruptcies_clean'] = trainData['pub_rec_bankruptcies'].map(lambda x:MakeupMissing(x))



#第二步：变量衍生

# 考虑申请额度与收入的占比

trainData['limit_income'] = trainData.apply(lambda x: x.loan_amnt / x.annual_inc, axis = 1)



# 考虑earliest_cr_line到申请日期的跨度，以月份记

trainData['earliest_cr_to_app'] = trainData.apply(lambda x: MonthGap(x.earliest_cr_line_clean,x.app_date_clean), axis = 1)



# 三、分箱：卡方分箱

'''

第三步：分箱，采用ChiMerge,要求分箱完之后：

（1）不超过5箱

（2）Bad Rate单调

（3）每箱同时包含好坏样本

（4）特殊值如－1，单独成一箱



连续型变量可直接分箱

类别型变量：

（a）当取值较多时，先用bad rate编码，再用连续型分箱的方式进行分箱

（b）当取值较少时：

    （b1）如果每种类别同时包含好坏样本，无需分箱

    （b2）如果有类别只包含好坏样本的一种，需要合并

'''

def BinBadRate(df, col, target, grantRateIndicator=0):#默认不返回总体坏样本率，也可以传1

    '''

    :param df: 需要计算好坏比率的数据集

    :param col: 需要计算好坏比率的特征

    :param target: 好坏标签

    :param grantRateIndicator: 1返回总体的坏样本率，0不返回

    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）

    '''

    total = df.groupby([col])[target].count() #数一下该类别型变量各种类里的样本数

    total = pd.DataFrame({'total': total})

    bad = df.groupby([col])[target].sum()  #将各种类里的样本的是否违约情况相加（因为之前将是否违约用0和1编码到'y'这一列，因此sum出的结果实际就是违约样本的个数）

    bad = pd.DataFrame({'bad': bad})

    regroup = total.merge(bad, left_index=True, right_index=True, how='left')#把坏样本数左连接到总样本数上

    regroup.reset_index(level=0, inplace=True) #新建从0，1,2开始的索引

    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1) #用bad样本数除以总样本数得到各类的bad rate

    dicts = dict(zip(regroup[col],regroup['bad_rate']))#这一步是把该类别型变量的各类的序列和对应的bad rate序列通过zip函数转化成（类别，bad rate）这样两两对应的元组组成的list，再转化成字典

    if grantRateIndicator==0:

        return (dicts, regroup)

    N = sum(regroup['total'])

    B = sum(regroup['bad'])

    overallRate = B * 1.0 / N

    return (dicts, regroup, overallRate)



def MergeBad0(df,col,target, direction='bad'):#默认值direction是bad，也可以传good

    '''

     :param df: 包含检验0％或者100%坏样本率

     :param col: 分箱后的变量或者类别型变量。检验其中是否有一组或者多组没有坏样本或者没有好样本。如果是，则需要进行合并

     :param target: 目标变量，0、1表示好、坏

     :return: 合并方案，使得每个组里同时包含好坏样本

     '''

    regroup = BinBadRate(df, col, target)[1]#return (dicts, regroup)取regroup这个dataframe

    if direction == 'bad':

        # 如果是合并0坏样本率的组，则跟最小的非0坏样本率的组进行合并

        regroup = regroup.sort_values(by  = 'bad_rate')#按照bad rate排序（默认升序）后bad rate为0和非0的最小bad rate组就挨着，这两个组就可以合并

    else:

        # 如果是合并0好样本样本率的组，则跟最小的非0好样本率的组进行合并

        regroup = regroup.sort_values(by='bad_rate',ascending=False)#降序排列

    regroup.index = range(regroup.shape[0])#按regroup的行数来建立从0开始的索引

    col_regroup = [[i] for i in regroup[col]] #col这个特征的各类组成的一个list，各类就相当于箱子（bin）

    del_index = []

    for i in range(regroup.shape[0]-1):#行数减1

        col_regroup[i+1] = col_regroup[i] + col_regroup[i+1] #把前两个箱子合并后赋值给第二个箱子，但是第一个箱子还在，下面会从新遍历

        del_index.append(i)

        if direction == 'bad': 

            if regroup['bad_rate'][i+1] > 0: #当每箱同时包含好坏样本的时候停止合并

                break

        else: #相当于direction==‘good’要合并good rate=0就相当于bad_rate==1

            if regroup['bad_rate'][i+1] < 1:

                break

    col_regroup2 = [col_regroup[i] for i in range(len(col_regroup)) if i not in del_index] #此处就是从新遍历取出得到一个新的箱子列表，把合并掉的箱子去除掉

    newGroup = {}

    for i in range(len(col_regroup2)):

        for g2 in col_regroup2[i]: #这个其实就是把每个箱子取出来，在箱子的名字前面加上bin然后组成一个新的newGroup并返回这个字典

            newGroup[g2] = 'Bin '+str(i) #g2是该箱子的名称作为key，对应value是bin几号几号之类的

    return newGroup



def BadRateEncoding(df, col, target):

    '''

    :param df: dataframe containing feature and target

    :param col: the feature that needs to be encoded with bad rate, usually categorical type

    :param target: good/bad indicator

    :return: the assigned bad rate to encode the categorical feature

    '''

    regroup = BinBadRate(df, col, target, grantRateIndicator=0)[1] #return (dicts, regroup)取regroup这个dataframe

    br_dict = regroup[[col,'bad_rate']].set_index([col]).to_dict(orient='index')#先选中这两列，然后把第一列当做索引替换掉原来的默认索引，再转成字典{u'car': {'bad_rate': 0.07731958762886598}}orient指定key吧，默认状态时u'car'和'bad_rate'位置调换

    for k, v in br_dict.items():

        br_dict[k] = v['bad_rate'] #把{u'car': {'bad_rate': 0.07731958762886598}}变成了{u'car': 0.07731958762886598}

    badRateEnconding = df[col].map(lambda x: br_dict[x]) #对这一列每个元素都在该字典里找到对应key的value也就是bad rate然后赋值给badRateEncoding

    return {'encoding':badRateEnconding, 'bad_rate':br_dict}#把上面新的br_dict字典和编码好的一列放在一个字典里，{'encoding':2144     0.108294

                                                                                                                           #14805    0.108294

                                                                                                                #,'bad_rate': {u'debt_consolidation': 0.10829405019747738,u'car': 0.07731958762886598}



        

#数值型变量

num_features = ['int_rate_clean','emp_length_clean','annual_inc', 'dti', 'delinq_2yrs', 'earliest_cr_to_app','inq_last_6mths', \

                'mths_since_last_record_clean', 'mths_since_last_delinq_clean','open_acc','pub_rec','total_acc','limit_income','earliest_cr_to_app']

#类别型变量

cat_features = ['home_ownership', 'verification_status','desc_clean', 'purpose', 'zip_code','addr_state','pub_rec_bankruptcies_clean']



more_value_features = []

less_value_features = []

#第一步 检查类别型变量中，哪些变量取值超过5

for var in cat_features:

    valueCounts = len(set(trainData[var]))

    print (valueCounts)

    if valueCounts > 5:

        more_value_features.append(var)

#取值超过5种的类别型变量，需要先进行bad rate编码，再用卡方分箱进行分箱        

    else:

        less_value_features.append(var)

#取值小于等于5时，若包含好坏样本，无需分箱，只包含一种，需要合并 



merge_bin_dict = {} #存放需要合并的变量及合并方法,{'home_ownership': {u'MORTGAGE': 'Bin 0',

                                                                    #   u'NONE': 'Bin 0',

                                                                    #   u'OTHER': 'Bin 3',

                                                                    #   u'OWN': 'Bin 1',

                                                                    #   u'RENT': 'Bin 2'}}

var_bin_list = []   #由于某个取值没有好/坏样本而需要合并的变量

for col in less_value_features:

    binBadRate = BinBadRate(trainData,col,'y')[0] #该函数返回return (dicts, regroup)，取样本和对应bad rate的字典

    if min(binBadRate.values())==0: #某样本因没有坏样本而合并

        print ('{} need to be combined due to 0 bad rate'.format(col))#这个其实就相当于%s那样把col列名写到{}里打印出来

        combine_bin = MergeBad0(trainData,col,'y') #这里就得到一个盒子名称对应盒子编号的一个字典

        merge_bin_dict[col] = combine_bin

        newVar = col + '_Bin' #新建一列该变量的数据的盒子号

        trainData[newVar] = trainData[col].map(combine_bin) #给数据打上盒子号标签

        var_bin_list.append(newVar)

        

    if max(binBadRate.values()) == 1:    #由于某个取值没有好样本而进行合并

        print ('{} need to be combined due to 0 good rate'.format(col))

        combine_bin = MergeBad0(trainData, col, 'y',direction = 'good')

        merge_bin_dict[col] = combine_bin

        newVar = col + '_Bin'

        trainData[newVar] = trainData[col].map(combine_bin)

        var_bin_list.append(newVar)



#保存merge_bin_dict

# file1 = open(file_add+'merge_bin_dict.pkl','wb')

# pickle.dump(merge_bin_dict,file1)

# file1.close()

    

#less_value_features里剩下不需要合并的变量

less_value_features = [i for i in less_value_features if i + '_Bin' not in var_bin_list]    



# （ii）当取值>5时：用bad rate进行编码，放入连续型变量里

br_encoding_dict = {}   #记录按照bad rate进行编码的变量，及编码方式

for col in more_value_features:

    br_encoding = BadRateEncoding(trainData, col, 'y')    #返回{'encoding':badRateEnconding, 'bad_rate':br_dict}字典

    trainData[col+'_br_encoding'] = br_encoding['encoding'] #在训练集大宽表中加一列编码

    br_encoding_dict[col] = br_encoding['bad_rate']  #把按照bad rate编码的变量和编码方式（哪个类别对应哪个bad rate）记录下来

    num_features.append(col+'_br_encoding')    #然后把大宽表里这个编好码的列名添加到数值型变量列表里，方便下一步卡方分箱

    

# file2 = open(file_add+'br_encoding_dict.pkl','wb')

# pickle.dump(br_encoding_dict,file2)

# file2.close()



### ChiMerge_MaxInterval: split the continuous variable using Chi-square value by specifying the max number of intervals

def ChiMerge(df, col, target, max_interval=5,special_attribute=[],minBinPcnt=0):

    '''

    :param df: 包含目标变量与分箱属性的数据框

    :param col: 需要分箱的属性

    :param target: 目标变量，取值0或1

    :param max_interval: 最大分箱数。如果原始属性的取值个数低于该参数，不执行这段函数

    :param special_attribute: 不参与分箱的属性取值

    :param minBinPcnt：最小箱的占比，默认为0

    :return: 分箱结果

    '''

    colLevels = sorted(list(set(df[col]))) #给这一列先去重再排序,得到一个list

    N_distinct = len(colLevels)

    if N_distinct <= max_interval:  #如果原始属性的取值个数低于max_interval，不执行这段函数

        print ("The number of original levels for {} is less than or equal to max intervals".format(col))

        return colLevels[:-1]

    else:

        if len(special_attribute)>=1:

            df1 = df.loc[df[col].isin(special_attribute)] #把等于这个不参与分箱的值挑出来，例-1

            df2 = df.loc[~df[col].isin(special_attribute)] #‘~’按位取反运算符：对数据的每个二进制位取反,即把1变为0,把0变为1 ,这里的效果是除去-1之外的值

        else:

            df2 = df.copy()

        N_distinct = len(list(set(df2[col]))) #一开始有多少种类别



        # 步骤一: 通过col对数据集进行分组，求出每组的总样本数与坏样本数

        if N_distinct > 100:

            split_x = SplitData(df2, col, 100) #得到把取值超过一百种的数据取舍后取值小于等于100种

            df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x)) #这时候就把多于100种类型的变量简化成100种以内

        else:

            df2['temp'] = df2[col]

        # 总体bad rate将被用来计算expected bad count

        (binBadRate, regroup, overallRate) = BinBadRate(df2, 'temp', target, grantRateIndicator=1)#return (dicts, regroup, overallRate)



        # 首先，每个单独的属性值将被分为单独的一组

        # 对属性值进行排序，然后两两组别进行合并

        colLevels = sorted(list(set(df2['temp']))) #去重（因为之前缩减到100内时产生很多重复值），排序

        groupIntervals = [[i] for i in colLevels] #把每个取值单独放到list里



        # 步骤二：建立循环，不断合并最优的相邻两个组别，直到：

        # 1，最终分裂出来的分箱数<＝预设的最大分箱数

        # 2，每箱的占比不低于预设值（可选）

        # 3，每箱同时包含好坏样本

        # 如果有特殊属性，那么最终分裂出来的分箱数＝预设的最大分箱数－特殊属性的个数

        split_intervals = max_interval - len(special_attribute)

        while (len(groupIntervals) > split_intervals):  # 终止条件: 当前分箱数＝预设的分箱数

            # 每次循环时, 计算合并相邻组别后的卡方值。具有最小卡方值的合并方案，是最优方案

            chisqList = []

            for k in range(len(groupIntervals)-1):

                temp_group = groupIntervals[k] + groupIntervals[k+1] #相邻两组两两合并计算卡方值

                df2b = regroup.loc[regroup['temp'].isin(temp_group)] #把合并的两组的数据从regroup中挑出来

                #chisq = Chi2(df2b, 'total', 'bad', overallRate)

                chisq = Chi2(df2b, 'total', 'bad') #传入数据集，总体样本数，坏样本数

                chisqList.append(chisq) #得到每两两合并的卡方值

            best_comnbined = chisqList.index(min(chisqList)) #得到最小卡方值的角标

            groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined+1]#合并是把第i个bin和i+1合并到i盒里，然后下面就把i+1bin删了

            # after combining two intervals, we need to remove one of them

            groupIntervals.remove(groupIntervals[best_comnbined+1])#删除i+1的bin

        groupIntervals = [sorted(i) for i in groupIntervals] #每个箱子排个序取得其实是分割点(但是没有最后一个箱子)

        cutOffPoints = [max(i) for i in groupIntervals[:-1]]

        

        # 检查是否有箱没有好或者坏样本。如果有，需要跟相邻的箱进行合并，直到每箱同时包含好坏样本

        groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints)) #得到每个数对应的盒子号

        df2['temp_Bin'] = groupedvalues #新加一列盒子号

        (binBadRate,regroup) = BinBadRate(df2, 'temp_Bin', target) #给盒子号进去看看bad rate，得到每一箱的坏样本率

        [minBadRate, maxBadRate] = [min(binBadRate.values()),max(binBadRate.values())] #

        while minBadRate ==0 or maxBadRate == 1:

            # 找出全部为好／坏样本的箱

            indexForBad01 = regroup[regroup['bad_rate'].isin([0,1])].temp_Bin.tolist()#结果是全为好坏样本的箱子号（'Bin x'）

            bin=indexForBad01[0]

            # 如果是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除

            if bin == max(regroup.temp_Bin):

                cutOffPoints = cutOffPoints[:-1]

            # 如果是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除,因为切分点的值是每个箱子的最大值，是一个具体的数值

            elif bin == min(regroup.temp_Bin):

                cutOffPoints = cutOffPoints[1:]

            # 如果是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值

            else:

                # 和前一箱进行合并，并且计算卡方值

                currentIndex = list(regroup.temp_Bin).index (bin) #得到这个箱子的下标

                prevIndex = list(regroup.temp_Bin)[currentIndex - 1] #前一箱

                df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, bin])] #把这两箱的数据取出来

                (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)#返回这两个箱子的bad rate和一个regroup的DataFrame

                #chisq1 = Chi2(df2b, 'total', 'bad', overallRate)

                chisq1 = Chi2(df2b, 'total', 'bad')#计算出卡方值

                # 和后一箱进行合并，并且计算卡方值

                laterIndex = list(regroup.temp_Bin)[currentIndex + 1]

                df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, bin])]

                (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)

                #chisq2 = Chi2(df2b, 'total', 'bad', overallRate)

                chisq2 = Chi2(df2b, 'total', 'bad')

                if chisq1 < chisq2:

                    cutOffPoints.remove(cutOffPoints[currentIndex - 1])

                else:

                    cutOffPoints.remove(cutOffPoints[currentIndex])

            # 完成合并之后，需要再次计算新的分箱准则下，每箱是否同时包含好坏样本

            groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints)) #重新标记一下盒子号

            df2['temp_Bin'] = groupedvalues

            (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)

            [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]

        # 需要检查分箱后的最小占比

        if minBinPcnt > 0: #这个是函数里初始值，不传进来就默认为0

            groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))

            df2['temp_Bin'] = groupedvalues

            valueCounts = groupedvalues.value_counts().to_frame() #这个求出来是每个箱子里有多少个数值

            valueCounts['pcnt'] = valueCounts['temp'].apply(lambda x: x * 1.0 / N) #前面binbadrate函数倒是定义了一个局部变量N = sum(regroup['total'])

            valueCounts = valueCounts.sort_index() #按照索引排序

            minPcnt = min(valueCounts['pcnt']) #找出占比最小的盒子的占比

            while minPcnt < minBinPcnt and len(cutOffPoints) > 2: #看看最小占比的箱子是否小于预设值，再看切分点是否大于2（也就是至少有三个箱子），这样保证合并后最起码还有两个箱子

                # 找出占比最小的箱

                indexForMinPcnt = valueCounts[valueCounts['pcnt'] == minPcnt].index.tolist()[0] #最小值箱子的索引

                # 如果占比最小的箱是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除

                if indexForMinPcnt == max(valueCounts.index):

                    cutOffPoints = cutOffPoints[:-1]

                # 如果占比最小的箱是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除

                elif indexForMinPcnt == min(valueCounts.index):

                    cutOffPoints = cutOffPoints[1:]

                # 如果占比最小的箱是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值

                else:

                    # 和前一箱进行合并，并且计算卡方值

                    currentIndex = list(valueCounts.index).index(indexForMinPcnt) #得到的是该最小占比箱子的索引的下标

                    prevIndex = list(valueCounts.index)[currentIndex - 1] #最小占比箱子前面箱子的索引

                    df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, indexForMinPcnt])]

                    (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target) #return（binbadrate，regroup）

                    #chisq1 = Chi2(df2b, 'total', 'bad', overallRate)

                    chisq1 = Chi2(df2b, 'total', 'bad')

                    # 和后一箱进行合并，并且计算卡方值

                    laterIndex = list(valueCounts.index)[currentIndex + 1]

                    df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, indexForMinPcnt])]

                    (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)

                    #chisq2 = Chi2(df2b, 'total', 'bad', overallRate)

                    chisq2 = Chi2(df2b, 'total', 'bad')

                    if chisq1 < chisq2:

                        cutOffPoints.remove(cutOffPoints[currentIndex - 1]) #如果跟前面箱子合并的卡方值小，则需要删除该最小占比箱索引-1的切分点的索引对应的切分值

                    else:

                        cutOffPoints.remove(cutOffPoints[currentIndex])

                        #与后面箱子合并卡方值较小的话，本来切分点就会比箱子数少一个，跟后面的合并，索引号就正好是上面那个跟前面箱子合并时删除切分点索引+1就正好等于该最小占比箱子的索引

        cutOffPoints = special_attribute + cutOffPoints

        return cutOffPoints

    

def SplitData(df, col, numOfSplit, special_attribute=[]):

    '''

    :param df: 按照col排序后的数据集

    :param col: 待分箱的变量

    :param numOfSplit: 切分的组别数

    :param special_attribute: 在切分数据集的时候，某些特殊值需要排除在外

    :return: 在原数据集上增加一列，把原始细粒度的col重新划分成粗粒度的值，便于分箱中的合并处理

    '''

    df2 = df.copy()

    if special_attribute != []: #如果有需要排除的特殊值

        df2 = df.loc[~df[col].isin(special_attribute)] #排除掉特殊值后的值

    N = df2.shape[0] #得到行数

    n = N/numOfSplit #行数除以100

    splitPointIndex = [i*n for i in range(1,numOfSplit)] #其实就是切分点的索引，假设N=200，n就=2，那这行代码执行结果就是2,4,6.。。到200

    rawValues = sorted(list(df2[col])) #排个序

    splitPoint = [rawValues[int(i)] for i in splitPointIndex] #从中取出对应切分点索引的数值

    splitPoint = sorted(list(set(splitPoint))) #去重，排序并返回就把所有超过100种取值的数值型变量取到100种

    return splitPoint



def AssignGroup(x, bin): #df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))

    '''

    :param x: 某个变量的某个取值

    :param bin: 上述变量的分箱结果

    :return: x在分箱结果下的映射

    '''

    N = len(bin)

    if x<=min(bin):

        return min(bin) #如果原数据比简化后的数据最小值小，则取数据最小值

    elif x>max(bin):

        return 10e10 #为什么大于最大值要返回一个这么大的值？而不是返回max(bin)，猜想有可能是因为只有收入这个变量需要这样处理，因此这个数就是针对收入变量的

    else:

        for i in range(N-1):

            if bin[i] < x <= bin[i+1]:#把区间内的多类数都变成区间右端的相同数

                return bin[i+1]



# def Chi2(df, total_col, bad_col, overallRate):

#     '''

#     :param df: 包含全部样本总计与坏样本总计的数据框

#     :param total_col: 全部样本的个数

#     :param bad_col: 坏样本的个数

#     :param overallRate: 全体样本的坏样本占比

#     :return: 卡方值

#     '''

#     df2 = df.copy()

#     # 期望坏样本个数＝全部样本个数*平均坏样本占比

#     df2['expected'] = df[total_col].apply(lambda x: x*overallRate)

#     combined = zip(df2['expected'], df2[bad_col])

#     chi = [(i[0]-i[1])**2/i[0] for i in combined]

#     chi2 = sum(chi)

#     return chi2





def Chi2(df, total_col, bad_col):

    '''

    :param df: 包含全部样本总计与坏样本总计的数据框

    :param total_col: 全部样本的个数

    :param bad_col: 坏样本的个数

    :return: 卡方值

    '''

    df2 = df.copy()

    # 求出df中，总体的坏样本率和好样本率

    badRate = sum(df2[bad_col])*1.0/sum(df2[total_col]) #合并后的这两个取值（箱子）总体对应的坏样本数除以总样本数

    # 当全部样本只有好或者坏样本时，卡方值为0

    if badRate in [0,1]: #这种情况合并出的箱子不符合同时包含好坏样本，还得继续合并

        return 0

    df2['good'] = df2.apply(lambda x: x[total_col] - x[bad_col], axis = 1)

    goodRate = sum(df2['good']) * 1.0 / sum(df2[total_col]) #乘以1.0是为了除出来带小数，不然就会是0，总体的好样本率

    # 期望坏（好）样本个数＝全部样本个数*平均坏（好）样本占比

    df2['badExpected'] = df[total_col].apply(lambda x: x*badRate)#这两个取值（bin）合并后组成的新箱子里两个值得坏样本的期望值=各取值的样本数乘以总体的坏样本率

    df2['goodExpected'] = df[total_col].apply(lambda x: x * goodRate)#同理

    badCombined = zip(df2['badExpected'], df2[bad_col]) #得到的其实是两个取值对应的tuple（坏样本期望，坏样本数）

    goodCombined = zip(df2['goodExpected'], df2['good'])

    badChi = [(i[0]-i[1])**2/i[0] for i in badCombined] #这个求出来的是第一个区间和第二个区间（取值）的坏样本卡方值，相当于这样计算结果的list[((A11-E11)**2)/E11,((A21-E21)**2)/E21]           

    goodChi = [(i[0]-i[1])**2/i[0] for i in goodCombined] #这个求出来的是第一个区间和第二个区间（取值）的坏样本卡方值，相当于这样计算结果的list[((A12-E12)**2)/E12,((A22-E22)**2)/E22]           

    chi2 = sum(badChi) + sum(goodChi) #这两个list求和之后再求和就得到这两个区间合并的卡方值了

    return chi2



#Chi2 的另外一种计算方法

# def Chi2(df, total_col, bad_col):

#     df2 = df.copy()

#     df2['good'] = df2[total_col] - df2[bad_col]

#     goodTotal = sum(df2['good'])

#     badTotal = sum(df2[bad_col])

#     p1 = df2.loc[0]['good']*1.0/df2.loc[0][total_col]

#     p2 = df2.loc[1]['good']*1.0/df2.loc[1][total_col]

#     w1 = df2.loc[0]['good']*1.0/goodTotal

#     w2 = df2.loc[0][bad_col]*1.0/badTotal

#     N = sum(df2[total_col])

#     return N*(p1-p2)*(w1-w2)



def AssignBin(x, cutOffPoints,special_attribute=[]):#，可以看关于这个函数理解的笔记，这个函数的作用其实就是把每个元素贴上对应几号箱子的标签

    '''

    :param x: 某个变量的某个取值

    :param cutOffPoints: 上述变量的分箱结果，用切分点表示

    :param special_attribute:  不参与分箱的特殊取值

    :return: 分箱后的对应的第几个箱，从0开始

    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3

    '''

    numBin = len(cutOffPoints) + 1 + len(special_attribute) 

    #cutOffPoints = [max(i) for i in groupIntervals[:-1]]这个取得是箱子分割点，因此这里要加1才能等于箱子数

    if x in special_attribute:

        i = special_attribute.index(x)+1

        return 'Bin {}'.format(0-i) #特殊值的标记箱子号是负着标的，例如第一个特殊值对应的箱子号是-1，第二个特殊值对应的箱子号是-2

    if x<=cutOffPoints[0]:#如果给的这列数据集数据小于第一个分割点，就给这些数标上0号箱子

        return 'Bin 0'

    elif x > cutOffPoints[-1]: #给这列数里大于最后一个分割点的数标上最后一个箱子

        return 'Bin {}'.format(numBin-1) #假设numBin=6，此时大于最后分割点的值被安排在最后这个盒子里这5个盒子是从0号到5号

    else:

        for i in range(0,numBin-1):#name这里就是range(0,4),取值0,1,2,3

            if cutOffPoints[i] < x <=  cutOffPoints[i+1]:#看笔记本上的图示就很容易理解了

                return 'Bin {}'.format(i+1)



## 判断某变量的坏样本率是否单调

def BadRateMonotone(df, sortByVar, target,special_attribute = []):

    '''

    :param df: 包含检验坏样本率的变量，和目标变量

    :param sortByVar: 需要检验坏样本率的变量

    :param target: 目标变量，0、1表示好、坏

    :param special_attribute: 不参与检验的特殊值

    :return: 坏样本率单调与否

    '''

    df2 = df.loc[~df[sortByVar].isin(special_attribute)]#得到不包括特殊值的数据

    if len(set(df2[sortByVar])) <= 2: #如果箱子数为2那就肯定符合bad rate单调性

        return True

    regroup = BinBadRate(df2, sortByVar, target)[1] #取出函数返回的binbadreta，regroup中的后者

    combined = zip(regroup['total'],regroup['bad'])#搞成两列元组组合（由盒子号对应的bad和总的bad）

    badRate = [x[1]*1.0/x[0] for x in combined] #得到每个盒子的bad rate

    #这个就是不单调的条件

    badRateNotMonotone = [badRate[i]<badRate[i+1] and badRate[i] < badRate[i-1] or badRate[i]>badRate[i+1] and badRate[i] > badRate[i-1]

                       for i in range(1,len(badRate)-1)]#两头不取

    if True in badRateNotMonotone:

        return False

    else:

        return True





def CalcWOE(df, col, target):

    '''

    :param df: 包含需要计算WOE的变量和目标变量

    :param col: 需要计算WOE、IV的变量，必须是分箱后的变量，或者不需要分箱的类别型变量

    :param target: 目标变量，0、1表示好、坏

    :return: 返回WOE和IV

    '''

    total = df.groupby([col])[target].count()

    total = pd.DataFrame({'total': total}) #每个箱子里有多少个样本

    bad = df.groupby([col])[target].sum() #每个箱子里有多少个bad样本（'y'变量为1）

    bad = pd.DataFrame({'bad': bad})

    regroup = total.merge(bad, left_index=True, right_index=True, how='left') #左连接

    regroup.reset_index(level=0, inplace=True)#重设索引

    N = sum(regroup['total'])#总体样本数

    B = sum(regroup['bad'])#总的坏样本数

    regroup['good'] = regroup['total'] - regroup['bad'] #每个箱子里好样本个数

    G = N - B #总的好样本数

    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B) #WOE编码公式的分母（也可以是分子，在一个模型里统一就行）

    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)#分子

    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1) #得到每一箱的WOE值

    WOE_dict = regroup[[col,'WOE']].set_index(col).to_dict(orient='index')

    #取这一列（也就是某变量箱子号列），和对应的WOE值，然后设置盒子号列为索引，然后转成字典{u'car': {'WOE': 0.07731958762886598}}

    for k, v in WOE_dict.items():

        WOE_dict[k] = v['WOE']#把{u'car': {'WOE': 0.07731958762886598}}变成了{u'car': 0.07731958762886598}盒子号：woe值

    IV = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)#得到每一箱的IV值

    IV = sum(IV) #把该变量的所有箱的IV加起来得到该变量的IV

    return {"WOE": WOE_dict, 'IV':IV}
# （iii）对连续型变量进行分箱，包括（ii）中的变量

continous_merged_dict = {}

for col in num_features:

    print ("{} is in processing".format(col))

    if -1 not in set(trainData[col]):   #－1会当成特殊值处理。如果没有－1，则所有取值都参与分箱

        max_interval = 5   #分箱后的最多的箱数

        cutOff = ChiMerge(trainData, col, 'y', max_interval=max_interval,special_attribute=[],minBinPcnt=0) #返回的切分点包含特殊值（-1）

        trainData[col+'_Bin'] = trainData[col].map(lambda x: AssignBin(x, cutOff,special_attribute=[])) #此时就按照这样处理好的切分点来分箱并给每个数打上箱号

        monotone = BadRateMonotone(trainData, col+'_Bin', 'y')   # 检验分箱后的单调性是否满足

        while(not monotone):#如果不满足单调性

            # 检验分箱后的单调性是否满足。如果不满足，则缩减分箱的个数。

            max_interval -= 1

            cutOff = ChiMerge(trainData, col, 'y', max_interval=max_interval, special_attribute=[],minBinPcnt=0)#再用卡方分箱发按照最大箱少一个的限制条件从新分箱

            trainData[col + '_Bin'] = trainData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[]))#再从新打上箱子号

            if max_interval == 2:

                # 当分箱数为2时，必然单调

                break

            monotone = BadRateMonotone(trainData, col + '_Bin', 'y')#看看最大分箱数减少一个后是否满足bad rate单调性，如果满足就结束循环

        newVar = col + '_Bin'

        trainData[newVar] = trainData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[]))

        var_bin_list.append(newVar)#这样，没有-1特殊值的数值型变量就分箱完了，把这些变量对应的分箱号的列名存起来

    else:

        max_interval = 5

        # 如果有－1，则除去－1后，其他取值参与分箱

        cutOff = ChiMerge(trainData, col, 'y', max_interval=max_interval, special_attribute=[-1],minBinPcnt=0)#得到切分点

        trainData[col + '_Bin'] = trainData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))#打上各列箱子名称

        monotone = BadRateMonotone(trainData, col + '_Bin', 'y',['Bin -1'])#传入的最后一个参数是特殊值

        while (not monotone):

            max_interval -= 1

            # 如果有－1，－1的bad rate不参与单调性检验

            cutOff = ChiMerge(trainData, col, 'y', max_interval=max_interval, special_attribute=[-1],

                                          minBinPcnt=0)

            trainData[col + '_Bin'] = trainData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))

            if max_interval == 3:

                # 当分箱数为3-1=2时，必然单调，因为有一箱是特殊值-1

                break

            monotone = BadRateMonotone(trainData, col + '_Bin', 'y',['Bin -1'])

        newVar = col + '_Bin'

        trainData[newVar] = trainData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))

        var_bin_list.append(newVar)#这样，有-1特殊值的数值型变量就分箱完了，把这些变量对应的分箱号的列名存起来

    continous_merged_dict[col] = cutOff #这个字典存的是{列名：[切分点],列名：[切分点]}



# file3 = open(file_add+'continous_merged_dict.pkl','wb')

# pickle.dump(continous_merged_dict,file3)

# file3.close()

'''

第四步：WOE编码、计算IV

'''

WOE_dict = {}

IV_dict = {}

# 分箱后的变量进行编码，包括：

# 1，初始取值个数小于5，且不需要合并的类别型变量。存放在less_value_features中

# 2，初始取值个数小于5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中

# 3，初始取值个数超过5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中

# 4，连续变量。分箱后新的变量存放在var_bin_list中

all_var = var_bin_list  + less_value_features #var_bin_list里都是些变量名_bin（变量名+盒）某个变量的盒子号列

for var in all_var:

    woe_iv = CalcWOE(trainData, var, 'y') #return {"WOE": WOE_dict, 'IV':IV}

    WOE_dict[var] = woe_iv['WOE']

    IV_dict[var] = woe_iv['IV']





# file4 = open(file_add+'WOE_dict.pkl','wb')

# pickle.dump(WOE_dict,file4)

# file4.close()





#将变量IV值进行降序排列，方便后续挑选变量

IV_dict_sorted = sorted(IV_dict.items(), key=lambda x: x[1], reverse=True) #以x[1]作为排序依据，默认升序，reverse是倒序



IV_values = [i[1] for i in IV_dict_sorted]#把IV值取出到list里

IV_name = [i[0] for i in IV_dict_sorted]#把对应IV值得变量名取出到list里

plt.title('feature IV')

plt.bar(range(len(IV_values)),IV_values)



'''

第五步：单变量分析和多变量分析，均基于WOE编码后的值。

（1）选择IV高于0.01的变量

（2）比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个

'''



#选取IV>0.02的变量

high_IV = {k:v for k, v in IV_dict.items() if v >= 0.02}

high_IV_sorted = sorted(high_IV.items(),key=lambda x:x[1],reverse=True)



short_list = high_IV.keys() #符合条件的IV值对应的变量名list

short_list_2 = []

for var in short_list:

    newVar = var + '_WOE' #新的变量名=旧的加上后缀

    trainData[newVar] = trainData[var].map(WOE_dict[var]) #{u'car': 0.07731958762886598}盒子号对应woe值，添加一列不同盒子号下WOE值

    short_list_2.append(newVar) #符合条件IV值对应变量加上woe后缀后的新名字组成的list



#对于上一步的结果，计算相关系数矩阵，并画出热力图进行数据可视化

trainDataWOE = trainData[short_list_2]#这是一个只有WOE值的DataFrame

f, ax = plt.subplots(figsize=(10, 8))#fig代表绘图窗口（Figure）其实就是这个图 ，ax代表这个绘图窗口上的坐标系（axes），figsize是每英寸的宽度和高度

corr = trainDataWOE.corr()#相关系数矩阵，index和columns是变量名

#heatmap:

# data:矩阵数据集，可以是numpy的数组（array），也可以是pandas的DataFrame。如果是DataFrame，则df的index/column信息会分别对应到heatmap的columns和rows，即pt.index是热力图的行标，pt.columns是热力图的列标

# mask:控制某个矩阵块是否显示出来。默认值是None。如果是布尔型的DataFrame，则将DataFrame里True的位置用白色覆盖掉 

# cmap:从数字到色彩空间的映射，取值是matplotlib包里的colormap名称或颜色对象，或者表示颜色的列表；改参数默认值：根据center参数设定

# square:设置热力图矩阵小块形状，默认值是False ,True就会得到正方形

# ax:设置作图的坐标轴，一般画多个子图时需要修改不同的子图的该值

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10,as_cmap=True),square=True, ax=ax)

    #np.zeros_like(corr, dtype=np.bool)意思是判断corr这个相关系数矩阵中的值是否等于0，用True/False表示

#创建分散颜色diverging_palette(h_neg, h_pos, s=75, l=50, sep=10, n=6, center='light', as_cmap=False)

# h_neg和h_pos：起始，终止颜色值

    # optional（可选参数）：

# s：饱和度

# l：亮度

# n:调色板中的颜色数（如果不返回cmap）

# center:调色板中心是亮是暗（调色板其实就是右边那个条），默认light

# as_cmap：如果为true，则返回一个matplotlib颜色映射对象，而不是一个颜色列表,这个函数单独运行一下就知道了



#两两间的线性相关性检验

#1，将候选变量按照IV进行降序排列

#2，计算第i和第i+1的变量的线性相关系数

#3，对于系数超过阈值的两个变量，剔除IV较低的一个



#单变量分析

deleted_index = []

cnt_vars = len(high_IV_sorted)#字典取len就是有几个键值对

#计算每个变量与其他变量的

for i in range(cnt_vars):

    if i in deleted_index:

        continue

    x1 = high_IV_sorted[i][0]+"_WOE"#第一个括号是取第几对，第二个括号是取得是key还是value，取key+woe后缀

    for j in range(cnt_vars):

        if i == j or j in deleted_index:

            continue

        y1 = high_IV_sorted[j][0]+"_WOE"#取得跟前面取不同的key加上woe后缀

        #求一列一列的相关系数，跟pd.corr()不同的是这个返回的是array，corr返回DataFrame

        roh = np.corrcoef(trainData[x1],trainData[y1])[0,1] #此处是两个变量相比较，求出系数矩阵后取第一行第二列数值

        if abs(roh)>0.7:#取绝对值，需要在相关性较高的两个变量里面舍弃一个

            x1_IV = high_IV_sorted[i][1]#取出符合条件的对应value也就是IV值

            y1_IV = high_IV_sorted[j][1]

            if x1_IV > y1_IV:#进行IV值比较，舍弃IV值较小的变量

                deleted_index.append(j)

            else:

                deleted_index.append(i)



multi_analysis_vars_1 = [high_IV_sorted[i][0]+"_WOE" for i in range(cnt_vars) if i not in deleted_index]

#从新取一遍，不取被舍弃掉的变量





'''

多变量分析：VIF

'''

X = np.matrix(trainData[multi_analysis_vars_1])#把这些经过woe编码并且通过变量两两相关性分析的变量组成一个matrix

VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]#一列一列的求

max_VIF = max(VIF_list)

print (max_VIF)

# 最大的VIF是1.32267733123，因此这一步认为没有多重共线性，一般大于10

multi_analysis = multi_analysis_vars_1



'''

第六步：逻辑回归模型。

要求：

1，变量显著

2，符号为负

'''

### (1)将多变量分析的后变量带入LR模型中

y = trainData['y']

X = trainData[multi_analysis]

X['intercept'] = [1]*X.shape[0]#截距项





LR = sm.Logit(y, X).fit()#python中可以用sklearn中的LogisticRegression 或者statsmodels中的Logit 进行拟合

summary = LR.summary() #模型的基本信息概况

pvals = LR.pvalues #各变量的P值

pvals = pvals.to_dict()#字典



# ### 有些变量不显著，需要逐步剔除

varLargeP = {k: v for k,v in pvals.items() if v >= 0.1}#P值大的变量不显著，要剔除

varLargeP = sorted(varLargeP.items(), key=lambda d:d[1], reverse = True)#对筛选出的P值大于0.1的键值对按P值降序排列

while(len(varLargeP) > 0 and len(multi_analysis) > 0):

    # 每次迭代中，剔除最不显著的变量，直到

    # (1) 剩余所有变量均显著

    # (2) 没有特征可选

    varMaxP = varLargeP[0][0]#取出第一个键值对的key

    print (varMaxP)

    if varMaxP == 'intercept':

        print ('the intercept is not significant!')

        break

    multi_analysis.remove(varMaxP)#去掉这个变量，然后重新拟合

    y = trainData['y']

    X = trainData[multi_analysis]

    X['intercept'] = [1] * X.shape[0]#一般不加入截距项的回归模型的估计被认为是有偏的，截距项据说会影响VIF的R平方值



    LR = sm.Logit(y, X).fit()

    pvals = LR.pvalues

    pvals = pvals.to_dict()

    varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}

    varLargeP = sorted(varLargeP.items(), key=lambda d: d[1], reverse=True)



summary = LR.summary()



trainData['prob'] = LR.predict(X)

auc = roc_auc_score(trainData['y'],trainData['prob'])  #AUC = 0.73

#AUC表示，随机抽取一个正样本和一个负样本，分类器正确给出正样本的score高于负样本的概率。

print('AUC:',auc)



#将模型保存

# saveModel =open(file_add+'LR_Model_Normal.pkl','wb')

# pickle.dump(LR,saveModel)

# saveModel.close()

def KS(df, score, target):

    '''

    :param df: 包含目标变量与预测值的数据集

    :param score: 得分或者概率，在这是预测为坏样本的概率

    :param target: 目标变量

    :return: KS值

    '''

    total = df.groupby([score])[target].count()#得到总样本数

    bad = df.groupby([score])[target].sum()#得到预测坏样本数

    all = pd.DataFrame({'total':total, 'bad':bad})

    all['good'] = all['total'] - all['bad']

    all[score] = all.index

    all = all.sort_values(by=score,ascending=False)#降序排列

    all.index = range(len(all))

    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()#cumsum函数是累计和

    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()

    KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)

    return max(KS)



def Prob2Score(prob, basePoint, PDO):

    #将概率转化成分数且为正整数

    y = np.log(prob/(1-prob))

    return int(basePoint+PDO/np.log(2)*(-y))#返回得分



def MergeByCondition(x,condition_list):

    #condition_list是条件列表。满足第几个condition，就输出几

    s = 0

    for condition in condition_list:

        if eval(str(x)+condition):

            return s

        else:

            s+=1

    return s

def ModifyDf(x, new_value):

    if np.isnan(x):

        return new_value

    else:

        return x



'''

将模型应用在测试数据集上

'''



# testDataFile = open(file_add+'testData.pkl','rb')

# testData = pickle.load((testDataFile))

# testDataFile.close()



'''

第一步：完成数据预处理

在实际工作中，可以只清洗模型实际使用的字段

'''



# 将带％的百分比变为浮点数

testData['int_rate_clean'] = testData['int_rate'].map(lambda x: float(x.replace('%',''))/100)



# 将工作年限进行转化，否则影响排序

testData['emp_length_clean'] = testData['emp_length'].map(CareerYear)



# 将desc的缺失作为一种状态，非缺失作为另一种状态

testData['desc_clean'] = testData['desc'].map(DescExisting)



# 处理日期。earliest_cr_line的格式不统一，需要统一格式且转换成python的日期

testData['app_date_clean'] = testData['issue_d'].map(lambda x: ConvertDateStr(x))

testData['earliest_cr_line_clean'] = testData['earliest_cr_line'].map(lambda x: ConvertDateStr(x))



# 处理mths_since_last_delinq。注意原始值中有0，所以用－1代替缺失

testData['mths_since_last_delinq_clean'] = testData['mths_since_last_delinq'].map(lambda x:MakeupMissing(x))



testData['mths_since_last_record_clean'] = testData['mths_since_last_record'].map(lambda x:MakeupMissing(x))



testData['pub_rec_bankruptcies_clean'] = testData['pub_rec_bankruptcies'].map(lambda x:MakeupMissing(x))



'''

第二步：变量衍生

'''

# 考虑申请额度与收入的占比

testData['limit_income'] = testData.apply(lambda x: x.loan_amnt / x.annual_inc, axis = 1)



# 考虑earliest_cr_line到申请日期的跨度，以月份记

testData['earliest_cr_to_app'] = testData.apply(lambda x: MonthGap(x.earliest_cr_line_clean,x.app_date_clean), axis = 1)



'''

第三步：分箱并代入WOE值

'''

# modelFile =open(file_add+'LR_Model_Normal.pkl','rb')

# LR = pickle.load(modelFile)

# modelFile.close()



#对变量的处理只需针对入模变量即可

var_in_model = list(LR.pvalues.index)

var_in_model.remove('intercept')#结果如下：

                                #['zip_code_br_encoding_Bin_WOE',

                                #  'int_rate_clean_Bin_WOE',

                                #  'annual_inc_Bin_WOE',

                                #  'purpose_br_encoding_Bin_WOE',

                                #  'inq_last_6mths_Bin_WOE',

                                #  'mths_since_last_record_clean_Bin_WOE',

                                #  'limit_income_Bin_WOE',

                                #  'dti_Bin_WOE',

                                #  'emp_length_clean_Bin_WOE']





# file1 = open(file_add+'merge_bin_dict.pkl','rb')

# merge_bin_dict = pickle.load(file1)

# file1.close()





# file2 = open(file_add+'br_encoding_dict.pkl','rb')

# br_encoding_dict = pickle.load(file2)

# file2.close()



# file3 = open(file_add+'continous_merged_dict.pkl','rb')

# continous_merged_dict = pickle.load(file3)

# file3.close()



# file4 = open(file_add+'WOE_dict.pkl','rb')

# WOE_dict = pickle.load(file4)

# file4.close()



for var in var_in_model:

    var1 = var.replace('_Bin_WOE','')#先把变量里这个后缀去掉



    # 有些取值个数少、但是需要合并的变量

    if var1 in merge_bin_dict.keys():#一个存放需要合并的变量及合并方法的字典

        print ("{} need to be regrouped".format(var1))

        testData[var1 + '_Bin'] = testData[var1].map(merge_bin_dict[var1])



    # 有些变量需要用bad rate进行编码

    if var1.find('_br_encoding')>-1:#find是寻找字符串，找到返回该字符串，找不到返回-1

        var2 =var1.replace('_br_encoding','')#把找到的带这个后缀的变量去掉这个后缀的list

        print ("{} need to be encoded by bad rate".format(var2))

        testData[var1] = testData[var2].map(br_encoding_dict[var2])#记录按照bad rate进行编码的{变量1，{值：编码}，变量2，{值：编码}}先从这个字典里取出这某变量，

        #就得到该变量每个值和各自对应的编码值，这一步其实就是寻找test集里和之前编过的变量值相等的值，直接传递给它，就不用再编码，剩下那些匹配不到的然后如下一行所说

        #需要注意的是，有可能在测试样中某些值没有出现在训练样本中，从而无法得出对应的bad rate是多少。故可以用最坏（即最大）的bad rate进行编码

        max_br = max(testData[var1])#先找到这些匹配过值的变量的bad rate编码最大值

        testData[var1] = testData[var1].map(lambda x: ModifyDf(x, max_br))#这个函数很简单，如果是空值则返回max_br，其实这一步就是把上一步没有匹配到的空值进行了一个最大值的填充





    #上述处理后，需要加上连续型变量一起进行分箱

    if -1 not in set(testData[var1]):#不含特殊值的变量打箱子号的处理

        testData[var1+'_Bin'] = testData[var1].map(lambda x: AssignBin(x, continous_merged_dict[var1]))

    else:

        testData[var1 + '_Bin'] = testData[var1].map(lambda x: AssignBin(x, continous_merged_dict[var1],[-1]))



    #WOE编码

    var3 = var.replace('_WOE','')

    testData[var] = testData[var3].map(WOE_dict[var3])#这一步其实就是把上一步箱子的编号对应一下woe值





'''

第四步：将WOE值代入LR模型，计算概率和分数

'''

testData['intercept'] = [1]*testData.shape[0]

#预测数据集中，变量顺序需要和LR模型的变量顺序一致

#例如在训练集里，变量在数据中的顺序是“负债比”在“借款目的”之前，对应地，在测试集里，“负债比”也要在“借款目的”之前

testData2 = testData[list(LR.params.index)]

testData['prob'] = LR.predict(testData2)#得到的是为坏样本的概率



#计算KS和AUC

auc = roc_auc_score(testData['y'],testData['prob'])

#评价模型的准确度，AUC的预测是需要提供真实值和预测值，得到混淆矩阵中FPR和TPR根据不同的阈值得到一组（FPR,TPR）坐标点形成ROC曲线，曲线下的面积就是AUC

ks = KS(testData, 'prob', 'y')#评价模型的区分度，累计好坏样本占总好坏样本的比例，KS越大分布差异越大，区分力越强

print('AUC:',auc)

print('KS:',ks)



#评分卡分数的计算

basePoint = 250 #基准分

PDO = 200 #好坏比每升高一倍，分数升高PDO

testData['score'] = testData['prob'].map(lambda x:Prob2Score(x, basePoint, PDO))

testData = testData.sort_values(by = 'score')#根据分数排序，默认升序





#画出得分的分布图

plt.hist(testData['score'], 100)#直方图，间隔100

plt.xlabel('score')

plt.ylabel('freq')

plt.title('distribution')