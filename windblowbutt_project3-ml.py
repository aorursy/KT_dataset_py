# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
data = pd.read_csv('../input/adult.csv')

data.describe()
data.loc[data['income']=='<=50K', 'income']=-1
data.loc[data['income']=='>50K', 'income']=1
data.rename(columns={'income':'is_income_more_than_50K'}, inplace=True)
import seaborn as sns
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'
data = data.replace('?', np.nan).dropna()
data.head()
# sns.set(rc={'figure.figsize':(13,5)})
# sns.countplot(x='education.num', data=data)
# sns.countplot(x='marital.status', data=data)
# sns.countplot(x='education', data=data)
cleaned_data= data.copy()
cleaned_data['capital_profit'] = cleaned_data['capital.gain']- cleaned_data['capital.loss']
cleaned_data.replace(
    {'marital.status': 
     {'Married-spouse-absent':7, 'Married-civ-spouse': 6, 'Married-AF-spouse': 5,'Divorced':2,'Never-married':4,'Separated':3,
      'Widowed':1
     },
     'relationship':{'Wife':1,'Own-child':1,'Husband':2, 'Not-in-family':3, 'Other-relative':4, 'Unmarried':5
     },
     'workclass':{'Private':3,'Self-emp-not-inc':3,'Self-emp-inc':3,'Federal-gov':2,'Local-gov':2,'State-gov':2,
                  'Without-pay':1,'Never-worked':1
    },
     'sex':{'Female':1,'Male':2
    },

    }, inplace=True)

# is_complete_family = cleaned_data['marital.status'].isin(['Married-civ-spouse','Married-AF-spous'])
# cleaned_data.loc[is_complete_family, 'marital.status']=1
# cleaned_data.loc[~is_complete_family, 'marital.status']=0

cleaned_data.drop(['education','native.country','capital.loss','capital.gain','race','occupation'], axis=1, inplace=True)
cleaned_data=cleaned_data[['age', 'workclass', 'fnlwgt', 'education.num', 'marital.status',
       'relationship', 'sex', 'hours.per.week', 
       'capital_profit','is_income_more_than_50K']]
# sns.pairplot(data=data, hue ='is_income_more_than_50K')
cleaned_data.head()
def split_data_to_k_folders(data, k=10):
    """Split data into k folders evenly and throw the remained data"""
    ##number of rows of data in every folder
    unit = len(data)//k
    data_folders = []
    for i in range(0,len(data), unit):
        data_folders.append(data.iloc[i:i+unit])
    return data_folders[:k]

def get_stratified_folder(d1, d2):
    combied_folder = []
    for i in range(len(d1)):
        temp = d1[i].append(d2[i])
        combied_folder.append(temp)
    return combied_folder

def split_data_train_test(test_num, combied_folder):
    temp_folder  = combied_folder.copy()
    index = test_num-1
    test_data = temp_folder[index].copy()
    del temp_folder[index]
    train_data = temp_folder[0].append(temp_folder[1:])
    return train_data, test_data

"""
Normalize all dataframe
"""
def get_norm_df(old_df, label):
    new_df= pd.DataFrame()
    for col in old_df.columns:
        if col!=label:
            new_df[col]=norm(old_df[col].values)
        else:
            new_df[col] = old_df[col].values
    return new_df


"""Normalize one of the col of the dataframe"""
def norm(col_values):
    col_values =(col_values-min(col_values))/max(col_values)
    return col_values

from random import *

def split_df_by_label(df):
    neg = df[df['is_income_more_than_50K']==-1]
    pos = df[df['is_income_more_than_50K']==1]
    return neg, pos


def train_test_split(f_list,test_i=0, ratio =0.8):

    def get_random_train_index(n,test_index, k):
        """n: the number of random number
             test_index:the random number should not be the t—index
             return type: [1,4,5,3,...]
        """
        all_i = np.arange(k).tolist()

        rand_list =[]
        while True:
            if len(rand_list)==n+1:
                break
            m=len(all_i)
            selected_i = randint(0,m-1)
            rand_list.append(all_i[selected_i])
            del all_i[selected_i]
        if test_index in rand_list:
            rand_list.remove(test_index)
            return rand_list

        return rand_list[1:]
    k = len(f_list)

    train_f_num= int(ratio*10)
    train_i_list = get_random_train_index(train_f_num, test_i,k)
    print(train_i_list)
    test_data = f_list[test_i]
    train_data=f_list[train_i_list[0]]
    for i in range(1,len(train_i_list)):
        train_data =train_data.append(f_list[train_i_list[i]])
    return test_data, train_data

def stratified_folders(df,k):
    """
    return the  folder split into k by ratio of -1 and +1
    
    """
    neg, pos = split_df_by_label(df)

    neg_folder_list= split_data_to_k_folders(neg, k=k)
    pos_folder_list = split_data_to_k_folders(pos,k=k)


    f_list = get_stratified_folder(neg_folder_list,pos_folder_list)
#     test_data,train_data=  train_test_split(f_list,test_i, ratio)

    print('There are {} folders and every folder has {} samples'.format(k, len(f_list[0])))
    return f_list




# import numpy as np
# def get_entropy(df, label):
#     """get the etropy of based on certain dataset(might be split by some node)"""
#     p_p_1 = (df[label]==1).sum()/len(df)
#     p_n_1= (df[label]==-1).sum()/len(df)
#     a = np.array([p_p_1,p_n_1])

#     b = np.log2([1 if p_p_1==0 else p_p_1, 1 if p_n_1==0 else p_n_1])
    
    

#     rs = -np.dot(a.T, b)
#     return -np.dot(a.T, b)
    
# def get_i_by_split(df,split_node_name):
#     n = len(df)
#     sub_partitions_list = df[split_node_name].unique()
#     prob_list = []
#     sub_entroies_list = []
#     for name in sub_partitions_list:
#         selected_rows = df[split_node_name]==name
#         prob_list.append(selected_rows.sum()/n)
        
#         sub_entroies_list.append(get_entropy(df[selected_rows], 'is_income_more_than_50K'))

#     prob_list=np.array(prob_list)
#     sub_entroies_list = np.array(sub_entroies_list)
#     rs =np.dot(prob_list.T, sub_entroies_list)

#     return rs
    
# def get_info_gain_by_split(node_name):
#     a = get_entropy(sample,'is_income_more_than_50K')
#     rs = a - get_i_by_split(sample, node_name) 
#     return rs if not np.isnan(rs) else 0
# # get_info_gain_by_split('age')

# sample = data.copy()
# def sort_node_by_infor_gain(node_list):
#     m = {}
#     for node in node_list:
#         m[node] = get_info_gain_by_split(node)
#     sorted_by_value = sorted(m.items(), key=lambda kv: -kv[1])
#     for item in sorted_by_value:
#         print('GAIN({}) = {} '.format(item[0],str(item[1])))
# a = list(col for col in sample.columns if col!='is_income_more_than_50K')
# sort_node_by_infor_gain(a)






from numpy import *

def loadDataSet(data):
    n =len(data[0])
    dataArr = data[:,:n-1].tolist()
    labelArr = data[:,-1].tolist()
    return dataArr,labelArr
    

def selectJrand(i,m): 
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):  
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

def kernelTrans(X, A, kTup): 
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': 
        K = X.dot(A)
    elif kTup[0]=='rbf': 
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) 
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K




class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  
        self.X = dataMatIn  
        self.labelMat = classLabels 
        self.C = C 
        self.tol = toler 
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0 
        self.eCache = mat(zeros((self.m,2))) 
        self.K = mat(zeros((self.m,self.m))) 
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)


def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]  
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE): 
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k): 
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i) 
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)): #检验这行数据是否符合KKT条件 参考《统计学习方法》p128公式7.111-113
        j,Ej = selectJ(i, oS, Ei) 
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]): 
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta 
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L) 
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < oS.tol): 
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#参考《统计学习方法》p127公式7.109
        updateEk(oS, i)
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]<oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j]<oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup):
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m): 
                alphaPairsChanged += innerL(i,oS)
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs: 
                alphaPairsChanged += innerL(i,oS)
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
#         print("      iteration number: %d" % iter)
    return oS.b,oS.alphas



def learning(data_train,C=200, tol=0.0001,max_iter=1000, kTup =('lin', 0)):
    dataArr,labelArr = loadDataSet(data_train) 
    b,alphas = smoP(dataArr, labelArr, C, tol, max_iter, kTup) 
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas)[0]  
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd] 
    return Classifier(alphas,b, sVs,labelSV,svInd,kTup)

class Classifier:
    def __init__(self,alphas,b,sVs,labelSV,svInd,kTup):
        self.alphas= alphas
        self.b= b
        self.sVs=sVs
        self.labelSV =labelSV
        self.svInd = svInd
        self.kTup=kTup

    def predict(self, test_data):
        test_X=test_data[:,:len(test_data[0])-1]
        test_y = test_data[:,-1]
#         print('tetex in predictmethod',test_X)
        """must be the np.array type for both x and y"""
        m,n = shape(test_X)
#         print('------', test_y)
        errorCount = 0
        predict_y= []
        for i in range(m):
            kernelEval = kernelTrans(self.sVs,test_X[i,:],self.kTup) 
            predict=kernelEval.T * multiply(self.labelSV,self.alphas[self.svInd]) + self.b
            predict_y.append(sign(predict))
        return PredictRes(predict_y=predict_y, real_y=test_y)
    
class PredictRes:
    def __init__(self, predict_y, real_y):
        predict_y=np.array(predict_y).reshape(1,-1)
        boolean_index = predict_y==real_y
        self.predict_y =predict_y
        self.n = self.predict_y[0].size
        self.real_y = real_y
        self.number_pos = (real_y==1).sum()
        self.number_neg = (real_y==-1).sum()
        self.TP = (predict_y[boolean_index]==1).sum()
        self.TN= (predict_y[boolean_index]==-1).sum()
        self.FP =(predict_y[~boolean_index]==1).sum()
        self.FN =(predict_y[~boolean_index]==-1).sum()
        self.precision = self.TP / (self.TP+self.FP) if (self.TP+self.FP)!=0 else 0
        self.recall = self.TP/(self.TP+self.FN) if (self.TP+self.FN)!=0 else 0
        self.total_accuracy = boolean_index.sum()/self.n
        self.f1_score = 2*self.recall*self.precision/(self.precision+self.recall) if (self.precision+self.recall)!=0 else 0
        
    def print_perfance_details(self):
        print('The test sample size is {} (cotain 70% +1, and 30% -1)'.format(self.n))
        print("Recall   \t", self.recall)
        print("Precision\t",  self.precision)
        print("F1 Score \t",  self.f1_score)
        print("Total Accuracy \t",  self.total_accuracy)

        
def sub_features(train, test, fl):
    return train[fl], test[fl]


def helper(features, train_data, test_data):
    sns.lmplot( x=features[0], y=features[1], data=train_data, fit_reg=False, hue='is_income_more_than_50K', legend=True)
    train, test = sub_features(train_data, test_data,features)

    return train, test


def c_v_test(folders):
    
    res= []
    for i in range(1,len(folders)+1):
        s ='Take Folder no.'+str(i)+' as test folder'
        print(s)
        train,test = split_data_train_test(i,part_f_list)
#         train, test = helper(['age','education.num','is_income_more_than_50K'], train_data, test_data)
        train=train.values
        test = test.values

        clf = learning(train,C=200, tol=0.0001,max_iter=20, kTup = ('rbf', 1))
        res_temp = clf.predict(test)
        res.append(res_temp)
    
    print('10 Cross Validation was ended')
    return res

        
norm_df  = get_norm_df(cleaned_data, 'is_income_more_than_50K')
f_list= stratified_folders(norm_df,1400)
part_f_list = f_list[10:20]
print('Testing size is {}'.format(len(part_f_list[0])))
print('Trainging size is {}'.format(len(part_f_list[0])*9))

res = c_v_test(part_f_list)

def cal_avg_and_var(pre_res_list):
    performan_detail = {'precisions':[],
    'recalls':[],
    'f1_scores':[],
    'total_accuracies':[]
    }
    for pre_res in pre_res_list:
        performan_detail['precisions'].append(pre_res.precision)
        performan_detail['recalls'].append(pre_res.recall)
        performan_detail['f1_scores'].append(pre_res.f1_score)
        performan_detail['total_accuracies'].append(pre_res.total_accuracy)
    def cal(d):
        avg_performan = {}
        for key in d:
            avg_performan[key] = 'The avg( {} )= {}, var( {} )={}.'.format(key,np.average(d[key]) , key, np.var(d[key]))

        return avg_performan
    avg_performan=cal(performan_detail)
    return performan_detail,avg_performan 
performan_detail,avg_performan = cal_avg_and_var(res)
avg_performan
class Bagging:
    def __init__(self,data_folders, test_data, n=15):
        self.data_folders =data_folders
        self.num_cls = n #n must be the odd number
        self.clfs =[]
        self.test_data=test_data
        self.random_train_data_folders =[]
    
    def bagging_data(self):
        #Get  10 folders in the all data folders as one trainning sample randomly
        random_train_data_folders =[]
        n=self.num_cls
        for i in range(n):
            select_no = np.random.randint(len(self.data_folders), size=n)
            temp = self.data_folders[select_no[0]]
            for i in range(1,len(select_no)):
                temp.append(self.data_folders[select_no[i]])
            random_train_data_folders.append(temp)
        self.random_train_data_folders = random_train_data_folders
        
    def build_clfs(self, C=200, tol=0.0001,max_iter=20, kTup = ('rbf', 1)):
        if len(self.random_train_data_folders)==0:
            print('There are no data for trainning, please bagging data first')
            return 
        clfs =[]
        i = 1
        print('Start Building All Classifiers')
        print(' Trainning Size : ', len(self.random_train_data_folders[0]))
        print(' Number of classifiers : ',self.num_cls)
        for sub_train in self.random_train_data_folders:
            print("\t-Trainning clf no.{}".format(i))
            i+=1
            clf = learning(sub_train.values,C, tol,max_iter, kTup)
            clfs.append(clf)
        print('All Classifiers were build successfully')
        self.clfs = clfs

    def predict_by_multi_clfs(self, test_data):
        all_pre_res = []
        real_y = None
        for clf in self.clfs:
            obj_res = clf.predict(test_data)
            all_pre_res.append(obj_res.predict_y)
            real_y = obj_res.real_y
        all_pre_res =np.array(all_pre_res)
        temp = np.sum(all_pre_res, 0)
        final_res= []
        for i in range(len(temp)):
            final_res.append(sign(temp[i]))
        predict_y = np.array(final_res)
        return PredictRes(predict_y=predict_y, real_y=real_y)
    
        
f_list= stratified_folders(norm_df,1000)
part_f_list = f_list[20:40]
test_data = f_list[1]
bagging = Bagging(part_f_list,test_data,n=11)
bagging.bagging_data()
bagging.build_clfs()
res = bagging.predict_by_multi_clfs(test_data.values)

res1 = bagging.predict_by_multi_clfs(f_list[25].values)

res1.print_perfance_details()
res.predict_y
from numpy import *

def loadDataSet(data): #读取数据
    n =len(data[0])
    dataArr = data[:,:n-1].tolist()
    labelArr = data[:,-1].tolist()
    return dataArr,labelArr
    

def selectJrand(i,m): #在0-m中随机选择一个不是i的整数
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):  #保证a在L和H范围内（L <= a <= H）
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

def kernelTrans(X, A, kTup): #核函数，输入参数,X:支持向量的特征树；A：某一行特征数据；kTup：('lin',k1)核函数的类型和参数
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': #线性函数
        K = X * A.T
    elif kTup[0]=='rbf': # 径向基函数(radial bias function)
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #返回生成的结果
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


#定义类，方便存储数据
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # 存储各类参数
        self.X = dataMatIn  #数据特征
        self.labelMat = classLabels #数据类别
        self.C = C #软间隔参数C，参数越大，非线性拟合能力越强
        self.tol = toler #停止阀值
        self.m = shape(dataMatIn)[0] #数据行数
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0 #初始设为0
        self.eCache = mat(zeros((self.m,2))) #缓存
        self.K = mat(zeros((self.m,self.m))) #核函数的计算结果
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)


def calcEk(oS, k): #计算Ek（参考《统计学习方法》p127公式7.105）
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

#随机选取aj，并返回其E值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]  #返回矩阵中的非零位置的行数
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE): #返回步长最大的aj
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k): #更新os数据
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

#首先检验ai是否满足KKT条件，如果不满足，随机选择aj进行优化，更新ai,aj,b值
def innerL(i, oS): #输入参数i和所有参数数据
    Ei = calcEk(oS, i) #计算E值
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)): #检验这行数据是否符合KKT条件 参考《统计学习方法》p128公式7.111-113
        j,Ej = selectJ(i, oS, Ei) #随机选取aj，并返回其E值
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]): #以下代码的公式参考《统计学习方法》p126
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
#             print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #参考《统计学习方法》p127公式7.107
        if eta >= 0:
#             print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta #参考《统计学习方法》p127公式7.106
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L) #参考《统计学习方法》p127公式7.108
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < oS.tol): #alpha变化大小阀值（自己设定）
#             print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#参考《统计学习方法》p127公式7.109
        updateEk(oS, i) #更新数据
        #以下求解b的过程，参考《统计学习方法》p129公式7.114-7.116
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]<oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j]<oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


#SMO函数，用于快速求解出alpha
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)): #输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（默认线性核）
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m): #遍历所有数据
                alphaPairsChanged += innerL(i,oS)
#                 print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)) #显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs: #遍历非边界的数据
                alphaPairsChanged += innerL(i,oS)
#                 print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("      iteration number: %d" % iter)
    return oS.b,oS.alphas

def testRbf(data_train,data_test,C=200, tol=0.0001,max_iter=1000, kTup = ('rbf', 1) ):
    dataArr,labelArr = loadDataSet(data_train) #读取训练数据
    b,alphas = smoP(dataArr, labelArr, C, tol, max_iter, kTup) #通过SMO算法得到b和alpha
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas)[0]  #选取不为0数据的行数（也就是支持向量）
    sVs=datMat[svInd] #支持向量的特征数据
    labelSV = labelMat[svInd] #支持向量的类别（1或-1）
#     print("there are %d Support Vectors" % shape(sVs)[0]) #打印出共有多少的支持向量
    m,n = shape(datMat) #训练数据的行列数
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup) #将支持向量转化为核函数
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b  #这一行的预测结果（代码来源于《统计学习方法》p133里面最后用于预测的公式）注意最后确定的分离平面只有那些支持向量决定。
        if sign(predict)!=sign(labelArr[i]): #sign函数 -1 if x < 0, 0 if x==0, 1 if x > 0
            errorCount += 1
#     print("the training error rate is: %f" % (float(errorCount)/m)) #打印出错误率
    train_err = float(errorCount)/m
    dataArr_test,labelArr_test = loadDataSet(data_test) #读取测试数据
    errorCount_test = 0
    datMat_test=mat(dataArr_test)
    labelMat = mat(labelArr_test).transpose()
    m,n = shape(datMat_test)
    for i in range(m): #在测试数据上检验错误率
        kernelEval = kernelTrans(sVs,datMat_test[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr_test[i]):
            errorCount_test += 1
#     print("the test error rate is: %f" % (float(errorCount_test)/m))
    test_err = float(errorCount_test)/m
    return {'train_err':train_err, 'test_err':test_err}


#主程序



def eval_perf(all_error):
    train_err = all_error[0]
    test_err = all_error[1]
    s1 ='avg(test_err) = {}, var(test_err) =  {}'.format(np.average(test_err), np.var(test_err))
    s2 ='avg(train_err) = {}, var(train_err) =  {}'.format(np.average(train_err), np.var(train_err))
    print(s1)
    print(s2)
# age	workclass	fnlwgt	education.num	marital.status	relationship	sex	hours.per.week	capital_profit	is_income_more_than_50K

def sub_features(train, test, fl):
    return train[fl], test[fl]


def helper(features, train_data, test_data):
    sns.lmplot( x=features[0], y=features[1], data=train_data, fit_reg=False, hue='is_income_more_than_50K', legend=True)
    train, test = sub_features(train_data, test_data,features)

    return train, test


        
norm_df  = get_norm_df(cleaned_data, 'is_income_more_than_50K')

f_list= stratified_folders(norm_df,600)
part_f_list = f_list[10:20]
print('test size is {}'.format(len(part_f_list[0])))
print('test size is {}'.format(len(part_f_list[0])*9))


def c_v_test(folders):
    all_train_error= []
    all_test_error= []
    for i in range(1,len(folders)+1):
        s ='Test folder no '+str(i)
        print(s)
        train,test = split_data_train_test(i,part_f_list)
#         train, test = helper(['age','education.num','is_income_more_than_50K'], train_data, test_data)
        train=train.values
        test = test.values
        error = testRbf(train,test,C=240, tol=0.0001,max_iter=30, kTup = ('rbf', 1))
        all_train_error.append(error['train_err'])
        all_test_error.append(error['test_err'])
    return all_train_error, all_test_error

all_error = c_v_test(part_f_list)

eval_perf(all_error)



# # test_data =test_data.sample(frac=1)
# # train_data = train_data.sample(frac=1)
# # sns.lmplot( x="age", y="education.num", data=test_data_age_and_fnlwgt, fit_reg=False, hue='is_income_more_than_50K', legend=True)




# def split_by_label(data):
#     """
#     dataset should be numpy format not dataframe
#     """
#     positive = data[data[:,-1]==1]
#     negative = data[data[:,-1]==-1]
#     return positive, negative

# positive, negative = split_by_label(train)


# X=train[:,:train[0].size-1]


# Y=train[:,-1]#Y
# C=40
# sigma=0.1
# SVMClassifier=SMO(X,Y,C,0.01,5,sigma)
# title = 'C={}, Sigma ={}'.format(C,sigma)
# SVMClassifier.visualize(positive,negative,colors=['green', 'blue','black'], xlabel=['age','education.num'],title=title)
# class CLASSIFIER:
#     def __init__(self, w,b):
#         self.w =w
#         self.b =b
#     def predit(self, test_X, test_y):
#         print('test size is ',len(test_y))
#         """Return the accuracy"""
#         temp =self.w.T*test_X
#         temp=np.sum(temp,axis=1)
#         res = temp+self.b
#         for i in range(len(res)):
#             if res[i]>0:
#                 res[i]=1
#             else:
#                 res[i]=-1
        
#         return ((res==test_y).sum())/len(test_y)




    
# w=  SVMClassifier.get_w()
# b=SVMClassifier.b
# classifier = CLASSIFIER(w,b)
# X=test[:,:test[0].size-1]
# y= test[:,-1]
# acc = classifier.predit(X, y)


