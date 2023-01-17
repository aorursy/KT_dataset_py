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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
import warnings
from numpy import hstack,vstack,array,nan

from sklearn.feature_selection import SelectKBest,mutual_info_classif
from minepy import MINE

def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

x_columns = [x for x in train.columns if x not in ['Id',
            'Soil_Type5'
             ,'Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type14'
             ,'Soil_Type15','Soil_Type16','Soil_Type18','Soil_Type19','Soil_Type21'
            ,'Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28'
             ,'Soil_Type34','Soil_Type36','Soil_Type37'
             ,'Cover_Type']]
x1_columns = [x for x in test.columns if x not in ['Id',
            'Soil_Type5'
             ,'Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type14'
             ,'Soil_Type15','Soil_Type16','Soil_Type18','Soil_Type19','Soil_Type21'
            ,'Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28'
             ,'Soil_Type34','Soil_Type36','Soil_Type37'
             ]]

X1 = test[x1_columns]
X = train[x_columns]
Y = train['Cover_Type']

X = X.values
Y = Y.values
X1 = X1.values

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=10)

ss=MaxAbsScaler()

X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
X1=ss.transform(X1)
X_test = X1

new_feature=[] #装每个模型产生的新特征列
new_label_test=[]
#模型模块
clf1=LogisticRegression()
clf2=KNeighborsClassifier(n_neighbors=3,weights='distance')
clf3=GaussianNB()
clf4=RandomForestClassifier(n_estimators= 182)

set_Train=[]
set_Test=[]
for k in range(5):
    span=int(X_train.shape[0]/5)
    i=k*span
    j=(k+1)*span
    #将X_train和y_train均划分成为5分被后续交叉验证使用
    set_Train.append(X_train[i:j])
    set_Test.append(y_train[i:j])

model=[clf1,clf2,clf3,clf4]
f = []
for index,clf in enumerate(model):
    model_list=[] #将每一轮交叉验证的预测label存入其中,再转为array做转置.
    label_list=[] #总的测试集对应的预测标签
    #k折交叉验证
    for k in range(5):
        ##选择做交叉验证的测试集
        XXtest=[]
        XXtest.append(set_Train[k])
        #选择做交叉验证训练的训练集
        XXtrain=[]
        YYtest=[]
        for kk in range(5):
            if kk==k:
                continue
            else:
                XXtrain.append(set_Train[kk])
                YYtest.append(set_Test[kk])
        #模型的训练
        XXXtrain=array(vstack((XXtrain[0],XXtrain[1],XXtrain[2],XXtrain[3]))) 
        YYYtrain=array(hstack((YYtest[0],YYtest[1],YYtest[2],YYtest[3])))
        XXXtest=array(vstack(XXtest)) #XXtest.shape=1*24*4,不是想要的96x4
        clf.fit(XXXtrain,YYYtrain)
        y_predict=clf.predict(XXXtest)
        model_list.append(y_predict) #将第k折验证中第k折为测试集的预测标签存储起来
        test_y_pred=clf.predict(X_test)
        label_list.append(test_y_pred)
    new_k_feature=array(hstack((model_list[0],model_list[1],model_list[2],model_list[3],model_list[4]))) #hstack() takes 1 positional argument,所以参数使用一个tuple来封装
    print ('new_k_feature',new_k_feature.shape)
    new_feature.append(new_k_feature)
    print (len(new_feature[index]))
    new_k_test=array(vstack((label_list[0],label_list[1],label_list[2],label_list[3],label_list[4]))).T #hstack() takes 1 positional argument,所以参数使用一个tuple来封装
    print ('new_k_test',new_k_test.shape)
    new_label_test.append(array(list(map(int,list(map(round,new_k_test.mean(axis=1)))))))
    print (len(new_label_test[index]))
#将5个基模型训练好的预分类标签组合成为新的特征供后续使用(X_train')
newfeature_from_train=array(vstack((new_feature[0],new_feature[1],new_feature[2],new_feature[3]))).T #拼接完成后做转置
print ('newfeature_from_train',newfeature_from_train.shape)
#将交叉验证获得的label拼接起来(X_test')
predict_from_test_average=array(vstack((new_label_test[0],new_label_test[1],new_label_test[2],new_label_test[3]))).T

clf5=GradientBoostingClassifier()
clf5.fit(newfeature_from_train,y_train)
predict1=clf5.predict(predict_from_test_average)

print ('predict over')
n = [x for x in range(15121,581013)]
predDf=pd.DataFrame(
    {'Id':n,
    'Cover_Type':predict1})

predDf.to_csv('mypred6.csv',index=False)
print ('over:')

