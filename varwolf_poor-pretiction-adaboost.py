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
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
data = pd.read_csv('../input/train.csv',dtype={'duration': int})
data_test = pd.read_csv('../input/test.csv',dtype={'duration': int})
subs = pd.read_csv('../input/sample_submission.csv',dtype={'duration': int})


data = data[data['parentesco1'] == 1]
def datas_plit(data_train):
    # 把 个户ID 和家庭ID 标签放在 前面来
    data_idhogar = data_train['idhogar']
    data_train = data_train.drop('idhogar',axis=1)
    data_train.insert(1,'idhogar',data_idhogar)
    
#     print(data_train['idhogar'],'=======!!!!+=================++++++++++++++++++++')
    
    indexs = np.array(data_train.columns.values)
#     print(indexs,'======================!!!!=====================','Target' in indexs)
    print('Target' in indexs,'========')
    flag = False
    if 'Target' in indexs:
        data_Target = data_train['Target']
        data_train = data_train.drop('Target',axis=1)
        data_train.insert(2,'Target',data_Target)
        flag = True
        print('Target in index ')
    else:
        print('Target NOT in index !!!!!')

    # # 在填充  均值
    data_train = data_train.fillna(0)
    # 依赖注入 关系  转变 0 1
    data_train.loc[data_train['dependency'] == 'yes', 'dependency'] = 1
    data_train.loc[data_train['dependency'] == 'no', 'dependency'] = 0
    # 起誓  0 1
    data_train.loc[data_train['edjefe'] == 'yes', 'edjefe'] = 1
    data_train.loc[data_train['edjefe'] == 'no', 'edjefe'] = 0

    data_train.loc[data_train['edjefa'] == 'yes', 'edjefa'] = 1
    data_train.loc[data_train['edjefa'] == 'no', 'edjefa'] = 0


    data_train['edjefa'] = data_train['edjefa'].astype(np.float64)
    data_train['edjefe'] = data_train['edjefe'].astype(np.float64)
    data_train['dependency'] = data_train['dependency'].astype(np.float64)
    

    if flag:
        data_training = data_train.iloc[:,3:]
        data_label = data_train.iloc[:,2]
        
        return data_training,data_label
    else:
        data_training = data_train.iloc[:,2:]
        data_training = data_training.astype(np.float64)
        
        return data_training
data_x,data_y = datas_plit(data)

test_data = datas_plit(data_test)


#把关联的列组合 删除
def comb(arr,data_csv):
    test = []
    for type in arr:
        test.append(data_csv[type])
        data_csv.drop([type], axis=1,inplace=True)
    array2 = np.array(test)
    max_p_row= array2.argmax(axis=0)
    return max_p_row

data_csv1 = data

#租金为空值的人的房屋情况
norent=data_csv1[data_csv1['v2a1'].isnull()]
print(norent.shape)
own_house=norent[norent['tipovivi1']==1]['Id'].count()
installments=norent[norent['tipovivi2']==1]['Id'].count()
rented=norent[norent['tipovivi3']==1]['Id'].count()
Precarious=norent[norent['tipovivi4']==1]['Id'].count()
other= norent[norent['tipovivi5']==1]['Id'].count()
num_list=[own_house,installments,rented,Precarious,other]
name_list=['own_house','installments','rented','Precarious','other']

# data_csv=data_csv.drop(data_csv.query('Target == 4').sample(frac=0.75).index)

plt.bar(range(len(num_list)),num_list,tick_label=name_list)
plt.show()


#把相关的特征进行处理
life_condition={'wall_material':['paredblolad','paredzocalo','paredpreb','pareddes','paredmad','paredzinc','paredfibras','paredother'],
                'elimbasu':['elimbasu1','elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6'],
                'floor':['pisomoscer','pisocemento','pisoother','pisonatur','pisonotiene','pisomadera'],
                'roof':['techozinc','techoentrepiso','techocane','techootro'],
                'water':['abastaguadentro','abastaguafuera','abastaguano'],
                'electricity':['public','planpri','noelec','coopele'],
                'sanitario':['sanitario1','sanitario2','sanitario3','sanitario5','sanitario6'],
                'energcocinar':['energcocinar1','energcocinar2','energcocinar3','energcocinar4'],
                'epared':['epared1','epared2','epared3'],
                 'etecho':['etecho1','etecho2','etecho3'],
                'eviv':['eviv1','eviv2','eviv3'],
                'sex':['male','female'],
                'estadocivil':['estadocivil1','estadocivil2','estadocivil3','estadocivil4','estadocivil5','estadocivil6','estadocivil7'],
                'parentesco':['parentesco1','parentesco2','parentesco3','parentesco4','parentesco5','parentesco6','parentesco7','parentesco8','parentesco9','parentesco10','parentesco11','parentesco12'],
                'instlevel':['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9'],
                'tipovivi':['tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5'],
                'lugar':['lugar1','lugar2','lugar3','lugar4','lugar5','lugar6'],
                'area':['area1',"area2"],
}
    

for life in (life_condition):
    _type = comb(life_condition[life], data_x) #
    data_x.insert(2,life, _type)


    
for life in (life_condition):
    _type = comb(life_condition[life], test_data) #
    test_data.insert(2,life, _type)
    
    
kind = ['regular', 'borderline1', 'borderline2', 'svm']
sm = SMOTE(kind='svm') 
X_res, y_res = sm.fit_sample(data_x, data_y)

columns = test_data.columns.values
X_res = pd.DataFrame(X_res,columns=columns)

print(X_res)
print(test_data)
#设定弱分类器CART
weakClassifier=DecisionTreeClassifier(max_depth=1)

#clf = AdaBoostClassifier(n_estimators=100,learning_rate=0.6,algorithm='SAMME.R')
clf = AdaBoostClassifier(n_estimators=500,learning_rate=0.2,algorithm='SAMME.R')
# clf.fit(data_x,data_y)
clf.fit(X_res,y_res)
test_predict = clf.predict(data_x)

score = accuracy_score(test_predict, data_y)

print("The accruacy score is %f" % score)
ans = clf.predict(test_data)

subs['Target'] = ans
print(subs)

subs.to_csv('sample_submission.csv',index=False)
