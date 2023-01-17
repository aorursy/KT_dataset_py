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
import torch 
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn import preprocessing
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
import matplotlib.pylab as plt1
data = pd.read_csv('../input/train.csv',dtype={'duration': int})
data_test = pd.read_csv('../input/test.csv',dtype={'duration': int})
subs = pd.read_csv('../input/sample_submission.csv',dtype={'duration': int})














data_csv1 = data
# data=data.drop(data.query('Target == 4').sample(frac=0.75).index)

# data = data[data['parentesco1']==1]
def datas_plit(data_train):
    # 把 个户ID 和家庭ID 标签放在 前面来
#     data_idhogar = data_train['idhogar']
#     data_train = data_train.drop('idhogar',axis=1)
#     data_train.insert(1,'idhogar',data_idhogar)
    
#     print(data_train['idhogar'],'=======!!!!+=================++++++++++++++++++++')
    
    indexs = np.array(data_train.columns.values)
#     print(indexs,'======================!!!!=====================','Target' in indexs)
    print('Target' in indexs,'========')
    flag = False
    if 'Target' in indexs:
        data_Target = data_train['Target']
        data_train = data_train.drop('Target',axis=1)
        data_train.insert(1,'Target',data_Target)
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
#         data_training = data_train.iloc[:,2:]
#         data_label = data_train.iloc[:,1]
        
        
        
#         return data_training,data_label
        data = data_train.iloc[:,1:]
        return data
    else:
        data_training = data_train.iloc[:,1:]
#         data_training = data_training.astype(np.float64)
        
        return data_training
    
data = datas_plit(data)

data_x = data.groupby('idhogar').mean().reset_index()


test_data = datas_plit(data_test)

data = data.drop('idhogar',axis=1)

data_y = data['Target'].values.reshape(-1,1)

print(data_y)
# 用 pytorch one-hot 标签

# def dense_to_one_hot(labels_dense, num_classes):
#     """Convert class labels from scalars to one-hot vectors."""
#     num_labels = labels_dense.shape[0]
#     index_offset = np.arange(num_labels) * num_classes
#     labels_one_hot = np.zeros((num_labels, num_classes))
#     labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
#     return labels_one_hot


# num_classes  = 5
# one_label = dense_to_one_hot(data_y,num_classes)

# data_y = one_label

# print(data_y)

# enc = preprocessing.OneHotEncoder()
# enc.fit(data_y)

# array = enc.transform(data_y).toarray()

# data_y = array
# print(data_y)


# yy = enc.inverse_transform(data_y)

# print(yy)

data = data.drop('Target',axis=1)
data_x = data
test_data = test_data.drop('idhogar',axis=1)

data_x = data_x.drop('cielorazo',axis=1)
test_data = test_data.drop('cielorazo',axis=1)




#把关联的列组合 删除
def comb(arr,data_csv):
    test = []
    for type in arr:
        test.append(data_csv[type])
        data_csv.drop([type], axis=1,inplace=True)
    array2 = np.array(test)
    max_p_row= array2.argmax(axis=0)
    return max_p_row


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

# data=data.drop(data.query('Target == 4').sample(frac=0.75).index)

plt.bar(range(len(num_list)),num_list,tick_label=name_list)
plt.show()


#把相关的特征进行处理
life_condition={'wall_material':['paredblolad','paredzocalo','paredpreb','pareddes','paredmad','paredzinc','paredfibras','paredother'],
#                 'elimbasu':['elimbasu1','elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6'],
                'floor':['pisomoscer','pisocemento','pisoother','pisonatur','pisonotiene','pisomadera'],
                'roof':['techozinc','techoentrepiso','techocane','techootro'],
                'water':['abastaguadentro','abastaguafuera','abastaguan1o'],
                'electricity':['public','planpri','noelec','coopele'],
                'sanitario':['sanitario1','sanitario2','sanitario3','sanitario5','sanitario6'],
                'energcocinar':['energcocinar1','energcocinar2','energcocinar3','energcocinar4'],
#                 'epared':['epared1','epared2','epared3'],
#                 'etecho':['etecho1','etecho2','etecho3'],
                'etecho':['etecho2','etecho3'],
                'eviv':['eviv1','eviv2','eviv3'],
#                 'sex':['male','female'],
                'estadocivil':['estadocivil1','estadocivil2','estadocivil3','estadocivil4','estadocivil5','estadocivil6','estadocivil7'],
                'parentesco':['parentesco1','parentesco2','parentesco3','parentesco4','parentesco5','parentesco6','parentesco7','parentesco8','parentesco9','parentesco10','parentesco11','parentesco12'],
                'instlevel':['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9'],
                'tipovivi':['tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5'],
#                 'lugar':['lugar2','lugar4','lugar6'],
#                 'lugar':['lugar1','lugar2','lugar3','lugar4','lugar5','lugar6'],
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
    
# model = xgb.XGBClassifier(max_depth=7, learning_rate=0.6, n_estimators=180,scale_pos_weight=5, silent=True, objective='multi:softmax')
# # model = KNeighborsClassifier(n_neighbors=2)
# # model = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
# model.fit(X_res, y_res)
# model = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8)
# model = GradientBoostingClassifier()

# model = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=7,max_features='sqrt',subsample=0.8)
# # model.fit(data_x, data_y)
# model.fit(X_res, y_res)

clfs = []
for i in range(24):
#     clf = GradientBoostingClassifier(learning_rate=0.1,random_state=100 + i, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8)
#     clf2 = xgb.XGBClassifier(max_depth=7, learning_rate=0.6, n_estimators=180,scale_pos_weight=5, silent=True, objective='multi:softmax')
       # 提升树的类型 gbdt,dart,goss,rf
    clf = lgb.LGBMClassifier(
                         objective='multiclass',
                         boosting_type='dart',
                         colsample_bytree=0.1, #训练特征采样率 列
                         learning_rate=0.02,
                         min_child_samples=19,
                         num_leaves=33,
                         subsample=0.552,
                         n_jobs=-1,
                         n_estimators=700,
                         silent=True,
                         scale_pos_weight=1,
                         metric='None',
                         random_state=300 + i,
                         class_weight='balanced'
    )
clfs.append(('lgbm{}'.format(i), clf))
# clfs.append(('clf1{}'.format(i), clf1))
# clfs.append(('clf2{}'.format(i), clf2))

model = VotingClassifier(clfs, voting='soft')

model.fit(data_x, data_y)
# model.fit(X_res, y_res)


# res_y = model.predict(X_res)
res_y = model.predict(data_x)

scores = accuracy_score(res_y, y_res)
# scores = accuracy_score(res_y, data_y)

print("Accuracy:  [%s]" % (scores))

ans = model.predict(test_data)




# y_test = model.predict( X_res )


# # 计算准确率
# # y_res = y_res.values
# cnt1 = 0
# cnt2 = 0
# for i in range(len(y_test)):
#     if y_res[i] == y_test[i]:
#         cnt1 += 1
#     else:
#         cnt2 += 1

# print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
# ans = model.predict(test_data)
ans = model.predict(test_data)

# yy = enc.inverse_transform(ans)

# print(yy)

subs['Target'] = ans
print(subs)

subs.to_csv('sample_submission.csv',index=False)


