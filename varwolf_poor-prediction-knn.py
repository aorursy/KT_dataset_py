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
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
data = pd.read_csv('../input/train.csv',dtype={'duration': int})
data_test = pd.read_csv('../input/test.csv',dtype={'duration': int})
subs = pd.read_csv('../input/sample_submission.csv',dtype={'duration': int})
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
models = []
models.append(("KNN",KNeighborsClassifier(n_neighbors=2)))  # 普通的k-近邻算法
models.append(("KNN with weights",KNeighborsClassifier(n_neighbors=2,weights='distance')))  # 带权值的K-近邻算法
models.append(("Radius Neighbors",RadiusNeighborsClassifier(n_neighbors=2,radius=500.0)))  # RadiusNeighborsClassifier的半径

#分别训练3个模型，并计算评分
results = []
for name,model in models:
    model.fit(data_x,data_y)
    results.append((name,model.score(data_x,data_y)))  # 评估模型

for i in range(len(results)):
    print('name: {}; score: {}'.format(results[i][0],results[i][1]))


# 带权值的K-近邻算法
KNN_W_model = KNeighborsClassifier(n_neighbors=3)
KNN_W_model.fit(data_x,data_y)

data_ytest = KNN_W_model.predict(test_data)
print(data_ytest)

subs['Target'] = data_ytest
print(subs)

subs.to_csv('sample_submission.csv',index=False)
