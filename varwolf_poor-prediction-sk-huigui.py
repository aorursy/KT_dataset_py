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
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
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
# 调用 XGBoost 回归 ,训练模型
model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='reg:gamma')
model.fit(data_x, data_y)
data_ans = model.predict(data_x)

# 计算准确率
y_test = data_y.values
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if int(data_ans[i]) == int(y_test[i]):
        cnt1 += 1
    else:
        cnt2 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

# 对测试集进行预测
ans = model.predict(test_data)

# 上传的文件 
subs['Target'] = ans.astype(np.uint)
print(subs)

subs.to_csv('sample_submission.csv',index=False)

plot_importance(model)
plt.show()
