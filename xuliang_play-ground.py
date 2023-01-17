# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#导入训练集

monster = pd.read_csv("../input/train.csv")



#输出统计信息

print (monster.describe())

print ("-------")

print (monster.info())

print ("-------")

print (monster.head())
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction import DictVectorizer 



enc = OneHotEncoder()

v = DictVectorizer(sparse=False)



one_hot = pd.get_dummies(monster['color'])



monster = monster.drop("color", axis=1)



monster = monster.join(one_hot)



monster.head()





# The whole code cell 



#02 数据

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#导入训练集

monster_test = pd.read_csv("../input/train.csv")





#03 数值化

# color 特征数值化

def color_code(df):

    #提取所有存在的颜色，组成一个列表

    monster_color = df["color"].unique()

    

    # 循环将color都替换为其在颜色列表中的索引（整数）

    for i_color, m_color in enumerate(monster_color):

        df.loc[df["color"] == m_color, "color"] = i_color

        





#04 特征工程

# 导入SelectKBest, f_classif模块

from sklearn.feature_selection import SelectKBest, f_classif

        

# 05 机器学习/交叉验证

from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier



        

# 特征数值化，注意测试集不含'type'列，所以只对'color'执行即可

color_code(monster_test)



# 训练分类器

clf.fit(monster[predictors], monster["type"])



# 对测试集做预测

predictions = clf.predict(monster_test[predictors])



# 生成包含结果的DataFrame对象

submission = pd.DataFrame({

    "id": monster_test["id"],

    "type": predictions

})



# 在计算过程中我们将'type'做了数值化，而比赛的评分系统只认原始的字符串，所以需要反向还原为字符

def type_decode(df):

    df.loc[df["type"] == 0, "type"] = "Ghoul"

    df.loc[df["type"] == 1, "type"] = "Goblin"

    df.loc[df["type"] == 2, "type"] = "Ghost"



type_decode(submission)

submission.to_csv("result.csv", index=False)



f = open("result.csv")

print(f.read())
