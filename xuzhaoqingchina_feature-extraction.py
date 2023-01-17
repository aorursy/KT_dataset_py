import pandas as pd
import numpy as np
train = pd.read_csv('../input/train.csv')
train.describe()
train.apply(lambda x: len(set(x))) #共有多少种类别
#频次统计
from collections import defaultdict
def get_counts2(sequence):
    counts = defaultdict(int) 
    for x in sequence:
        counts[x] += 1
    return counts
import time
st = train['context_timestamp'].apply(time.localtime)
train['time'] = st.apply(lambda x : time.strftime('%Y-%m-%d %H:%M:%S',x))
train.drop('context_timestamp',axis = 1,inplace = True)
category = train['item_category_list'].str.split(';',expand = True)
category.apply(np.count_nonzero) #第三级只有2029个数据是有第三类的
category.apply(lambda x:len(set(x))) #category里有三类的分布，可看出0这一列可直接舍弃
category.drop(0,axis=1,inplace=True)
print('C2','\n',category.apply(get_counts2)[1]) #第二种类频次统计
print('C3','\n',category.apply(get_counts2)[2]) #第三种类频次统计
category = category.replace(['4879721024980945592','2436715285093487584','8710739180200009128','2011981573061447208','22731265849056483','3203673979138763595','2642175453151805566','1968056100269760729'],'1')
train[['C1','C2','C3','C4','C5','C6']] = pd.get_dummies(category[1])
occupation = train['user_occupation_id']
get_counts2(occupation)
occupation = occupation.replace(-1,2005)
train[['O1','O2','O3','O4']] = pd.get_dummies(occupation)
train
gender = train['user_gender_id']
get_counts2(gender)
properties = train['item_property_list'].str.split(';',expand = True) #不会处理，property有100个特征

sum_properties = []
for i in range(0,100):
    sum_properties.append(list(set(properties[i])))

list(set(sum(sum_properties,[]))) #全部分类
properties.reshape()



cities = train['item_city_id']
cities




age = train['user_age_level']
get_counts2(age)
time = train['time']
time


#from imblearn.over_sampling import SMOTE
#sm = SMOTE(random_state=42)
#X_res, y_res = sm.fit_sample(X, y)











