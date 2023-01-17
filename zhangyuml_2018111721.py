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
import re
from pandas.core.frame import DataFrame
from gensim.models.word2vec import Word2Vec
import nltk.data
nltk.download('stopwords')
from nltk.corpus import stopwords
import datetime

from xgboost.sklearn import XGBClassifier
import gc

stopw=[]
for w in stopwords.words('english'):
    stopw.append(w)
def process(address,scri): #数据处理
    pf=[]
    tf = pd.read_csv(address)
    count=0
    for line in tf[scri]:
        line1= re.sub(r'[^a-zA-Z]',' ' ,line)  #去标点符号
        line2 = line1.lower().split()           #小写化，分割单词
        line3= [w for w in line2 if w not in stopw]
        pf.append(line3)
        count+=1
        if count%10000==0: print("+",end ='')
    return pf

print("处理训练数据......")
pf=process("../input/train.csv","question_text")
print("处理测试数据......")
tf=process("../input/test.csv","question_text")

del stopw
gc.collect()
print("读取标签......")
train_lable = pd.read_csv("../input/train.csv")["target"].head(300000)
test_qid = pd.read_csv("../input/test.csv")["qid"]
df=DataFrame({"question_test":pf})
all_df=DataFrame({"question_test":pf+tf})
tesdata=DataFrame({"question_test":tf})
del pf 
gc.collect()
del tf
gc.collect()
print("训练稠密矩阵.....")
num_features = 100    # Word vector dimensionality
min_word_count = 10   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words
Vec = Word2Vec(all_df["question_test"],workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
def to_review_vector(review):
    array = np.array([Vec[w] for w in review if w in Vec])
    return pd.Series(array.mean(axis=0))
print("生成训练集X......")
train_data_feature=pd.DataFrame()
for i in range(4):
    train_data_feature1=df.question_test[300000*i:(i+1)*300000].apply(to_review_vector)
    print("finish{}",i)
    train_data_feature=pd.concat([train_data_feature,train_data_feature1])
train_data_feature1=df.question_test[1200000:].apply(to_review_vector)
train_data_feature=pd.concat([train_data_feature,train_data_feature1])
print("finish X......")

print("生成测试集x......")
test_data_feature=tesdata.question_test.apply(to_review_vector)
print("finish x......")
del df
gc.collect()
del tesdata
gc.collect()
model_xgb = XGBClassifier()
print("分类模型训练中......")
for i in range(4):
    model_xgb.fit(train_data_feature[300000*i:300000*(i+1)],train_lable[300000*i:300000*(i+1)])
model_xgb.fit(train_data_feature[1200000:],train_lable[1200000:])
print("分类模型训练完毕！")


result=model_xgb.predict(test_data_feature)

submit_df = pd.DataFrame({"qid": test_qid, "prediction": result})
submit_df.to_csv("submission.csv", index=False)


print("结束")






