import warnings

warnings.filterwarnings('ignore')

!pip install jieba

!wget -nc "https://codeload.github.com/weiyunchen/nlp/zip/master"

!unzip -o master

import pandas as pd

import os

import time

from tqdm import tqdm_notebook

import re

import time

import copy

import random

import jieba

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.optim import lr_scheduler

data = pd.read_csv('../input/DMSC.csv', index_col=0)



# 按评分分成两类，1分2分为负面评价,345正面

data['Star']=((data.Star+0.5)/3.5+1).astype(int)
Movie = data['Movie_Name_CN'].value_counts()

Movie
sample_df = data.groupby(['Movie_Name_CN', 'Star']).apply(

    lambda x: x.sample(n=int(2125056/(28*200)), replace=True, random_state=0))



sample_df.shape
from sklearn.model_selection import train_test_split





comments = sample_df.values[:, 7]

star = sample_df.values[:, 6]



x_train, x_test, y_train, y_test, = train_test_split(

    comments, star, test_size=0.2, random_state=0)



len(y_train), len(y_test), len(x_train), len(x_test)
# 清理非中文字符

def clean_str(line):

    line.strip('\n')

    line = re.sub(r"[^\u4e00-\u9fff]", "", line)

    line = re.sub(

        "[0-9a-zA-Z\-\s+\.\!\/_,$%^*\(\)\+(+\"\')]+|[+——！，。？、~@#￥%……&*（）<>\[\]:：★◆【】《》;；=?？]+", "", line)

    return line.strip()





# 加载停用词

with open('nlp-master/stopwords.txt') as f:

    stopwords = [line.strip('\n') for line in f.readlines()]





def cut(data, labels, stopwords):

    result = []

    new_labels = []

    for index in tqdm_notebook(range(len(data))):

        comment = clean_str(data[index])

        label = labels[index]

        # 分词

        seg_list = jieba.cut(comment, cut_all=False, HMM=True)

        seg_list = [x.strip('\n')

                    for x in seg_list if x not in stopwords and len(x) > 1]

        if len(seg_list) > 1:

            result.append(seg_list)

            new_labels.append(label)

    # 返回分词结果和对应的标签

    return result, new_labels



# 分别对训练数据和测试数据分词

train_cut_result, train_labels = cut(x_train, y_train, stopwords)

test_cut_result, test_labels = cut(x_test, y_test, stopwords)
# TfidfVectorizer 传入原始文本

train_data = [' '.join(x) for x in train_cut_result]

test_data = [' '.join(x) for x in test_cut_result]



n_dim = 20000



# 数据的TF-IDF信息计算

# sublinear_tf=True 时生成一个近似高斯分布的特征，可以提高大概1~2个百分点

vectorizer = TfidfVectorizer(

    max_features=n_dim, smooth_idf=True, sublinear_tf=True)



# 对训练数据训练

train_vec_data = vectorizer.fit_transform(train_data)



# 训练完成之后对测试数据转换

test_vec_data = vectorizer.transform(test_data)
vectorizer.get_feature_names()[:10]
# 输出的类别为 2

n_categories = 2

# 学习率过大会导致 loss 震荡

learning_rate = 0.001

# 损失函数

criterion = nn.CrossEntropyLoss()

# 迭代次数

epochs = 3

# 每次迭代同时加载的个数

batch_size = 100
class TxtDataset(Dataset):

    def __init__(self, VectData, labels):

        # 传入初始数据，特征向量和标签

        self.VectData = VectData

        self.labels = labels



    def __getitem__(self, index):

        # DataLoader 会根据 index 获取数据

        # toarray() 是因为 VectData 是一个稀疏矩阵，如果直接使用 VectData.toarray() 占用内存太大，勿尝试

        return self.VectData[index].toarray(), self.labels[index]-1



    def __len__(self):

        return len(self.labels)



# 线下内存足够大可以考虑增大 num_workers，并行读取数据

# 加载训练数据集

train_dataset = TxtDataset(train_vec_data, train_labels)

train_dataloader = DataLoader(train_dataset,

                              batch_size=batch_size,

                              shuffle=True,

                              num_workers=1

                              )

# 加载测试数据集

test_dataset = TxtDataset(test_vec_data, test_labels)

test_dataloader = DataLoader(test_dataset,

                             batch_size=batch_size,

                             shuffle=False,

                             num_workers=1

                             )
class TxtModel(nn.Module):

    def __init__(self, input_size, output_size):

        super(TxtModel, self).__init__()

        self.classifier = nn.Sequential(

            nn.Linear(input_size, 1024),

            nn.ReLU(inplace=True),

            nn.Dropout(),

            nn.Linear(1024, 1024),

            nn.ReLU(inplace=True),

            nn.Dropout(),

            nn.Linear(1024, 1024),

            nn.ReLU(inplace=True),

            nn.Dropout(),

            nn.Linear(1024, 512),

            nn.ReLU(inplace=True),

            nn.Dropout(),

            nn.Linear(512, output_size)

        )



    def forward(self, x):

        output = self.classifier(x.double())

        return output.squeeze(1)
# 定义模型和优化器

model = TxtModel(n_dim, 2)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# 每两代衰减学习率

exp_lr_scheduler = lr_scheduler.StepLR(

    optimizer, step_size=int(epochs/2), gamma=0.1)

 

model = model.double()



# 保存准确度最高的模型

best_model = copy.deepcopy(model)

best_accuracy = 0.0



for epoch in range(epochs):

    exp_lr_scheduler.step()

    model.train()

    loss_total = 0

    st = time.time()

    # train_dataloader 加载数据集

    for data, label in tqdm_notebook(train_dataloader):

        output = model(data)

        # 计算损失

        loss = criterion(output, label)

        optimizer.zero_grad()

        # 反向传播

        loss.backward()

        optimizer.step()

        loss_total += loss.item()



    # 输出损失、训练时间等

    print('epoch {}/{}:'.format(epoch, epochs))

    print('training loss: {}, time resumed {}s'.format(

        loss_total/len(train_dataset), time.time()-st))



    model.eval()



    loss_total = 0

    st = time.time()



    correct = 0

    for data, label in test_dataloader:

        output = model(data)

        loss = criterion(output, label)

        loss_total += loss.item()



        _, predicted = torch.max(output.data, 1)

        correct += (predicted == label).sum().item()

    # 如果准确度取得最高，则保存准确度最高的模型

    if correct/len(test_dataset) > best_accuracy:

        best_model = copy.deepcopy(model)



    print('testing loss: {}, time resumed {}s, accuracy: {}'.format(

        loss_total/len(test_dataset), time.time()-st, correct/len(test_dataset)))
import json

import requests

# 26266893 为国产科幻佳作《流浪地球》，在此以《流浪地球》的影评为例

res = requests.get(

    'https://api.douban.com/v2/movie/subject/26266893/comments?apikey=0df993c66c0c636e29ecbb5344252a4a')

comments = json.loads(res.content.decode('utf-8'))['comments']

comments
def predict_comments(comments):

    test_comment = random.choice(comments)

# 选择其中一条分类，并去除非中文字符

    content = clean_str(test_comment['content'])

    rating = test_comment['rating']['value']

# 对评论分词

    seg_list = jieba.cut(content, cut_all=False, HMM=True)

# 去掉停用词和无意义的

    cut_content = ' '.join([x.strip('\n')

                        for x in seg_list if x not in stopwords and len(x) > 1])



# 转化为特征向量

    one_test_data = vectorizer.transform([cut_content])



# 转化为 pytorch 输入的 Tensor 数据，squeeze(0) 增加一个 batch 维度

    one_test_data = torch.from_numpy(one_test_data.toarray()).unsqueeze(0)

# 使用准确度最好的模型预测，softmax 处理输出概率，取得最大概率的下标再加 1 则为预测的标签

    pred = torch.argmax(F.softmax(best_model(one_test_data), dim=1)) + 1

    if rating<3:

        rat='差评1'

    else:

        rat='好评2'

    print('评论内容: ',content)

    print('关键字: ',cut_content)

    print('观众评价: ',rat)

    print('预测评价: ',pred)
for i in range(5):

    print('观后感: ',i)

    print(predict_comments(comments))