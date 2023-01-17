# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
import torch.nn as nn  #神经网络工具箱torch.nn 
import torch.nn.functional as F  #神经网络函数torch.nn.functional
import torch.utils.data as tud  #Pytorch读取训练集需要用到torch.utils.data类
from torch.nn.parameter import Parameter  #参数更新和优化函数

from collections import Counter #Counter 计数器
import numpy as np 
import random
import math 

import pandas as pd
import scipy #SciPy是基于NumPy开发的高级模块，它提供了许多数学算法和函数的实现
import sklearn
from sklearn.metrics.pairwise import cosine_similarity #余弦相似度函数l
torch.cuda.is_available()
USE_CUDA = torch.cuda.is_available() #有GPU可以用

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)
    
# 设定一些超参数   
K = 100 # number of negative samples 负样本随机采样数量
C = 3 # nearby words threshold 指定周围三个单词进行预测
NUM_EPOCHS = 2 # The number of epochs of training 迭代轮数
MAX_VOCAB_SIZE = 30000 # the vocabulary size 词汇表多大
BATCH_SIZE = 128 # the batch size 每轮迭代1个batch的数量
LEARNING_RATE = 0.2 # the initial learning rate #学习率
EMBEDDING_SIZE = 100 #词向量维度
       
    
LOG_FILE = "word-embedding.log"

# tokenize函数，把一篇文本转化成一个个单词
def word_tokenize(text): 
    return text.split()

with open("../input/word2vec/text8.train.txt", "r") as fin: #读入文件
    text = fin.read()
    
text = [w for w in word_tokenize(text.lower())] 
#分词，在这里类似于text.split()

vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
#字典格式，把（MAX_VOCAB_SIZE-1）个最频繁出现的单词取出来，-1是留给不常见的单词

vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
#unk表示不常见单词数=总单词数-常见单词数
#这里计算的到vocab["<unk>"]=29999

idx_to_word = [word for word in vocab.keys()] 
#取出字典的所有单词key

word_to_idx = {word:i for i, word in enumerate(idx_to_word)}
#取出所有单词的单词和对应的索引，索引值与单词出现次数相反，最常见单词索引为0。

word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
#所有单词的频数values

word_freqs = word_counts / np.sum(word_counts)
#所有单词的频率

word_freqs = word_freqs ** (3./4.)
#论文里乘以3/4次方

word_freqs = word_freqs / np.sum(word_freqs) # 用来做 negative sampling
# 重新计算所有单词的频率

VOCAB_SIZE = len(idx_to_word) #词汇表单词数30000=MAX_VOCAB_SIZE
VOCAB_SIZE
class WordEmbeddingDataset(tud.Dataset): #tud.Dataset父类
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        ''' text: a list of words, all text from the training dataset
            word_to_idx: the dictionary from word to idx
            idx_to_word: idx to word mapping
            word_freq: the frequency of each word
            word_counts: the word counts
        '''
        super(WordEmbeddingDataset, self).__init__() #初始化模型
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE-1) for t in text]
        #字典 get() 函数返回指定键的值（第一个参数），如果值不在字典中返回默认值（第二个参数）。
        #取出text里每个单词word_to_idx字典里对应的索引,不在字典里返回"<unk>"的索引
        #"<unk>"的索引=29999，get括号里第二个参数应该写word_to_idx["<unk>"]，不应该写VOCAB_SIZE-1，虽然数值一样。
        
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        #变成tensor类型，这里变成longtensor，也可以torch.LongTensor(self.text_encoded)
        
        self.word_to_idx = word_to_idx #保存数据
        self.idx_to_word = idx_to_word  #保存数据
        self.word_freqs = torch.Tensor(word_freqs) #保存数据
        self.word_counts = torch.Tensor(word_counts) #保存数据
        
    def __len__(self): #数据集有多少个item 
        #魔法函数__len__
        ''' 返回整个数据集（所有单词）的长度
        '''
        return len(self.text_encoded) #所有单词的总数
        
    def __getitem__(self, idx):
        #魔法函数__getitem__，这个函数跟普通函数不一样
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
        '''
        center_word = self.text_encoded[idx] 
        #print(center_word)
        #中心词索引
        #这里__getitem__函数是个迭代器，idx代表了所有的单词索引。
        
        pos_indices = list(range(idx-C, idx)) + list(range(idx+1, idx+C+1))
        #周围词索引的索引，比如idx=0时。pos_indices = [-3, -2, -1, 1, 2, 3] 
        #老师讲这里的时候，我不是特别明白
        
        pos_indices = [i%len(self.text_encoded) for i in pos_indices]
        #range(idx+1, idx+C+1)超出词汇总数时，需要特别处理，取余数
        
        pos_words = self.text_encoded[pos_indices]
        #周围词索引，就是希望出现的正例单词
        #print(pos_words)
        
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        #负例采样单词索引，torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标。
        #取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大。
        #每个正确的单词采样K个，pos_words.shape[0]是正确单词数量
        #print(neg_words)
        
        return center_word, pos_words, neg_words 
dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
# list(dataset) 可以把尝试打印下center_word, pos_words, neg_words看看

dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)    
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        ''' 初始化输出和输出embedding
        '''
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size  #30000
        self.embed_size = embed_size  #100
        
        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        #模型输出nn.Embedding(30000, 100)
        self.out_embed.weight.data.uniform_(-initrange, initrange)
        #权重初始化的一种方法
        
        
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
         #模型输入nn.Embedding(30000, 100)
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        #权重初始化的一种方法
        
        
    def forward(self, input_labels, pos_labels, neg_labels):
        '''
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]
        
        return: loss, [batch_size]
        '''
        
        batch_size = input_labels.size(0)  #input_labels是输入的标签，tud.DataLoader()返回的。相已经被分成batch了。
        
        input_embedding = self.in_embed(input_labels) 
        # B * embed_size
        #这里估计进行了运算：（128,30000）*（30000,100）= 128(B) * 100 (embed_size)
        
        pos_embedding = self.out_embed(pos_labels) # B * (2*C) * embed_size
        #同上，增加了维度(2*C)，表示一个batch有B组周围词单词，一组周围词有(2*C)个单词，每个单词有embed_size个维度。
        
        neg_embedding = self.out_embed(neg_labels) # B * (2*C * K) * embed_size
        #同上，增加了维度(2*C*K)
      
    
        #torch.bmm()为batch间的矩阵相乘（b,n.m)*(b,m,p)=(b,n,p)
        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze() # B * (2*C)
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze() # B * (2*C*K)
        #unsqueeze(2)指定位置升维，.squeeze()压缩维度。
        
        #下面loss计算就是论文里的公式
        log_pos = F.logsigmoid(log_pos).sum(1)
        
        log_neg = F.logsigmoid(log_neg).sum(1) # batch_size     
        loss = log_pos + log_neg
        
        return -loss
    
    def input_embeddings(self):   #取出self.in_embed数据参数
        return self.in_embed.weight.data.cpu().numpy()
        
model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
#得到model，有参数，有loss，可以优化了

if USE_CUDA:
    model = model.cuda()
for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        print(input_labels, pos_labels, neg_labels)
        if i>5:
            break

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
#随机梯度下降

for e in range(NUM_EPOCHS): #开始迭代
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        #print(input_labels, pos_labels, neg_labels)
        
        # TODO
        input_labels = input_labels.long() #longtensor
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()
       
        #下面第一节课都讲过的   
        optimizer.zero_grad() #梯度归零
        loss = model(input_labels, pos_labels, neg_labels).mean()
        
        loss.backward()
        optimizer.step()
       
        #打印结果。
        if i % 100 == 0:
            with open(LOG_FILE, "a") as fout:
                fout.write("epoch: {}, iter: {}, loss: {}\n".format(e, i, loss.item()))
                print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))
            
        
#         if i % 2000 == 0:
#             embedding_weights = model.input_embeddings()
#             sim_simlex = evaluate(".//test//simlex-999.txt", embedding_weights)
#             sim_men = evaluate(".//test//men.txt", embedding_weights)
#             sim_353 = evaluate(".//test//wordsim353.csv", embedding_weights)
#             with open(LOG_FILE, "a") as fout:
#                 print("epoch: {}, iteration: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
#                     e, i, sim_simlex, sim_men, sim_353, find_nearest("monster")))
#                 fout.write("epoch: {}, iteration: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
#                     e, i, sim_simlex, sim_men, sim_353, find_nearest("monster")))
                
    embedding_weights = model.input_embeddings()
    np.save("embedding-{}".format(EMBEDDING_SIZE), embedding_weights)
    torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))