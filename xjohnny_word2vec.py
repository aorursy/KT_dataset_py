# 加载必要的程序包

# PyTorch的程序包

import torch

from torch.autograd import Variable

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



# 数值运算和绘图的程序包

import numpy as np

import matplotlib.pyplot as plt

import matplotlib





# 加载机器学习的软件包

from sklearn.decomposition import PCA



#加载Word2Vec的软件包

import gensim as gensim

from gensim.models import Word2Vec

from gensim.models.keyedvectors import KeyedVectors

from gensim.models.word2vec import LineSentence



#加载‘结巴’中文分词软件包



import jieba



#加载正则表达式处理的包

import re



%matplotlib inline
#读入原始文件



#f = open("../input/三体.txt", 'r', encoding='utf-8')

# 若想加快运行速度，使用下面的语句（选用了三体的其中一章）：

f = open("../input/wordvector/3body.txt", 'r', encoding='utf-8') 

text = str(f.read())

f.close()



text
# 分词

temp = jieba.lcut(text)

print(len(temp))

words = []

for i in temp:

    #过滤掉所有的标点符号

    i = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", i)

    if len(i) > 0:

        words.append(i)

print(len(words))

words
# 构建三元组列表.  每一个元素为： ([ i-2位置的词, i-1位置的词 ], 下一个词)

# 我们选择的Ngram中的N，即窗口大小为2

trigrams = [([words[i], words[i + 1]], words[i + 2]) for i in range(len(words) - 2)]

# 打印出前三个元素看看

print(trigrams[:3])
# 得到词汇表

vocab = set(words)

print(len(vocab))

# 两个字典，一个根据单词索引其编号，一个根据编号索引单词

#word_to_idx中的值包含两部分，一部分为id，另一部分为单词出现的次数

#word_to_idx中的每一个元素形如：{w:[id, count]}，其中w为一个词，id为该词的编号，count为该单词在words全文中出现的次数

word_to_idx = {} 

idx_to_word = {}

ids = 0



#对全文循环，构建这两个字典

for w in words:

    cnt = word_to_idx.get(w, [ids, 0])

    if cnt[1] == 0:

        ids += 1

    cnt[1] += 1

    word_to_idx[w] = cnt

    idx_to_word[ids] = w
word_to_idx
class NGram(nn.Module):



    def __init__(self, vocab_size, embedding_dim, context_size):

        super(NGram, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  #嵌入层

        self.linear1 = nn.Linear(context_size * embedding_dim, 128) #线性层

        self.linear2 = nn.Linear(128, vocab_size) #线性层



    def forward(self, inputs):

        #嵌入运算，嵌入运算在内部分为两步：将输入的单词编码映射为one hot向量表示，然后经过一个线性层得到单词的词向量

        #inputs的尺寸为：1*context_size

        embeds = self.embeddings(inputs)

        #embeds的尺寸为: context_size*embedding_dim

        embeds = embeds.view(1, -1)

        #此时embeds的尺寸为：1*embedding_dim

        # 线性层加ReLU

        out = self.linear1(embeds)

        out = F.relu(out)

        #此时out的尺寸为1*128

        

        # 线性层加Softmax

        out = self.linear2(out)

        #此时out的尺寸为：1*vocab_size

        log_probs = F.log_softmax(out, dim = 1)

        return log_probs

    def extract(self, inputs):

        embeds = self.embeddings(inputs)

        return embeds
losses = [] #纪录每一步的损失函数

criterion = nn.NLLLoss() #运用负对数似然函数作为目标函数（常用于多分类问题的目标函数）

model = NGram(len(vocab), 10, 2) #定义NGram模型，向量嵌入维数为10维，N（窗口大小）为2

optimizer = optim.SGD(model.parameters(), lr=0.001) #使用随机梯度下降算法作为优化器



#循环100个周期

for epoch in range(100):

    total_loss = torch.Tensor([0])

    for context, target in trigrams:



        # 准备好输入模型的数据，将词汇映射为编码

        context_idxs = [word_to_idx[w][0] for w in context]

        

        # 包装成PyTorch的Variable

        context_var = torch.tensor(context_idxs, dtype = torch.long)



        # 清空梯度：注意PyTorch会在调用backward的时候自动积累梯度信息，故而每隔周期要清空梯度信息一次。

        optimizer.zero_grad()



        # 用神经网络做计算，计算得到输出的每个单词的可能概率对数值

        log_probs = model(context_var)



        # 计算损失函数，同样需要把目标数据转化为编码，并包装为Variable

        loss = criterion(log_probs, torch.tensor([word_to_idx[target][0]], dtype = torch.long))



        # 梯度反传

        loss.backward()

        

        # 对网络进行优化

        optimizer.step()

        

        # 累加损失函数值

        total_loss += loss.data

    losses.append(total_loss)

    print('第{}轮，损失函数为：{:.2f}'.format(epoch, total_loss.numpy()[0]))
# 从训练好的模型中提取每个单词的向量

vec = model.extract(torch.tensor([v[0] for v in word_to_idx.values()], dtype = torch.long))

vec = vec.data.numpy()



# 利用PCA算法进行降维

X_reduced = PCA(n_components=2).fit_transform(vec)





# 绘制所有单词向量的二维空间投影

fig = plt.figure(figsize = (30, 20))

ax = fig.gca()

ax.set_facecolor('white')

ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.4, color = 'black')





# 绘制几个特殊单词的向量

words = ['智子', '地球', '三体', '质子', '科学', '世界', '文明', '太空', '加速器', '平面', '宇宙', '信息']



# 设置中文字体，否则无法在图形上显示中文

zhfont1 = matplotlib.font_manager.FontProperties(fname='../input/wordvector/华文仿宋.ttf', size=16)

for w in words:

    if w in word_to_idx:

        ind = word_to_idx[w][0]

        xy = X_reduced[ind]

        plt.plot(xy[0], xy[1], '.', alpha =1, color = 'red')

        plt.text(xy[0], xy[1], w, fontproperties = zhfont1, alpha = 1, color = 'black')
# 定义计算cosine相似度的函数

def cos_similarity(vec1, vec2):

    

    norm1 = np.linalg.norm(vec1)

    norm2 = np.linalg.norm(vec2)

    norm = norm1 * norm2

    dot = np.dot(vec1, vec2)

    result = dot / norm if norm > 0 else 0

    return result

    

# 在所有的词向量中寻找到与目标词（word）相近的向量，并按相似度进行排列

def find_most_similar(word, vectors, word_idx):

    vector = vectors[word_to_idx[word][0]]

    simi = [[cos_similarity(vector, vectors[num]), key] for num, key in enumerate(word_idx.keys())]

    sort = sorted(simi)[::-1]

    words = [i[1] for i in sort]

    return words



# 与智子靠近的词汇

find_most_similar('智子', vec, word_to_idx)
# 读入文件、分词，形成一句一句的语料

# 注意跟前面处理不一样的地方在于，我们一行一行地读入文件，从而自然利用行将文章分开成“句子”

f = open("../input/wordvector/三体.txt", 'r', encoding='utf-8')

lines = []

for line in f:

    temp = jieba.lcut(line)

    words = []

    for i in temp:

        #过滤掉所有的标点符号

        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)

        if len(i) > 0:

            words.append(i)

    if len(words) > 0:

        lines.append(words)

    
# 调用Word2Vec的算法进行训练。

# 参数分别为：size: 嵌入后的词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值

model = Word2Vec(lines, size = 20, window = 2 , min_count = 0)
model.wv.most_similar('智子', topn = 20)
# 将词向量投影到二维空间

rawWordVec = []

word2ind = {}

for i, w in enumerate(model.wv.vocab):

    rawWordVec.append(model[w])

    word2ind[w] = i

rawWordVec = np.array(rawWordVec)

X_reduced = PCA(n_components=2).fit_transform(rawWordVec)
# 绘制星空图

# 绘制所有单词向量的二维空间投影

fig = plt.figure(figsize = (15, 10))

ax = fig.gca()

ax.set_facecolor('white')

ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.3, color = 'black')





# 绘制几个特殊单词的向量

words = ['智子', '地球', '三体', '质子', '科学', '世界', '文明', '太空', '加速器', '平面', '宇宙', '进展','的']



# 设置中文字体，否则无法在图形上显示中文

zhfont1 = matplotlib.font_manager.FontProperties(fname='../input/wordvector/华文仿宋.ttf', size=16)

for w in words:

    if w in word2ind:

        ind = word2ind[w]

        xy = X_reduced[ind]

        plt.plot(xy[0], xy[1], '.', alpha =1, color = 'green')

        plt.text(xy[0], xy[1], w, fontproperties = zhfont1, alpha = 1, color = 'blue')
# 加载词向量

word_vectors = KeyedVectors.load_word2vec_format('../input/chinese-word-vector/vectors.bin', binary=True, unicode_errors='ignore')

len(word_vectors.vocab)
# PCA降维

rawWordVec = []

word2ind = {}

for i, w in enumerate(word_vectors.vocab):

    rawWordVec.append(word_vectors[w])

    word2ind[w] = i

rawWordVec = np.array(rawWordVec)

X_reduced = PCA(n_components=2).fit_transform(rawWordVec)
# 查看相似词

word_vectors.most_similar('物理', topn = 20)
# 绘制星空图

# 绘制所有的词汇

fig = plt.figure(figsize = (30, 15))

ax = fig.gca()

ax.set_facecolor('black')

ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.1, color = 'white')



ax.set_xlim([-12,12])

ax.set_ylim([-10,20])





# 选择几个特殊词汇，不仅画它们的位置，而且把它们的临近词也画出来

words = {'徐静蕾','吴亦凡','物理','红楼梦','量子'}

all_words = []

for w in words:

    lst = word_vectors.most_similar(w)

    wds = [i[0] for i in lst] # 词

    metrics = [i[1] for i in lst] # 相似度

    wds = np.append(wds, w)

    all_words.append(wds)





zhfont1 = matplotlib.font_manager.FontProperties(fname='../input/wordvector/华文仿宋.ttf', size=16)

colors = ['red', 'yellow', 'orange', 'green', 'cyan', 'cyan']

for num, wds in enumerate(all_words):

    for w in wds:

        if w in word2ind:

            ind = word2ind[w] # 获取index

            xy = X_reduced[ind] # 获取每个word的二维坐标

            plt.plot(xy[0], xy[1], '.', alpha =1, color = colors[num])

            plt.text(xy[0], xy[1], w, fontproperties = zhfont1, alpha = 1, color = colors[num])

plt.savefig('88.png',dpi =600)
# 女人－男人＝？－国王

words = word_vectors.most_similar(positive=['女人', '国王'], negative=['男人'])

words
# 北京－中国＝？－俄罗斯

words = word_vectors.most_similar(positive=['北京', '俄罗斯'], negative=['中国'])

words
# 自然科学－物理学＝？－政治学

words = word_vectors.most_similar(positive=['自然科学', '政治学'], negative=['物理学'])

words
# 王菲－章子怡＝？－汪峰

words = word_vectors.most_similar(positive=['王菲', '汪峰'], negative=['章子怡'])

words
# 尽可能多地选出所有的货币

words = word_vectors.most_similar(positive=['美元', '英镑'], topn = 100)

words = word_vectors.most_similar(positive=['美元', '英镑', '日元'], topn = 100)

#words = word_vectors.most_similar(positive=['美元', '英镑', '日元'], negative = ['原油价格', '7800万'], topn = 100)

words