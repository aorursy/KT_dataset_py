!pip install snownlp

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from snownlp import SnowNLP

import jieba

from gensim import corpora, models

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data={}

data['comment'] = pd.read_csv('/kaggle/input/mengniutieba/comment.csv',header=None)

data['post'] = pd.read_csv('/kaggle/input/mengniutieba/post.csv',header=None,usecols=list(range(7)))



part1 = data['comment'][[2,3]]

part1.columns = ['text','time']

part2 = data['post'][[3,4]]

part2.columns = ['text','time']



text = pd.concat((part1,part2),axis=0, ignore_index=True)

del part1,part2,data
import re

replace_url = lambda s : re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',s)

replace_digit = lambda s : re.sub(r'[0-9a-zA-Z\t\n�]','',s)



text = text.drop_duplicates()



text = text.dropna()



text = text.sort_values(by='time')

text['time'] = text['time'].apply(lambda x:x[:7])

text = text.set_index('time').iloc[1:-2]



text['text'] = text['text'].apply(replace_url).apply(replace_digit)

text['len'] = text['text'].apply(lambda x:len(x))

text=text[text['len']>1]

TXT = text

del text
sentiment = TXT.iloc[:, 0].apply(lambda x: SnowNLP(x).sentiments)

serise = sentiment.groupby('time').agg(np.mean)
import matplotlib.pyplot as plt

import seaborn as sns     #more beautiful more simple but lack plasticity with seaborn

#seaborn depend on matplotlib



%matplotlib inline 

%pylab inline



plt.rcParams['font.sans-serif'] = ['SimHei'] #to show CH tag

plt.rcParams['axes.unicode_minus'] = False # to show signal

pylab.rcParams['figure.figsize'] = (10, 6)   #set scale of img

sns.set(color_codes=True) #seaborn set background



serise.plot()
from statsmodels.graphics.tsaplots import plot_acf  

from statsmodels.tsa.stattools import adfuller as ADF 

from statsmodels.graphics.tsaplots import plot_pacf    

from statsmodels.stats.diagnostic import acorr_ljungbox 

from statsmodels.tsa.arima_model import ARIMA

plot_acf(serise).show()

plot_pacf(serise).show()

print(u'ADF-test：', ADF(serise))

# print(u'Ljung-Box-test：',acorr_ljungbox(serise) )

model = ARIMA(serise, (1,0,1)).fit() #try model of ARIMA(1, 0, 1)

model.summary2()
T_ser = pd.Series(TXT['text'])

jieba.enable_parallel(4)

cut = lambda s: ' '.join(jieba.cut(s))

T_cut = T_ser.apply(cut)
stopword=[word.strip() for word in open("../input/stopwordch/stopwordCh.txt",'r').readlines()]

T_sep = T_cut.apply(lambda s: s.split(' ')).apply(lambda x: [i for i in x if i.encode('utf-8') not in stopword])
w_lst = T_sep['2018-11':].to_list()

words=[]

for lst in w_lst:

    words.extend(lst)

    

import wordcloud

w = wordcloud.WordCloud(font_path = '../input/chinesefonts/simkai.ttf')

w.generate(' '.join(words))

# s=' '.join(list(TXT['2018-11':]['text']))

# w.generate(s)

w.to_image()

# w.to_file('output1.png')
import jieba.posseg as jp

flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')#word type

seg_word = lambda s : [w.word for w in jp.cut(s) if w.flag in flags and w.word not in stopword]

Text = pd.concat((T_ser,sentiment),axis=1)

Text.columns=['text','index']

Text['seg'] = Text['text'].apply(seg_word)

# Text['index'>0.9]['text']
def LDA(word_list):

    words_ls=word_list

    dictionary = corpora.Dictionary(words_ls)



    corpus = [dictionary.doc2bow(words) for words in words_ls]



    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2)

    print('topics')

    for topic in lda.print_topics(num_words=5):

        print(topic)

    return topic

# positive part 

_ = LDA(Text[Text['index']>0.8]['seg'])

# negtive part 

_ = LDA(Text[Text['index']<0.2]['seg'])