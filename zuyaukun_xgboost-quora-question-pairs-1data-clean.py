#Colab中文字体解决办法，https://blog.csdn.net/xieyan0811/article/details/80371201

!wget -O /usr/share/fonts/truetype/liberation/simhei.ttf https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import nltk

from nltk.corpus import stopwords

from nltk import word_tokenize

from nltk.stem import SnowballStemmer

from nltk.stem import WordNetLemmatizer

import re

from string import punctuation
import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

matplotlib.rcParams['font.family']='simhei'#修改了全局变量

matplotlib.rcParams['font.size']=20



zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/liberation/simhei.ttf')

plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
path = '/kaggle/input/quora-question-pairs/'

csv_train = path+"train.csv"

csv_test = path+"test.csv"

train_orig = pd.read_csv(csv_train)

test_orig = pd.read_csv(csv_test)
display(train_orig.shape)

display(train_orig.head())

display(test_orig.shape)

display(test_orig.head())



print('潜在相似问题对数量: {}'.format(train_orig.shape[0]))

print('\'is_duplicate\' 正例比例: {}%'.format(round(train_orig['is_duplicate'].mean()*100, 2)))

qids = pd.Series(train_orig['qid1'].tolist() + train_orig['qid2'].tolist())

print('总问题数量: {}'.format(len(np.unique(qids))))

print('"重复问题" 出现次数: {}'.format(np.sum(qids.value_counts() > 1)))
plt.figure(figsize=(12, 5))

plt.hist(qids.value_counts(), bins=50)

plt.yscale('log', nonposy='clip')

plt.title('问题出现次数的对数直方图', fontproperties=zhfont)

plt.xlabel('问题出现的频数', fontproperties=zhfont)

plt.ylabel('频数计数', fontproperties=zhfont)

print(train_orig.isnull().sum())

print(test_orig.isnull().sum())
display(train_orig[train_orig.isnull().values==True])

display(test_orig[test_orig.isnull().values==True])
train_orig = train_orig.fillna(" ")

test_orig = test_orig.fillna(" ")
#apply显示进度条

from tqdm import tqdm

tqdm.pandas(desc="my bar!")
def common_words_transformation_remove_punctuation(text):

    #转换为小写

    text = text.lower()



    #清理字符

    text = re.sub(r"what's", "what is", text)

    text = re.sub(r"who's", "who is", text)

    text = re.sub(r"where's", "where is", text)

    text = re.sub(r"when's", "when is", text)

    text = re.sub(r"how's", "how is", text)

    text = re.sub(r"it's", "it is", text)

    text = re.sub(r"he's", "he is", text)

    text = re.sub(r"she's", "she is", text)

    text = re.sub(r"that's", "that is", text)

    text = re.sub(r"there's", "there is", text)



    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"\'s", " ", text)  # 除了上面的特殊情况外，“\'s”只能表示所有格，应替换成“ ”

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "can not ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am", text)

    text = re.sub(r" m ", " am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"60k", " 60000 ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e-mail", "email", text)

    text = re.sub(r"\s{2,}", " ", text)

    text = re.sub(r"quikly", "quickly", text)

    text = re.sub(r" usa ", " america ", text)

    text = re.sub(r" u s ", " america ", text)

    text = re.sub(r" uk ", " england ", text)

    text = re.sub(r"imrovement", "improvement", text)

    text = re.sub(r"intially", "initially", text)

    text = re.sub(r" dms ", "direct messages ", text)  

    text = re.sub(r"demonitization", "demonetization", text) 

    text = re.sub(r"actived", "active", text)

    text = re.sub(r"kms", " kilometers ", text)

    text = re.sub(r" cs ", " computer science ", text)

    text = re.sub(r" ds ", " data science ", text)

    text = re.sub(r" ee ", " electronic engineering ", text)

    text = re.sub(r" upvotes ", " up votes ", text)

    text = re.sub(r" iphone ", " phone ", text)

    text = re.sub(r"\0rs ", " rs ", text) 

    text = re.sub(r"calender", "calendar", text)

    text = re.sub(r"ios", "operating system", text)

    text = re.sub(r"programing", "programming", text)

    text = re.sub(r"bestfriend", "best friend", text)

    text = re.sub(r"III", "3", text) 

    text = re.sub(r"the us", "america", text)

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ", text)

    text = re.sub(r"\/", " ", text)

    text = re.sub(r"\^", " ", text)

    text = re.sub(r"\+", " ", text)

    text = re.sub(r"\-", " ", text)

    text = re.sub(r"\=", " ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " ", text)

    text = re.sub(r"\0s", "0", text)

    

    text = "".join([c for c in text if c not in punctuation])

        

    return text





#train_orig["question1"] = train_orig["question1"].apply(common_words_transformation_remove_punctuation)

#train_orig["question2"] = train_orig["question2"].apply(common_words_transformation_remove_punctuation)

#test_orig["question1"] = test_orig["question1"].apply(common_words_transformation_remove_punctuation)

#test_orig["question2"] = test_orig["question2"].apply(common_words_transformation_remove_punctuation)



train_orig["question1"] = train_orig["question1"].progress_apply(common_words_transformation_remove_punctuation)

train_orig["question2"] = train_orig["question2"].progress_apply(common_words_transformation_remove_punctuation)

test_orig["question1"] = test_orig["question1"].progress_apply(common_words_transformation_remove_punctuation)

test_orig["question2"] = test_orig["question2"].progress_apply(common_words_transformation_remove_punctuation)



#保存为文件

train_orig.to_csv("train_orig_trans.csv", index = False)

test_orig.to_csv("test_orig_trans.csv", index = False)



train_orig.head()
nltk.download('stopwords')

stopwords.words("english")



nltk.download('punkt')
def remove_stopwords(text):

    stops = set(stopwords.words("english"))

    text = word_tokenize(text)

    text = [w for w in text if not w in stops]

    text = " ".join(text)

    return text



train_stop, test_stop = train_orig.copy(deep = True), test_orig.copy(deep = True)



#train_stop["question1"] = train_stop["question1"].apply(remove_stopwords)

#train_stop["question2"] = train_stop["question2"].apply(remove_stopwords)

#test_stop["question1"] = test_stop["question1"].apply(remove_stopwords)

#test_stop["question2"] = test_stop["question2"].apply(remove_stopwords)



train_stop["question1"] = train_stop["question1"].progress_apply(remove_stopwords)

train_stop["question2"] = train_stop["question2"].progress_apply(remove_stopwords)

test_stop["question1"] = test_stop["question1"].progress_apply(remove_stopwords)

test_stop["question2"] = test_stop["question2"].progress_apply(remove_stopwords)



#保存为文件

train_stop.to_csv("train_stop.csv", index = False)

test_stop.to_csv("test_stop.csv", index = False)



train_stop.head()
def stem_words(text):

    text = word_tokenize(text)

    stemmer = SnowballStemmer("english")

    stemmed_words = [stemmer.stem(word) for word in text]

    text = " ".join(stemmed_words)

    return text



train_stem, test_stem = train_stop.copy(deep = True), test_stop.copy(deep = True)



#train_stem["question1"] = train_stem["question1"].progress_apply(stem_words)

#train_stem["question2"] = train_stem["question2"].progress_apply(stem_words)

#test_stem["question1"] = test_stem["question1"].progress_apply(stem_words)

#test_stem["question2"] = test_stem["question2"].progress_apply(stem_words)



train_stem["question1"] = train_stem["question1"].progress_apply(stem_words)

train_stem["question2"] = train_stem["question2"].progress_apply(stem_words)

test_stem["question1"] = test_stem["question1"].progress_apply(stem_words)

test_stem["question2"] = test_stem["question2"].progress_apply(stem_words)



#保存为文件

train_stem.to_csv("train_stem.csv", index = False)

test_stem.to_csv("test_stem.csv", index = False)



train_stem.head()
nltk.download('wordnet')
def lemmatize_words(text):

    text = word_tokenize(text)

    wordnet_lemmatizer = WordNetLemmatizer()

    lammatized_words = [wordnet_lemmatizer.lemmatize(word) for word in text]

    text = " ".join(lammatized_words)

    return text



train_lem, test_lem = train_stop.copy(deep = True), test_stop.copy(deep = True)



#train_lem["question1"] = train_lem["question1"].apply(lemmatize_words)

#train_lem["question2"] = train_lem["question2"].apply(lemmatize_words)

#test_lem["question1"] = test_lem["question1"].apply(lemmatize_words)

#test_lem["question2"] = test_lem["question2"].apply(lemmatize_words)



train_lem["question1"] = train_lem["question1"].progress_apply(lemmatize_words)

train_lem["question2"] = train_lem["question2"].progress_apply(lemmatize_words)

test_lem["question1"] = test_lem["question1"].progress_apply(lemmatize_words)

test_lem["question2"] = test_lem["question2"].progress_apply(lemmatize_words)



train_lem.to_csv("train_lem.csv", index = False)

test_lem.to_csv("test_lem.csv", index = False)



train_lem.head()
display(train_orig.head(3)) #符号转义和清除

display(train_stop.head(3)) #去除停用词

display(train_stem.head(3)) #提取单词词根

display(train_lem.head(3)) #还原单词的原型
#train_orig = pd.read_csv('train_orig_trans.csv')



train_qs = pd.Series(train_orig['question1'].tolist() + train_orig['question2'].tolist()).astype(str)



from wordcloud import WordCloud

cloud_train = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud_train)

plt.axis('off')

plt.show()
#test_orig = pd.read_csv('test_orig_trans.csv')



test_qs = pd.Series(test_orig['question1'].tolist() + test_orig['question2'].tolist()).astype(str)



from wordcloud import WordCloud

cloud_test = WordCloud(width=1440, height=1080).generate(" ".join(test_qs.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud_test)

plt.axis('off')

plt.show()
!ls -lh 