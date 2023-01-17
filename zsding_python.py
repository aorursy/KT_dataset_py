import warnings

warnings.filterwarnings("ignore")                     #Ignoring unnecessory warnings



import numpy as np                                  #for large and multi-dimensional arrays

import pandas as pd                                 #for data manipulation and analysis

import nltk                                         #Natural language processing tool-kit



from nltk.corpus import stopwords                   #Stopwords corpus

from nltk.stem import PorterStemmer                 # Stemmer



from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words

from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF

from gensim.models import Word2Vec                                   #For Word2Vec

data_path = "../input/Reviews.csv"

data = pd.read_csv(data_path)

data_sel = data.head(10000)                                #Considering only top 10000 rows
# Shape of our data

data_sel.columns
data_score_removed = data_sel[data_sel['Score']!=3]       #中立评论过滤掉
def partition(x):

    if x < 3:

        return 'positive'

    return 'negative'



score_upd = data_score_removed['Score']

# print(score_upd)

t = score_upd.map(partition)    #这儿用map函数

data_score_removed['Score']=t
'''

drop_duplicates(self, subset=None, keep='first', inplace=False)



Return DataFrame with duplicate rows removed, optionally only

considering certain columns.



Parameters

----------

subset : column label or sequence of labels, optional

    Only consider certain columns for identifying duplicates, by

    default use all of the columns

keep : {'first', 'last', False}, default 'first'

    - ``first`` : Drop duplicates except for the first occurrence.

    - ``last`` : Drop duplicates except for the last occurrence.

    - False : Drop all duplicates.

inplace : boolean, default False

    Whether to drop duplicates in place or to return a copy



Returns

-------

deduplicated : DataFrame

None

'''

#上面的注解是通过np.info获取的pandas.DateFrame.drop_duplicates的函数

#将subset中四列全部相同的进行去重

final_data = data_score_removed.drop_duplicates(subset={"UserId","ProfileName","Time","Text"})

# print(final_data)  #剩余8718行数组
#再次确认下，负面评价的评分比正面评价的低

final = final_data[final_data['HelpfulnessNumerator'] <= final_data['HelpfulnessDenominator']]

final_X = final['Text']

final_y = final['Score']
stop = set(stopwords.words('english')) 

print(stop)
import re

temp =[]

snow = nltk.stem.SnowballStemmer('english')

for sentence in final_X:

    sentence = sentence.lower()                 # 变小写

    cleanr = re.compile('<.*?>')                #？是非贪婪匹配，这个正则表达式匹配的是<>包裹的标签

    sentence = re.sub(cleanr, ' ', sentence)        #Removing HTML tags

    sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)        #Removing Punctuations 移除标点符号

    

    words = [snow.stem(word) for word in sentence.split() if word not in stopwords.words('english')]   # Stemming and removing stopwords

    temp.append(words)

    

final_X = temp    
print(final_X[1])
sent = []

for row in final_X:

    sequ = ''

    for word in row:

        sequ = sequ + ' ' + word

    sent.append(sequ)



final_X = sent

print(final_X[1])
count_vect = CountVectorizer(max_features=5000)

bow_data = count_vect.fit_transform(final_X)

print(bow_data[1])

final_B_X = final_X

print(final_B_X[1])
#ngram_range的参数设为（1,2）就是 Bi-Gram BoW

count_vect = CountVectorizer(ngram_range=(1,2))

Bigram_data = count_vect.fit_transform(final_B_X)

print(Bigram_data[1])
'''

两个步骤：

    1.依据已有文本做出词典dictionary

    2. 计算句子分词后的tf-idf值，得到一个向量，这个向量的大小等于词典

'''

final_tf = final_X

tf_idf = TfidfVectorizer(max_features=5000)

tf_data = tf_idf.fit_transform(final_tf)

print(tf_data[1])
w2v_data = final_X

print(w2v_data[1])
splitted = []

for row in w2v_data: 

    splitted.append([word for word in row.split()])     #splitting words
train_w2v = Word2Vec(splitted,min_count=5,size=50, workers=4)
avg_data = []

for row in splitted:

    vec = np.zeros(50)

    count = 0

    for word in row:

        try:

            vec += train_w2v[word]

            count += 1

        except:

            pass

    avg_data.append(vec/count)

    
print(avg_data[1])

print(np.shape(avg_data[1]))
tf_w_data = final_X

tf_idf = TfidfVectorizer(max_features=5000)

tf_idf_data = tf_idf.fit_transform(tf_w_data)

print(tf_idf_data[1])
tf_w_data = []

tf_idf_data = tf_idf_data.toarray()

i = 0

for row in splitted:     #每一行

    vec = [0 for i in range(50)]

    

    temp_tfidf = []

    for val in tf_idf_data[i]:     #将每一行单词的tf-idf值加入到列表中

        if val != 0:

            temp_tfidf.append(val)

    

    count = 0

    tf_idf_sum = 0

    for word in row:    #每条评论的每一个单词

        try:

            count += 1

            tf_idf_sum = tf_idf_sum + temp_tfidf[count-1]    #求一条评论的所有单词的tf-idf值的和

            vec += (temp_tfidf[count-1] * train_w2v[word])    #shape是（50,）

        except:

            pass

    vec = (float)(1/tf_idf_sum) * vec

    tf_w_data.append(vec)

    i = i + 1



print(tf_w_data[1])

    