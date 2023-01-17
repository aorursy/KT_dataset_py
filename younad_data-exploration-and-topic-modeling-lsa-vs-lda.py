import numpy as np 
import pandas as pd 
import matplotlib as mp
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation,TruncatedSVD
import matplotlib.pyplot as plt
#uploading data in dataframe
train=pd.read_csv("../input/train.csv",sep=',')
test=pd.read_csv("../input/test.csv",sep=',')
#displayin shapes
print ('train shapes : %s'%str(train.shape))
print ('test shapes : %s'%str(test.shape))
#displaying exemple data
train.head(5)
#displaying exemple of insincere data 
train[train.target==1].head(5)
#displayin dataframe info
train.info()
#counting target values
train.target.value_counts()
train['word_count'] = train['question_text'].apply(lambda x: len(str(x).split(" ")))
#train['char_count'] = train['question_text'].str.len()
#stop = stopwords.words('english')
#train['stopwords'] = train['question_text'].apply(lambda x: len([x for x in x.split() if x in stop]))
#train['numerics'] = train['question_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
#train['upper'] = train['question_text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))

#basic statistic about word_count
train.word_count.describe()
#ploting box plot of word_count by target without outlier
train.boxplot(column='word_count', by='target', grid=False,showfliers=False)
#lower case
train['question_text'] = train['question_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#Removing Punctuation
train['question_text'] = train['question_text'].str.replace('[^\w\s]','')
#Removing numbers
train['question_text'] = train['question_text'].str.replace('[0-9]','')
#Remooving stop words and words with length <=2
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['question_text'] = train['question_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop and len(x)>2))
#Stemming
#from nltk.stem import SnowballStemmer
#ss=SnowballStemmer('english')
#train['question_text'] = train['question_text'].apply(lambda x: " ".join(ss.stem(x) for x in x.split()))
from nltk.stem import WordNetLemmatizer
wl = WordNetLemmatizer()
train['question_text'] = train['question_text'].apply(lambda x: " ".join(wl.lemmatize(x,'v') for x in x.split()))
from nltk.stem import SnowballStemmer,WordNetLemmatizer,PorterStemmer,LancasterStemmer
wl = WordNetLemmatizer()
ss=SnowballStemmer('english')
ps=PorterStemmer()
ls=LancasterStemmer()
test_list=['does','peaople','writing','beards','enjoyment','bought','leaves','gave','given','generaly','would']
for item in test_list :
    print('lemmatizer : %s'%wl.lemmatize(item,'v'))
    print('SS stemmer : %s'%ss.stem(item))
    print('PS stemmer : %s'%ps.stem(item))
    print('LS stemmer : %s'%ls.stem(item))

train.head(5)
def get_words_freq(corpus):
    vec = CountVectorizer(ngram_range={1,2}).fit(corpus)
    #bag of words its a sparse document item matrix
    bag_of_words = vec.transform(corpus)
    #we calculate the occurrence for each term. warning, the sum of matrix is a 1 row matrix
    sum_words = bag_of_words.sum(axis=0) 
    # Vocabulary_ its a dictionary { word :position }  
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],reverse=True)
    return words_freq
top_sincere=get_words_freq(train[train.target==0].question_text)
print(top_sincere[:30])
top_insincere=get_words_freq(train[train.target==1].question_text)
print(top_insincere[:30])
#[y[0] for y in top_sincere].index('black people')
from wordcloud import WordCloud
wc=WordCloud(background_color='white')
wc.generate(''.join(train[train.target==1].question_text))
#let's plot
plt.figure(1, figsize=(15, 15))
plt.axis('off')
plt.imshow(wc)
plt.show()
tfidf_v = TfidfVectorizer(min_df=20,max_df=0.8,sublinear_tf=True,ngram_range={1,2})
#matrixTFIDF= tfidf_v.fit_transform(train.question_text)
matrixTFIDF= tfidf_v.fit_transform(train[train.target==1].question_text)
print(matrixTFIDF.shape)
plt.boxplot(np.array(matrixTFIDF.mean(axis=0).transpose()),showfliers=False)
plt.show()
svd=TruncatedSVD(n_components=15, n_iter=10,random_state=42)
X=svd.fit_transform(matrixTFIDF)             
plt.plot(svd.singular_values_[0:15])
#Explained variance by our components
np.sum(svd.explained_variance_ratio_[0:15])
#components_ give the word contribution for each component 
svd.components_.shape
def get_topics(components, feature_names, n=15):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx))
        print([(feature_names[i], topic[i])
                        for i in topic.argsort()[:-n - 1:-1]])
get_topics(svd.components_,tfidf_v.get_feature_names())
lda=LatentDirichletAllocation(n_components=15,random_state=42,max_iter=10)
Z=lda.fit_transform(matrixTFIDF)  
get_topics(lda.components_,tfidf_v.get_feature_names(),n=15)