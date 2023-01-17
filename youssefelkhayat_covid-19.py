import os
from textblob import TextBlob
from nltk.corpus import stopwords
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
root_path = '/kaggle/input/CORD-19-research-challenge/'

dirs = [
    root_path+'document_parses/pdf_json',
]
documents = []
for d in dirs:
    for file in tqdm(os.listdir(d)):
        j = json.load(open(d+f"/{file}","rb")) 
        
        title = j['metadata']['title']
        
        abstract = ""
        if len(j['abstract']) > 0:
            abstract = j['abstract'][0]["text"]
            
        text = ''
        for t in j["body_text"]:
            text += t['text'] + "\n\n"
            
        documents += [[title, abstract, text]] 

df = pd.DataFrame(documents, columns=["title", 'abstract','text'])

df.head()
import gensim

path = "../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin"

model = gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = df['text']

vectorizer = TfidfVectorizer()
X = vectorizer.fit(corpus)
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

processed_abstracts = []
for abstract in tqdm(df['abstract'].tolist()):
    # Normalization
    normalized = re.sub(r"[^a-zA-Z0-9]", " ", abstract.lower())

    # Tokenization
    words = word_tokenize(normalized)

    # Removing stopwords
    no_stopwords = [w for w in words if w not in stopwords.words("english")]

    # Stemming 
    p = PorterStemmer()
    stemmed = [p.stem(w) for w in no_stopwords]

    processed_abstracts += [stemmed]

df['processed_abstracts'] = [ " ".join(x) for x in processed_abstracts ]
df.head()
def getDocumnetsContain(keyword):
    return df[df['abstract'].str.contains(keyword)]

getDocumnetsContain('pregnancy').head()
from nltk.tokenize import sent_tokenize

def getSentencesContain(keyword):
    def sentContain(x):
        # tokenize sentences 
        sentences  = sent_tokenize(x)
        # filter el conatins el keyword
        return [sent for sent in sentences if keyword in sent]

    return np.array(list(map(sentContain, np.array(getDocumnetsContain(keyword)['text']))))
getSentencesContain("risk")
def getDocumentsAbout(keyword, limitCount=10):
    sents = getSentencesContain(keyword)
    docs = getDocumnetsContain(keyword)
    return docs[[(len(s)>limitCount) for s in sents]].iloc[:, 0:3]
getDocumentsAbout('smoking', 5).head()
def getDocumentWithMostFreq(keyword):
    index = vectorizer.get_feature_names().index(keyword)
    docs = getDocumentsAbout(keyword)
    if(len(docs) > 0):
        maxDoc = np.argmax(vectorizer.transform(docs['text'])[:,index].toarray())
    return docs.iloc[maxDoc] if (len(docs) > 0) else "No documents found"
getDocumentWithMostFreq("pregnancy")['title']
def getMostSimilarNDocuments(s, topNumber = 5):
    from sklearn.metrics.pairwise import linear_kernel
    sent = [s]
    sentVec = vectorizer.transform(sent)
    d = []
    for word in s.split(' '):
        d += [getDocumnetsContain(word)]
    d = pd.concat(d, ignore_index=True)
    docsVec = vectorizer.transform(d['text'])
    cosine_similarities = linear_kernel(sentVec, docsVec).flatten()
    indices = cosine_similarities.argsort()[:-topNumber:-1]
    return df.iloc[indices]
getMostSimilarNDocuments('smoking risk factors pregnant drinking', 2)
# Top Tfidf Score
def topUsedWordsWith(keyword, visualize=True):
    docs = getDocumentsAbout(keyword)
    if(len(docs) == 0): return "No documents found"
    tfidf_result =  vectorizer.transform(docs['text'])
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    count = 0
    words = []
    for item in sorted_scores:
        if item[0] not in stopwords.words("english"):
            words += [(item[0], item[1])]
            count+=1
        if count > 20:
            break
    
    k = words
    names = [x[0] for x in k]
    values = [x[1] for x in k]

    plt.figure(figsize=(30, 7))  
    plt.bar(range(len(names)),values,tick_label=names)
    plt.show()
    
    return words
topUsedWordsWith('risk')
def getSimilarWords(keyword, number = 3):
     return model.most_similar(positive=[keyword], topn = number)
def relevantSentenceQuery(keyword, filters = [], number = 3):
    similar_keywords = [(keyword,0)] + model.most_similar(positive=[keyword], topn = number)
    print(f"similar words to '{keyword}'': {similar_keywords[0][0]}, {similar_keywords[1][0]}, {similar_keywords[2][0]}")
    result = []
    for key in similar_keywords:
        x = getSentencesContain(key[0])
        result += getSentencesContain(key[0]).tolist()
    flattened = []
    for i in result:
        for j in i:
            add = True
            for f in filters:
                add *= (f in j)
            if add:
                flattened += [j]
            
    return list(set(flattened))
relevantSentenceQuery('smoking', filters= ['risk', 'factor', 'progression','COVID-19'])
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
def flatten(l):
    flattened = []
    for i in l:
        for j in i:
            flattened += [j]
    return flattened
def graphWordsCount(words):
    counts = []
    for w in words:
        counts += [len(flatten(getSentencesContain(w)))]
    
    
    k = words
    names = words
    values = counts

    plt.figure(figsize=(30, 7))  
    plt.bar(range(len(names)),values,tick_label=names)
    # plt.savefig('bar.png')
    plt.show()
graphWordsCount(["drinking", "pregnant", "drinking"])