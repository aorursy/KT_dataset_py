!pip install -U newspaper3k
!pip install -U stanza
!pip install -U numpy
import stanza
import hashlib

import requests
from newspaper import Article
from newspaper.article import ArticleException

import numpy as np
from typing import List, Dict, Tuple, Iterable, Type
from numpy import ndarray
stanza.download('en')
nlp = stanza.Pipeline(processors='tokenize', lang='en', use_gpu=True)
URL = 'https://edition.cnn.com/2020/03/04/health/debunking-coronavirus-myths-trnd/'
print(URL)
def parse_article(url):
    try:
        r = requests.get(url, timeout=10)
        article = Article(url)
        article.download()
        article.parse()
    except:
        article = None
    return article
article = parse_article(URL)
ALL_SENTENCES = []

file_id = hashlib.md5(article.url.encode()).hexdigest()
article_url = article.url
article_published_datetime = article.publish_date

article_title = article.title
doc = nlp(article_title)
for sent in doc.sentences:
    #S = ' '.join([w.text for w in sent.words])
    S = str(sent.text)
    sH = hashlib.md5(S.encode('utf-8')).hexdigest()
    print(file_id, sH, "%10i"%len(ALL_SENTENCES), (S,))
    ALL_SENTENCES.append([file_id, sH, S, article_published_datetime, article_url])

article_text = [a.strip() for a in article.text.splitlines() if a.strip()]
for paragraph in article_text:
    doc = nlp(paragraph)
    for sent in doc.sentences:
        S = str(sent.text)
        sH = hashlib.md5(S.encode('utf-8')).hexdigest()
        print(file_id, sH, "%10i"%len(ALL_SENTENCES), (S,))
        ALL_SENTENCES.append([file_id, sH, S, article_published_datetime, article_url])

len(ALL_SENTENCES)
CLAIMS, NO_CLAIMS = [], []
for file_id, sH, sentence, article_published_datetime, article_url in ALL_SENTENCES:
    # clean the sentences (from "obvious" markers)
    if 'Myth:' in sentence:
        sentence = sentence.replace('Myth:','').strip()
        print(sH)
        print(sentence)
        print()
        CLAIMS.append(sentence)
    else:
        sentence = sentence.replace('Reality:','').strip()
        NO_CLAIMS.append(sentence)
len(CLAIMS), len(NO_CLAIMS)
X, Y = [], []
for c in list(set(CLAIMS)):
    X.append(c)
    Y.append('claim')
for s in list(set(NO_CLAIMS)):
    X.append(s)
    Y.append('no_claim')
len(X), len(Y)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42, shuffle=True)
len(X_train), len(X_test), len(Y_train), len(Y_test)
from collections import Counter
print(len(Y_train), len(Y_test))
C_train = Counter(Y_train)
print(C_train)
C_test = Counter(Y_test)
print(C_test)
print("%.2f"%(100*C_train['claim'] / sum(list(C_train.values()))), "%", "claims in train set!")
print("%.2f"%(100*C_test['claim'] / sum(list(C_test.values()))), "%", "claims in test set!")
input_length_chars = []
input_length_words = []
for x,l in zip(X_train, Y_train):
    l = len(x)
    input_length_chars.append(l)
    doc = nlp(x)
    for sent in doc.sentences:
        S = ' '.join([w.text for w in sent.words])
        L = len(S.split(' '))
        input_length_words.append(L)
    print("%-10s"%l, "%4i"%l, "%2i"%L, x)
(min(input_length_chars), max(input_length_chars)), (min(input_length_words), max(input_length_words))
LENGTH_IN_CHAR_MIN = 10
LENGTH_IN_CHAR_MAX = 250
!wget https://github.com/minimaxir/char-embeddings/raw/master/glove.840B.300d-char.txt
with open('glove.840B.300d-char.txt','r') as f:
    pretrained_glove_char_embeddings_txt = f.readlines()
len(pretrained_glove_char_embeddings_txt)
pretrained_glove_char_embeddings = dict()
for v in pretrained_glove_char_embeddings_txt:
    v = v.strip().split(' ')
    char = v[0]
    vec = np.asarray([float(f) for f in v[1:]])
    assert vec.shape[0]==300
    print(char, vec.shape)
    pretrained_glove_char_embeddings[char] = vec
len(pretrained_glove_char_embeddings)
np.random.seed(42)

char = 'UNK'
vec = np.random.rand(300,)
print(char, vec.shape)

pretrained_glove_char_embeddings[char] = vec
len(pretrained_glove_char_embeddings)
class Embedder():
    def __init__(self, min_char_len, max_char_len, emb_dim=300):
        self.min_char_len = min_char_len #TODO: check for min length
        self.max_char_len = max_char_len
        self.emb_dim = emb_dim
    #    self.pretrained_glove_char_embeddings = _load_char_embeddings()
    
    #def _load_char_embeddings():
        
        
    def encode(self, sentences: List[str]) -> List[ndarray]:
        sentences_emb = []
        for sentence in sentences:
            self.sentence_emb_seq = np.zeros((min(len(sentence),self.max_char_len),self.emb_dim))
            for i, char in enumerate(sentence[:self.max_char_len]):
                if char in pretrained_glove_char_embeddings.keys():
                    self.sentence_emb_seq[i] = pretrained_glove_char_embeddings[char]
                else:
                    self.sentence_emb_seq[i] = pretrained_glove_char_embeddings['UNK']
            self.sentence_emb = np.average(self.sentence_emb_seq, axis=0)
            sentences_emb.append(self.sentence_emb)
        return np.asarray(sentences_emb)
from sklearn.ensemble import RandomForestClassifier
emb = Embedder(min_char_len=10, max_char_len=250)
X_train_emb = emb.encode(X_train)
len(X_train_emb), X_train_emb[0].shape
clr_rf = RandomForestClassifier(max_depth=1, n_estimators=10)
clr_rf.fit(X_train_emb, Y_train)
X_test_emb = emb.encode(X_test)
len(X_test_emb), X_test_emb[0].shape
Y_pred_rfc = clr_rf.predict(X_test_emb)
len(Y_pred_rfc)
p,r,f1,s = precision_recall_fscore_support(y_true=Y_test, y_pred=Y_pred_rfc, average='macro', warn_for=tuple())
print("%.3f Precision\n%.3f Recall\n%.3f F1"%(p,r,f1))
print(classification_report(y_true=Y_test, y_pred=Y_pred_rfc))
for i,l in enumerate(Y_pred_rfc):
    if l=='claim':
        print(X_test[i])
!pip install -U skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr_rf, initial_types=initial_type)
with open("rfc_claims.onnx", "wb") as f:
    f.write(onx.SerializeToString())
!pip install -U onnxruntime
import onnxruntime as rt
sess = rt.InferenceSession("rfc_claims.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], 
    {
        input_name: X_test_emb.astype(np.float32)
    }
)[0]