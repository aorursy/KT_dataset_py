!wget https://raw.githubusercontent.com/DeFacto/SimpleLSTM/master/data/fever/fever_simple_claim/fever_simple_claim.json
import json
import pandas as pd
import datetime as dt
with open('fever_simple_claim.json','r') as my_file:
    fever_simple_claim = json.load(my_file)
len(fever_simple_claim)
for count_l, FSC in enumerate(fever_simple_claim):
    print("%4i"%count_l,'-'*80)
    for k, V in FSC.items():
        print(k)
        print(len(V), type(V))
        if type(V)==list:
            for v in V:
                print((v,))
        else:
            print(V)
        print()
    print()
    if count_l>=0:
        break
DATA_list = []
for count_l, FSC in enumerate(fever_simple_claim):
    c = FSC['claim']
    DATA_list.append(['claim',c])
    for s in FSC['sentences']:
        DATA_list.append(['sentences',s])
len(DATA_list)
DATA = pd.DataFrame(DATA_list, columns=['type','sentence'])
len(DATA)
DATA.groupby('type').count()
!pip install -U sentence_transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
X, Y = [], []
for x, y in DATA[['sentence','type']].values.tolist():
    X.append(x)
    Y.append(y)
len(X), len(Y)
!pip install -U sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.svm import SVC
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
len(X_train), len(X_test), len(Y_train), len(Y_test)
from collections import Counter
print(len(Y_train), len(Y_test))
C_train = Counter(Y_train)
C_test = Counter(Y_test)
print(C_train)
print(C_test)
print("%.2f"%(100*C_train['claim'] / sum(list(C_train.values()))), "%", "claims in train set!")
print("%.2f"%(100*C_test['claim'] / sum(list(C_test.values()))), "%", "claims in test set!")
X_train_embedds = model.encode(X_train)
len(X_train_embedds)
clf = SVC(kernel='rbf', gamma='auto')
print(dt.datetime.now())
clf.fit(X_train_embedds, Y_train)
print(dt.datetime.now())
X_test_embedds = model.encode(X_test)
len(X_test_embedds)
Y_pred = clf.predict(X_test_embedds)
len(Y_pred)
p,r,f1,s = precision_recall_fscore_support(y_true=Y_test, y_pred=Y_pred, average='macro', warn_for=tuple())
print("%.3f Precision\n%.3f Recall\n%.3f F1"%(p,r,f1))
print(classification_report(y_true=Y_test, y_pred=Y_pred))
for i, y_pred in enumerate(Y_pred):
    if y_pred=='claim':
        print(X_test[i])
        print()
    if i>=10:
        break
!pip install -U stanza
import stanza
import hashlib
stanza.download('en')
nlp = stanza.Pipeline(processors='tokenize', lang='en', use_gpu=True)
!git clone https://github.com/COVIEWED/coviewed_web_scraping
!pip install -r coviewed_web_scraping/requirements.txt
#EXAMPLE_URL = 'https://www.euronews.com/2020/04/01/the-best-way-prevent-future-pandemics-like-coronavirus-stop-eating-meat-and-go-vegan-view'
EXAMPLE_URL = 'https://edition.cnn.com/2020/03/04/health/debunking-coronavirus-myths-trnd/'
print(EXAMPLE_URL)
!cd coviewed_web_scraping/ && python3 src/scrape.py -u={EXAMPLE_URL}
txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
len(txt_files), txt_files
data_path = 'coviewed_web_scraping/data/'
fname = txt_files[0]
print(fname)
with open(os.path.join(data_path, fname), 'r') as my_file:
    txt_data = my_file.readlines()
txt_data = [line.strip() for line in txt_data if line.strip()]
len(txt_data)
article_url = txt_data[0]
print(article_url)
article_published_datetime = txt_data[1]
print(article_published_datetime)
article_title = txt_data[2]
print(article_title)
article_text = "\n\n".join(txt_data[3:])
print(article_text)
ALL_SENTENCES = []
txt = [p.strip() for p in article_text.split('\n') if p.strip()]
file_id = fname.split('.')[0]
print(file_id)
print()
for i, paragraph in enumerate(txt):
    doc = nlp(paragraph)
    for sent in doc.sentences:
        S = ' '.join([w.text for w in sent.words])
        sH = hashlib.md5(S.encode('utf-8')).hexdigest()
        print(sH)
        print(S)
        print()
        ALL_SENTENCES.append([file_id, sH, S])
fname = file_id+'_sentences.tsv'
print(fname)
AS = pd.DataFrame(ALL_SENTENCES, columns=['file_id','sentenceHash','sentence'])
len(AS)
sentences = [sentence for sentence in AS.sentence.values.tolist()]
len(sentences)
embedded_sentences = model.encode(sentences)
len(embedded_sentences)
predictions = clf.predict(embedded_sentences)
len(predictions)
len(predictions), len(sentences)
Counter(predictions)
for i, prediction in enumerate(predictions):
    if prediction=='claim':
        print(sentences[i])
        print()
