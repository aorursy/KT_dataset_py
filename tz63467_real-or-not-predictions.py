import re
import nltk
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.util import ngrams
stop = set(stopwords.words('english'))
# from google.colab import drive
# drive.mount('/gdrive')
# df_train = pd.read_csv('/gdrive/My Drive/nlp_getting_statred/train.csv')
# df_test = pd.read_csv('/gdrive/My Drive/nlp_getting_statred/test.csv')
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')
df_test = pd.read_csv('../input/nlp-getting-started/train.csv')
def dsply_all(df):
    with pd.option_context('display.max_rows', 5000, 'display.max_columns', 300):
        display(df)
dsply_all(df_train.head(3000))
df_train[df_train['target'] == 1].head()
df_train[df_train['target'] == 0].head()
# converting all the data into lowercase
df_train['text'] = df_train['text'].str.lower()
df_test['text'] = df_test['text'].str.lower()
df_train.tail()
df_test.tail()
# Removing noise before doing any processing to the data

def rmove_noise(df):
    #remove html markup
    df = re.sub("(<.*?>)", "", df)
    #remove non-ascii and digits
    df = re.sub("(\\W|\\d)", " ", df)
    #remove whitespace
    df = df.strip()
    return df
df_train['text'] = df_train['text'].apply(rmove_noise)
df_test['text'] = df_test['text'].apply(rmove_noise)
df_train
real_disater = df_train[df_train['target'] == 1].shape[0]
not_real = df_train[df_train['target'] == 0].shape[0]
# getting an estimate of the real disasters and unreal ones using a bar graph

plt.rcParams['figure.figsize'] = (7,5)
plt.bar(10, real_disater,3 ,  label = 'real_disaster', color = 'green')
plt.bar(15, not_real, 3,  label = 'not disaster', color = 'blue')
plt.legend()
plt.ylabel('Number of Examples')
plt.title('proportion of tweets')   
plt.show()
def mking_toekns(df):
     txt = "".join([c for c in df if c not in string.punctuation])
     tokens = re.split('\W+', df)
     return tokens
df_train['text'] = df_train['text'].apply(mking_toekns)
df_test['text'] = df_test['text'].apply(mking_toekns)
# Now replacing bbreviations, slangs and misspelled words for real meanings -- Normalization.
# following abbreviations are being used along with the nltk lemmatizer.
# I'm using this before stopwords because it fullforms comtain many stopwords.
# Normalization-- This tweets are not written in a gramatical fashion, So in order to understand that the shortcuts and abbreviatons have the
# same meaning.
abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",   
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"}

def convert_abbrev(word):
    return abbreviations[word.lower()] if word.lower()  in abbreviations.keys() else word

def convert_abbrev_in_text(text):
    tokens = [convert_abbrev(word) for word in text]
    # text = ' '.join(tokens)
    return text

df_train['text'] = df_train['text'].apply(convert_abbrev_in_text)
df_test['text'] = df_test['text'].apply(convert_abbrev_in_text)
# Here we will remove the stopwords, so that we it so that the data gets clean and lite.
def remove_stopwords(df):
    # txt = "".join([c for c in df if c not in string.punctuation])
    # tokens = re.split('\W+', df)
    txt = [w for w in df if w not in stop]
    return txt
# remove_stopwords = lambda x:[[w for w in word_tokenize(sent) if w not in stop] for sent in sent_tokenize(x)]

df_train['text'] = df_train['text'].apply(remove_stopwords)
df_test['text'] = df_test['text'].apply(remove_stopwords)
dsply_all(df_train['text'].head(3000))
# Using WordNetLemmatizer to lemmatize the words which have same meaning but
# different spellings, later in the section we will lemmatize some abbrs.

# words_lematize = lambda k:[w for w in lemmatizer.lemmatize(k)]
def words_lematize(df):
    wn = WordNetLemmatizer()
    text = [wn.lemmatize(w) for w in df]
    text = ' '.join(text)
    return text

df_train['text'] = df_train['text'].apply(words_lematize)
df_test['text'] = df_test['text'].apply(words_lematize)
dsply_all(df_train.head(2000))
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.initializers import Constant
import zipfile 
zip_ref = zipfile.ZipFile("/gdrive/My Drive/glove.6B.zip", 'r')
zip_ref.extractall("/tmp")
zip_ref.close()
def create_corpus(df):
    corpus=[]
    for tweet in tqdm(df['text']):
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]
        corpus.append(words)
    return corpus
corpus=create_corpus(df_train)
df_train
embedding_dict={}
with open('/tmp/glove.6B.100d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()

MAX_LEN=50
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index
print('Number of unique words:',len(word_index))
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))

for word,i in tqdm(word_index.items()):
    if i > num_words:
        continue
    
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec
model=Sequential()

embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)

model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


optimzer=Adam(learning_rate=1e-5)

model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
model.summary()
model.summary()
train=tweet_pad[:df_train.shape[0]]
test=tweet_pad[:df_test.shape[0]]
X_train,X_test,y_train,y_test=train_test_split(train,df_train['target'].values,test_size=0.2)
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_test.shape)
history=model.fit(X_train,y_train,batch_size=4,epochs=50,validation_data=(X_test,y_test),verbose=2)
# history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)
model.save('realornot2.hdf5')
train_pred = model.predict(X_train)
train_bool = np.argmax(train_pred, axis=1)
test_pred = model.predict(X_test)
test_bool = np.argmax(test_pred, axis=1)
from sklearn.metrics import classification_report
X_train_score = classification_report(y_train, train_bool)
X_test_score = classification_report(y_test,test_bool)
# print('F1 Score for train and test data resp.: {} & {}'.format(X_train_score,X_test_score))
print(X_train_score)
print(X_test_score)
y_pre=model.predict(test_trial)
y_pre=np.round(y_pre).astype(int).reshape(3263)
sub=pd.DataFrame({'id':df_test['id'].values.tolist(),'target':y_pre})
sub.to_csv('submission.csv',index=False)
y_pre.shape
df_test