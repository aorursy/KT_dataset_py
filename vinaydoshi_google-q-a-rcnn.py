import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

#from wordcloud import WordCloud, STOPWORDS

import pyprind 

import warnings

warnings.filterwarnings('ignore')

import gc

import re

import gensim

from gensim.models import KeyedVectors

import operator

import string

import tensorflow as tf

import json

from tqdm import tqdm, tqdm_notebook

tqdm_notebook().pandas()

from zipfile import ZipFile

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Add,Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Flatten,Dense, Embedding, Concatenate, Input, Dropout,Bidirectional, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, LeakyReLU, Activation, LSTM, SpatialDropout1D, BatchNormalization

#Dense, Embedding, Bidirectional, CuDNNGRU, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Input, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, BaseLogger

from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split

#from sklearn.metrics import precision_recall_fscore_support



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/google-quest-challenge/train.csv', header=0,encoding='utf-8')

test = pd.read_csv('../input/google-quest-challenge/test.csv', header=0,encoding='utf-8')

df = pd.concat([train,test],axis=0,ignore_index=True)

print(f'''Train Shape: {train.shape}

Test Shape: {test.shape}

Df Shape:{df.shape}''')
# df['treated_question_title'] = df['question_title'].progress_apply(lambda x: x.lower().replace('\n',' ').strip())

# df['treated_question_body'] = df['question_body'].progress_apply(lambda x: x.lower().replace('\n',' ').strip())

# df['treated_answer'] = df['answer'].progress_apply(lambda x: x.lower().replace('\n',' ').strip())
def separate_puncts(sentence, category):

    if category=='STACKOVERFLOW' or  category=='TECHNOLOGY':

        sentence = re.sub(r"(<.*?>)|(/*.*?\*/)|(for.*?\+\+)|(lt;.*?}})|(gt\s.*?}})|({\s.*?})|([&\s]lt;.*?gt[;\s])",' programming code ',sentence)

        sentence = re.sub(r"(programming code[\s\W]*)", ' programming code ',sentence)

    sentence = re.sub(r'http\S+', 'url', sentence)

    emojis=re.findall(r'\s(?::|;|=|\^)(?:-|\s)?(?:\)|\\|\/|\(|D|P|\^|\|)\s',sentence)

    for i in emojis:

        sentence = sentence.replace(i,'')

    sentence = re.sub(r"(\$\$.*?\$\$)", ' mathematical formula ', sentence)

    sentence = re.sub(r"((?<=\w)[^\s\w'?\-](?![^\s\w]))|([^\s\w'?\-](?![^\s\w])(?=\w))",' ',sentence)

    sentence = re.sub(r"((?<=\w)[.?\/\-])|([.?\/\-](?=\w))",' \g<1>\g<2> ',sentence)

    return ' '.join([sentence]+emojis)
def build_vocab(sentences, vocab={}, verbose=True):

    sentences = sentences.apply(lambda x: x.split()).values

    #vocab={}

    for sentence in pyprind.prog_bar(sentences):

        for word in sentence:

            try:

                vocab[word] +=1

            except KeyError:

                vocab[word] = 1

    return vocab
# def load_embed(file):

#     def get_coefs(word, *arr):

#         return word, np.array(arr, dtype='float16')

    

#     if file == '../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':

#         embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(file,encoding='utf-8') if len(o)>100)

    

#     elif file == '../input/quora-insincere-questions-classification/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin':

#         embeddings_index = KeyedVectors.load_word2vec_format(file, binary=True)

    

#     elif file == '../input/quora-insincere-questions-classification/embeddings.zip/paragram_300_sl999/paragram_300_sl999.txt':

#         embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin') if len(o)>100)

        

#     elif file == '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt':

#         embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

        

#     return embeddings_index







# glove = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'

# paragram =  '..//input/quora-insincere-questions-classification/embeddings.zip/paragram_300_sl999/paragram_300_sl999.txt'

# wiki_news = '../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

# google_embed = '../input/quora-insincere-questions-classification/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
def load_embed(file, name):

    def get_coefs(word, *arr):

        return word, np.array(arr, dtype='float16')

    

    if name == 'wiki':

        embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(file,encoding='utf-8') if len(o)>100)

    

    elif name == 'word2vec':

        embeddings_index = KeyedVectors.load_word2vec_format(file, binary=True)

    

    elif name == 'paragram':

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin') if len(o)>100)

        

    elif name == 'glove':

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

        

    return embeddings_index
# with ZipFile('../input/quora-insincere-questions-classification/embeddings.zip','r') as f:

#     f.printdir()

#     f.extract('paragram_300_sl999/paragram_300_sl999.txt')

#     print('Done')
# print("Extracting GloVe embedding")

# embed_glove = load_embed(glove)

print("Extracting Paragram embedding")

embed_paragram = load_embed('../input/paragram-300-sl999/paragram_300_sl999/paragram_300_sl999/paragram_300_sl999.txt','paragram')



# print("Extracting Wiki embedding")

# embed_wiki = load_embed(wiki_news)

# print("Extracting Google embedding")

# embed_google = load_embed(google_embed)

print('Done')
len(embed_paragram)
# os.remove('/kaggle/working/paragram_300_sl999/paragram_300_sl999.txt')
def check_coverage(vocab,embeddings_index):

    known_words={}

    unknown_words={}

    nb_known_words=0

    nb_unknown_words=0

    for word in vocab.keys():

        if word in embeddings_index:

            known_words[word]=embeddings_index[word]

            nb_known_words += vocab[word]

        elif word.capitalize() in embeddings_index:

            known_words[word] = embeddings_index[word.capitalize()]

            nb_known_words += vocab[word]

        elif word.lower() in embeddings_index:

            known_words[word] = embeddings_index[word.lower()]

            nb_known_words += vocab[word]

        elif word.upper() in embeddings_index:

            known_words[word] = embeddings_index[word.upper()]

            nb_known_words += vocab[word]

        else:

            unknown_words[word] = vocab[word]

            nb_unknown_words += vocab[word]

    print(f'Found embeddings for {round((len(known_words)/len(vocab))*100,5)}% of the vocab\nFound embeddings for {round((nb_known_words/(nb_known_words+nb_unknown_words))*100,5)}% of all text')

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1),reverse=True)#[::-1]

    return unknown_words
# len(vocab),sorted(vocab.items(),key = operator.itemgetter(1),reverse=True)
contraction_mapping = {"\'":"'", #,"..":'',

    "Trump's" : 'trump is',"'cause": 'because',"â€™": "'",",cause": 'because',";cause": 'because',"ain't": 'am not',"ain,t": 'am not',

    "ain;t": 'am not',"ain't": 'am not',"ain’t": 'am not',"aren't": 'are not',"â€“": '-',"â€œ":'"',

    "aren,t": 'are not',"aren;t": 'are not',"aren't": 'are not',"aren’t": 'are not',"can't": 'cannot',"can't've": 'cannot have',"can,t": 'cannot',"can,t,ve": 'cannot have',

    "can;t": 'cannot',"can;t;ve": 'cannot have',

    "can't": 'cannot',"can't´ve": 'cannot have',"can’t": 'cannot',"can’t’ve": 'cannot have',

    "could've": 'could have',"could,ve": 'could have',"could;ve": 'could have',"couldn't": 'could not',"couldn't've": 'could not have',"couldn,t": 'could not',"couldn,t,ve": 'could not have',"couldn;t": 'could not',

    "couldn;t;ve": 'could not have',"couldn't": 'could not',

    "couldn't´ve": 'could not have',"couldn’t": 'could not',"couldn’t’ve": 'could not have',"could´ve": 'could have',

    "could’ve": 'could have',"didn't": 'did not',"didn,t": 'did not',"didn;t": 'did not',"didn't": 'did not',

    "didn’t": 'did not',"doesn't": 'does not',"doesn,t": 'does not',"doesn;t": 'does not',"doesn't": 'does not',

    "doesn’t": 'does not',"don't": 'do not',"don,t": 'do not',"don;t": 'do not',"don't": 'do not',"don’t": 'do not',

    "hadn't": 'had not',"hadn't've": 'had not have',"hadn,t": 'had not',"hadn,t,ve": 'had not have',"hadn;t": 'had not',

    "hadn;t;ve": 'had not have',"hadn't": 'had not',"hadn't´ve": 'had not have',"hadn’t": 'had not',"hadn’t’ve": 'had not have',"hasn't": 'has not',"hasn,t": 'has not',"hasn;t": 'has not',"hasn't": 'has not',"hasn’t": 'has not',

    "haven't": 'have not',"haven,t": 'have not',"haven;t": 'have not',"haven't": 'have not',"haven’t": 'have not',"he'd": 'he would',

    "he'd've": 'he would have',"he'll": 'he will',

    "he's": 'he is',"he,d": 'he would',"he,d,ve": 'he would have',"he,ll": 'he will',"he,s": 'he is',"he;d": 'he would',

    "he;d;ve": 'he would have',"he;ll": 'he will',"he;s": 'he is',"he'd": 'he would',"he'd've": 'he would have',"he´ll": 'he will',

    "he´s": 'he is',"he’d": 'he would',"he’d’ve": 'he would have',"he’ll": 'he will',"he’s": 'he is',"how'd": 'how did',"how'll": 'how will',

    "how's": 'how is',"how,d": 'how did',"how,ll": 'how will',"how,s": 'how is',"how;d": 'how did',"how;ll": 'how will',

    "how;s": 'how is',"how´d": 'how did',"how´ll": 'how will',"how´s": 'how is',"how’d": 'how did',"how’ll": 'how will',

    "how’s": 'how is',"i'd": 'i would',"i'll": 'i will',"i'm": 'i am',"i've": 'i have',"i,d": 'i would',"i,ll": 'i will',

    "i,m": 'i am',"i,ve": 'i have',"i;d": 'i would',"i;ll": 'i will',"i;m": 'i am',"i;ve": 'i have',"isn't": 'is not',

    "isn,t": 'is not',"isn;t": 'is not',"isn't": 'is not',"isn’t": 'is not',"it'd": 'it would',"it'll": 'it will',"It's":'it is',

    "it's": 'it is',"it,d": 'it would',"it,ll": 'it will',"it,s": 'it is',"it;d": 'it would',"it;ll": 'it will',"it;s": 'it is',"it´d": 'it would',"it´ll": 'it will',"it´s": 'it is',

    "it’d": 'it would',"it’ll": 'it will',"it’s": 'it is',

    "i´d": 'i would',"i´ll": 'i will',"i´m": 'i am',"i´ve": 'i have',"i’d": 'i would',"i’ll": 'i will',"i’m": 'i am',

    "i’ve": 'i have',"let's": 'let us',"let,s": 'let us',"let;s": 'let us',"let´s": 'let us',

    "let’s": 'let us',"ma'am": 'madam',"ma,am": 'madam',"ma;am": 'madam',"mayn't": 'may not',"mayn,t": 'may not',"mayn;t": 'may not',

    "mayn't": 'may not',"mayn’t": 'may not',"ma´am": 'madam',"ma’am": 'madam',"might've": 'might have',"might,ve": 'might have',"might;ve": 'might have',"mightn't": 'might not',"mightn,t": 'might not',"mightn;t": 'might not',"mightn't": 'might not',

    "mightn’t": 'might not',"might´ve": 'might have',"might’ve": 'might have',"must've": 'must have',"must,ve": 'must have',"must;ve": 'must have',

    "mustn't": 'must not',"mustn,t": 'must not',"mustn;t": 'must not',"mustn't": 'must not',"mustn’t": 'must not',"must´ve": 'must have',

    "must’ve": 'must have',"needn't": 'need not',"needn,t": 'need not',"needn;t": 'need not',"needn't": 'need not',"needn’t": 'need not',"oughtn't": 'ought not',"oughtn,t": 'ought not',"oughtn;t": 'ought not',

    "oughtn't": 'ought not',"oughtn’t": 'ought not',"sha'n't": 'shall not',"sha,n,t": 'shall not',"sha;n;t": 'shall not',"shan't": 'shall not',

    "shan,t": 'shall not',"shan;t": 'shall not',"shan't": 'shall not',"shan’t": 'shall not',"sha´n't": 'shall not',"sha’n’t": 'shall not',

    "she'd": 'she would',"she'll": 'she will',"she's": 'she is',"she,d": 'she would',"she,ll": 'she will',

    "she,s": 'she is',"she;d": 'she would',"she;ll": 'she will',"she;s": 'she is',"she´d": 'she would',"she´ll": 'she will',

    "she´s": 'she is',"she’d": 'she would',"she’ll": 'she will',"she’s": 'she is',"should've": 'should have',"should,ve": 'should have',"should;ve": 'should have',

    "shouldn't": 'should not',"shouldn,t": 'should not',"shouldn;t": 'should not',"shouldn't": 'should not',"shouldn’t": 'should not',"should´ve": 'should have',

    "should’ve": 'should have',"that'd": 'that would',"that's": 'that is',"that,d": 'that would',"that,s": 'that is',"that;d": 'that would',

    "that;s": 'that is',"that´d": 'that would',"that´s": 'that is',"that’d": 'that would',"that’s": 'that is',"there'd": 'there had',

    "there's": 'there is',"there,d": 'there had',"there,s": 'there is',"there;d": 'there had',"there;s": 'there is',

    "there´d": 'there had',"there´s": 'there is',"there’d": 'there had',"there’s": 'there is',

    "they'd": 'they would',"they'll": 'they will',"they're": 'they are',"they've": 'they have',

    "they,d": 'they would',"they,ll": 'they will',"they,re": 'they are',"they,ve": 'they have',"they;d": 'they would',"they;ll": 'they will',"they;re": 'they are',

    "they;ve": 'they have',"they´d": 'they would',"they´ll": 'they will',"they´re": 'they are',"they´ve": 'they have',"they’d": 'they would',"they’ll": 'they will',

    "they’re": 'they are',"they’ve": 'they have',"wasn't": 'was not',"wasn,t": 'was not',"wasn;t": 'was not',"wasn't": 'was not',

    "wasn’t": 'was not',"we'd": 'we would',"we'll": 'we will',"we're": 'we are',"we've": 'we have',"we,d": 'we would',"we,ll": 'we will',

    "we,re": 'we are',"we,ve": 'we have',"we;d": 'we would',"we;ll": 'we will',"we;re": 'we are',"we;ve": 'we have',

    "weren't": 'were not',"weren,t": 'were not',"weren;t": 'were not',"weren't": 'were not',"weren’t": 'were not',"we´d": 'we would',"we´ll": 'we will',

    "we´re": 'we are',"we´ve": 'we have',"we’d": 'we would',"we’ll": 'we will',"we’re": 'we are',"we’ve": 'we have',"what'll": 'what will',"what're": 'what are',"what's": 'what is',

    "what've": 'what have',"what,ll": 'what will',"what,re": 'what are',"what,s": 'what is',"what,ve": 'what have',"what;ll": 'what will',"what;re": 'what are',

    "what;s": 'what is',"what;ve": 'what have',"what´ll": 'what will',

    "what´re": 'what are',"what´s": 'what is',"what´ve": 'what have',"what’ll": 'what will',"what’re": 'what are',"what’s": 'what is',

    "what’ve": 'what have',"where'd": 'where did',"where's": 'where is',"where,d": 'where did',"where,s": 'where is',"where;d": 'where did',

    "where;s": 'where is',"where´d": 'where did',"where´s": 'where is',"where’d": 'where did',"where’s": 'where is',

    "who'll": 'who will',"who's": 'who is',"who,ll": 'who will',"who,s": 'who is',"who;ll": 'who will',"who;s": 'who is',

    "who´ll": 'who will',"who´s": 'who is',"who’ll": 'who will',"who’s": 'who is',"won't": 'will not',"won,t": 'will not',"won;t": 'will not',

    "won't": 'will not',"won't": 'will not',"wouldn't": 'would not',"wouldn,t": 'would not',"wouldn;t": 'would not',"wouldn't": 'would not',

    "wouldn't": 'would not',"you'd": 'you would',"you'll": 'you will',"you're": 'you are',"you,d": 'you would',"you,ll": 'you will',

    "you,re": 'you are',"you;d": 'you would',"you;ll": 'you will',

    "you;re": 'you are',"you´d": 'you would',"you´ll": 'you will',"you´re": 'you are',"you’d": 'you would',"you’ll": 'you will',"you’re": 'you are',

    "´cause": 'because',"’cause": 'because',"you've": "you have","could'nt": 'could not',

    "havn't": 'have not',"here’s": "here is","i'm": 'i am' ,"i'l": "i will","i'v": 'i have',"wan't": 'want',"was'nt": "was not","who'd": "who would",

    "who're": "who are","who've": "who have","why'd": "why would","would've": "would have","y'all": "you all","y'know": "you know","you.i": "you i",

    "your'e": "you are","arn't": "are not","agains't": "against","c'mon": "common","doens't": "does not","don't": "do not","dosen't": "does not",

    "dosn't": "does not","shoudn't": "should not","that'll": "that will","there'll": "there will","there're": "there are",

    "this'll": "this all","u're": "you are", "ya'll": "you all","you'r": "you are","you’ve": "you have","d'int": "did not","did'nt": "did not","din't": "did not","dont't": "do not","gov't": "government",

    "i'ma": "i am","is'nt": "is not","‘I":'I',"ᴀɴᴅ":'and',"ᴛʜᴇ":'the',"ʜᴏᴍᴇ":'home',"ᴜᴘ":'up',"ʙʏ":'by',"ᴀᴛ":'at',"…and":'and',"civilbeat":'civil beat',"TrumpCare":'Trump care',"Trumpcare":'Trump care', "OBAMAcare":'Obama care',"ᴄʜᴇᴄᴋ":'check',"ғᴏʀ":'for',"ᴛʜɪs":'this',"ᴄᴏᴍᴘᴜᴛᴇʀ":'computer',"ᴍᴏɴᴛʜ":'month',"ᴡᴏʀᴋɪɴɢ":'working',"ᴊᴏʙ":'job',"ғʀᴏᴍ":'from',"Sᴛᴀʀᴛ":'start',"gubmit":'submit',"CO₂":'carbon dioxide',"ғɪʀsᴛ":'first',"ᴇɴᴅ":'end',"ᴄᴀɴ":'can',"ʜᴀᴠᴇ":'have',"ᴛᴏ":'to',"ʟɪɴᴋ":'link',"ᴏғ":'of',"ʜᴏᴜʀʟʏ":'hourly',"ᴡᴇᴇᴋ":'week',"ᴇɴᴅ":'end',"ᴇxᴛʀᴀ":'extra',"Gʀᴇᴀᴛ":'great',"sᴛᴜᴅᴇɴᴛs":'student',"sᴛᴀʏ":'stay',"ᴍᴏᴍs":'mother',"ᴏʀ":'or',"ᴀɴʏᴏɴᴇ":'anyone',"ɴᴇᴇᴅɪɴɢ":'needing',"ᴀɴ":'an',"ɪɴᴄᴏᴍᴇ":'income',

    "ʀᴇʟɪᴀʙʟᴇ":'reliable',"ғɪʀsᴛ":'first',"ʏᴏᴜʀ":'your',"sɪɢɴɪɴɢ":'signing',"ʙᴏᴛᴛᴏᴍ":'bottom',"ғᴏʟʟᴏᴡɪɴɢ":'following',"Mᴀᴋᴇ":'make',

    "ᴄᴏɴɴᴇᴄᴛɪᴏɴ":'connection',"ɪɴᴛᴇʀɴᴇᴛ":'internet',"financialpost":'financial post', "ʜaᴠᴇ":' have ', "ᴄaɴ":' can ', "Maᴋᴇ":' make ', "ʀᴇʟɪaʙʟᴇ":' reliable ', "ɴᴇᴇᴅ":' need ',

    "ᴏɴʟʏ":' only ', "ᴇxᴛʀa":' extra ', "aɴ":' an ', "aɴʏᴏɴᴇ":' anyone ', "sᴛaʏ":' stay ', "Sᴛaʀᴛ":' start', "SHOPO":'shop',

    " :-/ ":'Perplexed smilee'," :/ ":'Perplexed smilee'," ;-/ ":'Perplexed'," ;/ ":'Perplexed', " ;/ ":'Perplexed', " :/ ":'Perplexed',

    " =( ":'Sad'," :-( ":'Sad'," :( ":'Sad', "=gt":'=>', "=gt":'=>', "n't":" not", "'s": " is", 'amp;amp':'ampersand', '..':'', '////':'', 'nbsp;':'',

    '\\\\':'', "usepackage" : "use package",'instrumentsettingsid':'instrumental settings id','rippleshaderProgram' : 'ripple shader program','shaderprogramconstants':'shader program constants','storedElements':'stored elements','stackSize' : 'stack size'                   

    }
#bar = pyprind.ProgBar(df.shape[0], bar_char='█')

def clean_contractions(text, mapping):

    #text = text.split()

    #text_val = text.lower()

    #map_val = 

    for word in mapping.keys():

        word_val = word.lower()

        if word in text:

            text = text.replace(word, mapping[word])

        elif word_val in text:

            text = text.replace(word_val, mapping[word])

#         elif word.lower() in text:

#             text = text.replace(word, mapping[word])

#         elif word.capitalize() in text:

#             text = text.replace(word, mapping[word])

#         elif word.upper() in text:

#             text = text.replace(word, mapping[word])



#     bar.update()

    return text
# df['treated_question_title'] = df['treated_question_title'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))

# df['treated_question_body'] = df['treated_question_body'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))

# df['treated_answer'] = df['treated_answer'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))
punct = "-?!.,#$%\*+<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

punct_mapping = {"‘": "'", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': ' divided by ', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
len(embed_paragram['smilee'])
def unknown_punct(embed, punct):

    unknown_punct=''

    for val in punct:

        if val not in embed:

            unknown_punct += val+' '

    return unknown_punct

# print(f'Glove:\n{unknown_punct(embed_glove, punct)}')

# print(f'Paragram:\n{unknown_punct(embed_paragram, punct)}')

# print(f'Wiki:\n{unknown_punct(embed_wiki, punct)}')

# print(f'Google:\n{unknown_punct(embed_google, punct)}')
def clean_puncts(text, mapping, punct):

    for p in punct_mapping.keys():

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

#     for word in text:

#         for p in '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~':

#             word = word.replace(p, f' {p} ')

    

#     specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}

#     for p in specials.keys():

#         text = text.replace(p, specials[p])

#    bar.update()    

    return text



# df['treated_question_title'] = df['treated_question_title'].progress_apply(lambda x: clean_puncts(x, punct_mapping, punct))

# df['treated_question_body'] = df['treated_question_body'].progress_apply(lambda x: clean_puncts(x, punct_mapping, punct))

# df['treated_answer'] = df['treated_answer'].progress_apply(lambda x: clean_puncts(x, punct_mapping, punct))



df['treated_question_title'] = df['question_title'].apply(lambda x: x.lower().replace('\n',' ').strip())

df['treated_question_body'] = df['question_body'].apply(lambda x: x.lower().replace('\n',' ').strip())

df['treated_answer'] = df['answer'].apply(lambda x: x.lower().replace('\n',' ').strip())

print('Stripping Done')



df['treated_question_title'] = df[['treated_question_title','category']].progress_apply(lambda x: separate_puncts(x['treated_question_title'], x['category']),axis=1)

df['treated_question_body'] = df[['treated_question_body','category']].progress_apply(lambda x: separate_puncts(x['treated_question_body'], x['category']),axis=1)

df['treated_answer'] = df[['treated_answer','category']].progress_apply(lambda x: separate_puncts(x['treated_answer'], x['category']),axis=1)

print('Cleaning Data Done')



df['treated_question_title'] = df['treated_question_title'].apply(lambda x: clean_contractions(x, contraction_mapping))

df['treated_question_body'] = df['treated_question_body'].apply(lambda x: clean_contractions(x, contraction_mapping))

df['treated_answer'] = df['treated_answer'].apply(lambda x: clean_contractions(x, contraction_mapping))

print('Contraction Mapping Done')



df['treated_question_title'] = df['treated_question_title'].apply(lambda x: clean_puncts(x, punct_mapping, punct))

df['treated_question_body'] = df['treated_question_body'].apply(lambda x: clean_puncts(x, punct_mapping, punct))

df['treated_answer'] = df['treated_answer'].apply(lambda x: clean_puncts(x, punct_mapping, punct))

print('Clean Punctuation Done')
vocab = build_vocab(df['treated_question_title'])

print(len(vocab))

vocab = build_vocab(df['treated_question_body'], vocab=vocab)

print(len(vocab))

vocab = build_vocab(df['treated_answer'], vocab=vocab)

print(len(vocab))

# print("Glove : ")

# oov_glove = check_coverage(vocab, embed_glove)

print("Paragram : ")

oov_paragram = check_coverage(vocab, embed_paragram)



# print("Wiki : ")

# oov_glove = check_coverage(vocab, embed_wiki)

# print("Google : ")

# oov_paragram = check_coverage(vocab, embed_google)
import traceback

traceback.extract_stack()[-2]
find = re.compile(r"^[^.]*")

df['web_category'] = df['url'].apply(lambda x: re.findall(find, x)[0].replace('http://',''))

df[['url', 'web_category']]
df.groupby(['category','web_category']).size()
for i in df['category'].unique():

    print(i, len(df[df['category']==i]['web_category'].unique()))

    
dums = pd.get_dummies(df[['category','web_category']])

df = pd.concat([df,dums], axis=1)
# df.loc[500, ['category','web_category','category_STACKOVERFLOW','web_category_stackoverflow']]



df.drop(columns=['question_title','question_body','answer', 'category','web_category'],inplace=True)
X_train = df[df['answer_helpful'].notna()]

X_test = df[df['answer_helpful'].isna()]
title_max, title_mean = df['treated_question_title'].apply(lambda x: x.split()).map(len).max(), df['treated_question_title'].apply(lambda x: x.split()).map(len).mean()

body_max, body_mean = df['treated_question_body'].apply(lambda x: x.split()).map(len).max(), df['treated_question_body'].apply(lambda x: x.split()).map(len).mean() 

ans_max, ans_mean = df['treated_answer'].apply(lambda x: x.split()).map(len).max(), df['treated_answer'].apply(lambda x: x.split()).map(len).mean() 

print(f'Df Max Title Size is:\t{title_max}\twith average size being:\t{title_mean}')

print(f'Df Max Body Size is:\t{body_max}\twith average size being:\t{body_mean}')

print(f'Df Max Ans Size is:\t{ans_max}\twith average size being:\t{ans_mean}')
title_max, title_mean = X_train['treated_question_title'].apply(lambda x: x.split()).map(len).max(), X_train['treated_question_title'].apply(lambda x: x.split()).map(len).mean()

body_max, body_mean = X_train['treated_question_body'].apply(lambda x: x.split()).map(len).max(), X_train['treated_question_body'].apply(lambda x: x.split()).map(len).mean() 

ans_max, ans_mean = X_train['treated_answer'].apply(lambda x: x.split()).map(len).max(), X_train['treated_answer'].apply(lambda x: x.split()).map(len).mean() 

print(f'X_train Max Title Size is:\t{title_max}\twith average size being:\t{title_mean}')

print(f'X_train Max Body Size is:\t{body_max}\twith average size being:\t{body_mean}')

print(f'X_train Max Ans Size is:\t{ans_max}\twith average size being:\t{ans_mean}')
title_max, title_mean = X_test['treated_question_title'].apply(lambda x: x.split()).map(len).max(), X_test['treated_question_title'].apply(lambda x: x.split()).map(len).mean()

body_max, body_mean = X_test['treated_question_body'].apply(lambda x: x.split()).map(len).max(), X_test['treated_question_body'].apply(lambda x: x.split()).map(len).mean() 

ans_max, ans_mean = X_test['treated_answer'].apply(lambda x: x.split()).map(len).max(), X_test['treated_answer'].apply(lambda x: x.split()).map(len).mean() 

print(f'X_test Max Title Size is:\t{title_max}\twith average size being:\t{title_mean}')

print(f'X_test Max Body Size is:\t{body_max}\twith average size being:\t{body_mean}')

print(f'X_test Max Answ Size is:\t{ans_max}\twith average size being:\t{ans_mean}')
maxlen = 250

length_to_keep = 1000000
def length_maker(text, maxlen=maxlen):

    text = text.split()

    if len(text)>220:

        text = text[:125]+text[-125:]

    return ' '.join(text)
df['treated_question_title'] = df['treated_question_title'].progress_apply(lambda x: length_maker(x))

df['treated_question_body'] = df['treated_question_body'].progress_apply(lambda x: length_maker(x))

df['treated_answer'] = df['treated_answer'].progress_apply(lambda x: length_maker(x))
X_train['treated_question_title'] = X_train['treated_question_title'].progress_apply(lambda x: length_maker(x))

X_train['treated_question_body'] = X_train['treated_question_body'].progress_apply(lambda x: length_maker(x))

X_train['treated_answer'] = X_train['treated_answer'].progress_apply(lambda x: length_maker(x))
X_test['treated_question_title'] = X_test['treated_question_title'].apply(lambda x: length_maker(x))

X_test['treated_question_body'] = X_test['treated_question_body'].apply(lambda x: length_maker(x))

X_test['treated_answer'] = X_test['treated_answer'].apply(lambda x: length_maker(x))
title_max, title_mean = X_test['treated_question_title'].apply(lambda x: x.split()).map(len).max(), X_test['treated_question_title'].apply(lambda x: x.split()).map(len).mean()

body_max, body_mean = X_test['treated_question_body'].apply(lambda x: x.split()).map(len).max(), X_test['treated_question_body'].apply(lambda x: x.split()).map(len).mean() 

ans_max, ans_mean = X_test['treated_answer'].apply(lambda x: x.split()).map(len).max(), X_test['treated_answer'].apply(lambda x: x.split()).map(len).mean() 

print(f'X_test Max Title Size is:\t{title_max}\twith average size being:\t{title_mean}')

print(f'X_test Max Body Size is:\t{body_max}\twith average size being:\t{body_mean}')

print(f'X_test Max Answ Size is:\t{ans_max}\twith average size being:\t{ans_mean}')
cnt=0

cat_cols=[]

for i in X_train.columns:

    

    if i.startswith('category') or i.startswith('web_category'):

        cnt+=1

        cat_cols.append(i)

print(cat_cols, len(cat_cols))
targets = [

        'question_asker_intent_understanding',

        'question_body_critical',

        'question_conversational',

        'question_expect_short_answer',

        'question_fact_seeking',

        'question_has_commonly_accepted_answer',

        'question_interestingness_others',

        'question_interestingness_self',

        'question_multi_intent',

        'question_not_really_a_question',

        'question_opinion_seeking',

        'question_type_choice',

        'question_type_compare',

        'question_type_consequence',

        'question_type_definition',

        'question_type_entity',

        'question_type_instructions',

        'question_type_procedure',

        'question_type_reason_explanation',

        'question_type_spelling',

        'question_well_written',

        'answer_helpful',

        'answer_level_of_information',

        'answer_plausible',

        'answer_relevance',

        'answer_satisfaction',

        'answer_type_instructions',

        'answer_type_procedure',

        'answer_type_reason_explanation',

        'answer_well_written'    

    ]
y_train = X_train[targets]

X_train.drop(columns=targets+['answer_user_name', 'answer_user_page', 'host', 'qa_id', 'question_user_name', 'question_user_page', 'url'], inplace=True)


y_test = X_test[targets]

X_test.drop(columns=targets+['answer_user_name', 'answer_user_page', 'host', 'qa_id', 'question_user_name', 'question_user_page', 'url'], inplace=True)
X_test.shape, X_train.shape
X_train.columns
all_text = pd.concat([X_train['treated_question_body'],X_train['treated_answer'],X_test['treated_question_body'],X_test['treated_answer'],X_train["treated_question_title"],X_test["treated_question_title"]])
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=train['category'])
# def tokenize_treated_data(text, text_test, maxlen=maxlen):

#     t=Tokenizer(num_words=length_to_keep, filters=' ')

#     t.fit_on_texts(text)

#     text = t.texts_to_sequences(text)

#     text_test = t.texts_to_sequences(text_test)

#     text = pad_sequences(text, maxlen=maxlen)

#     text_test = pad_sequences(text_test, maxlen=maxlen)

#     return text, t.word_index, text_test

t=Tokenizer(num_words=length_to_keep, filters=' ')

t.fit_on_texts(all_text)

def tokenize_treated_data(text, text_test, maxlen=maxlen, t=t):

#     t=Tokenizer(num_words=length_to_keep, filters=' ')

#     t.fit_on_texts(text)

    text = t.texts_to_sequences(text)

    text_test = t.texts_to_sequences(text_test)

    text = pad_sequences(text, maxlen=maxlen)

    text_test = pad_sequences(text_test, maxlen=maxlen)

    return text, t.word_index, text_test
train_question_body, word_index, test_question_body = tokenize_treated_data(X_tr['treated_question_body'],

                                                                                          X_test['treated_question_body']

                                                                                         )

_, _, val_question_body = tokenize_treated_data(X_tr['treated_question_body'],

                                                                                          X_val['treated_question_body']

                                                                                         )
train_answer, _, test_answer = tokenize_treated_data(X_tr['treated_answer'],

                                                                                          X_test['treated_answer']

                                                                                         )



_, _, val_answer = tokenize_treated_data(X_train['treated_answer'],

                                                                                          X_val['treated_answer']

                                                                                         )
train_question_title, _, test_question_title = tokenize_treated_data(X_tr['treated_question_title'],

                                                                                          X_test['treated_question_title'], maxlen=X_tr['treated_question_title'].apply(lambda x: x.split()).map(len).max()

                                                                                         )



_, _, val_question_title = tokenize_treated_data(X_tr['treated_question_title'],

                                                                                          X_val['treated_question_title'], maxlen=X_tr['treated_question_title'].apply(lambda x: x.split()).map(len).max()

                                                                                         )
def embed_matrix(embed_paragram, word_index, length_to_keep = length_to_keep):

    embeddings = np.stack(embed_paragram.values())

    

#    embeddings_mean, embeddings_std = embeddings.mean(), embeddings.std(ddof=1)

#    print(embeddings_mean)

     

    embeddings_shape = embeddings.shape[1]

    embedding_matrix = np.zeros((length_to_keep, 300))

    

    for word, i in word_index.items():

        if i >= length_to_keep:

            continue

        embeddings_vector = embed_paragram.get(word)

        if embeddings_vector is not None:

            embedding_matrix[i] = embeddings_vector

    return embedding_matrix
# embedding_question_body = embed_matrix(embed_paragram, question_body_word_index, length_to_keep)

# embedding_question_title = embed_matrix(embed_paragram, question_title_word_index, length_to_keep)

# embedding_answer = embed_matrix(embed_paragram, answer_word_index, length_to_keep)
embedding_mat = embed_matrix(embed_paragram, word_index, length_to_keep)
del embed_paragram

gc.collect()
# # del vocab, embed_paragram, oov_paragram, df

# del df

# gc.collect()
# embedding_matrices = {'answer':embedding_answer, 'body':embedding_question_body, 'title':embedding_question_title}
#gbdgdfbdbdxvbdfhdf





#fghfffffff
# def make_model(embedding_matrices=embedding_matrices, embed_size=300, loss='binary_crossentropy'):

#     input_title = Input(shape=(X_tr['treated_question_title'].apply(lambda x: x.split()).map(len).max(),), name='title')

#     input_body = Input(shape=(maxlen,), name='body')

#     input_answer = Input(shape=(maxlen,), name='answer')

#     input_category = Input(shape=(len(cat_cols),),name='categorical')

#     GRU_nums = 64

#     filter_size=[3,4,5]

#     num_filters=128

#     ###  Answer  ##########

#     x = Embedding(length_to_keep, embed_size, weights=[embedding_matrices['answer']],trainable=False)(input_answer)

#     ans_lstm = Bidirectional(LSTM(GRU_nums, return_sequences=True), merge_mode='sum')(x)

# #     x = Bidirectional(LSTM(GRU_nums, return_sequences=True))(x)

#     x = Concatenate(axis=2)([ans_lstm, x])

# #     x = Bidirectional(LSTM(GRU_nums, return_sequences=True),merge_mode='sum')(x)

#     conv2 = Conv1D(kernel_size=filter_size[0], strides=1, filters=num_filters, padding='same', name='ans_conv1')(x)

#     conv2 = LeakyReLU()(conv2)

# #     conv2 = BatchNormalization()(conv2)

#     conv3 = Conv1D(kernel_size=filter_size[1], strides=1, filters=num_filters, padding='same')(x)

#     conv3 = LeakyReLU()(conv3)

# #     conv3 = BatchNormalization()(conv3)

# #     conv4 = Conv1D(kernel_size=filter_size[2], strides=1, filters=num_filters, padding='same')(x)

# #     conv4 = LeakyReLU()(conv4)



#     conv2_max = GlobalMaxPooling1D()(conv2)

#     conv3_max = GlobalMaxPooling1D()(conv3)

    

#     conv2_avg = GlobalAveragePooling1D()(conv2)

#     conv3_avg = GlobalAveragePooling1D()(conv3)

    

# #     conv4 = GlobalMaxPooling1D()(conv4)

#     conc_answer_maxp = Concatenate(axis=1)([conv2_max, conv3_max])#, conv4])

#     conc_answer_avgp = Concatenate(axis=1)([conv2_avg, conv3_avg])

    

#     conc_answer = Concatenate(axis=1)([conc_answer_maxp, conc_answer_avgp])#, conv4])

    

#     Answer = Model(inputs=input_answer, outputs = conc_answer)

    

#     ###  Body  ##########

#     y = Embedding(length_to_keep, embed_size, weights=[embedding_matrices['body']],trainable=False)(input_body)

#     body_lstm = Bidirectional(LSTM(GRU_nums, return_sequences=True), merge_mode='sum')(y)

# #     y = Bidirectional(LSTM(GRU_nums, return_sequences=True))(y)

# #     y = Bidirectional(LSTM(GRU_nums, return_sequences=True),merge_mode='sum')(y)

    

#     y = Concatenate(axis=2)([body_lstm, y])

    

#     conv2_body = Conv1D(kernel_size=filter_size[0], strides=1, filters=num_filters, padding='same', name='body_conv1')(y)

#     conv2_body = LeakyReLU()(conv2_body)

# #     conv2_body = BatchNormalization()(conv2_body)

#     conv3_body = Conv1D(kernel_size=filter_size[1], strides=1, filters=num_filters, padding='same')(y)

#     conv3_body = LeakyReLU()(conv3_body)

# #     conv3_body = BatchNormalization()(conv3_body)

# #     conv4_body = Conv1D(kernel_size=filter_size[2], strides=1, filters=num_filters, padding='same')(y)

# #     conv4_body = LeakyReLU()(conv4_body)



#     conv2_body_max = GlobalMaxPooling1D()(conv2_body)

#     conv3_body_max = GlobalMaxPooling1D()(conv3_body)

    

#     conv2_body_avg = GlobalAveragePooling1D()(conv2_body)

#     conv3_body_avg = GlobalAveragePooling1D()(conv3_body)

# #     conv4_body = GlobalMaxPooling1D()(conv4_body)

    

#     conc_body_maxp = Concatenate(axis=1)([conv2_body_max, conv3_body_max])#, conv4_body])

#     conc_body_avgp = Concatenate(axis=1)([conv2_body_avg, conv3_body_avg])

#     conc_body = Concatenate(axis=1)([conc_body_maxp, conc_body_avgp])

#     Body = Model(inputs=input_body, outputs = conc_body)

    

    

#     ### Title  ###########

#     z = Embedding(length_to_keep, embed_size, weights=[embedding_matrices['title']],trainable=False)(input_title)

#     title_lstm = Bidirectional(LSTM(GRU_nums, return_sequences=True), merge_mode='sum')(z)

    

#     z = Concatenate(axis=2)([title_lstm, z])

    

#     conv2_title = Conv1D(kernel_size=filter_size[0], strides=1, filters=num_filters, padding='same')(z)

#     conv2_title = LeakyReLU()(conv2_title)

# #     conv2_title = BatchNormalization()(conv2_title)

# #     conv3_title = Conv1D(kernel_size=filter_size[1], strides=1, filters=num_filters, padding='same')(z)

# #     conv3_title = LeakyReLU()(conv3_title)

# #     conv4_title = Conv1D(kernel_size=filter_size[2], strides=1, filters=num_filters, padding='same')(z)

# #     conv4_title = LeakyReLU()(conv4_title)

    

#     conv2_title_maxp = GlobalMaxPooling1D()(conv2_title)

#     conv2_title_avgp = GlobalAveragePooling1D()(conv2_title)

# #     conv3_title = GlobalMaxPooling1D()(conv3_title)

# #     conv4_title = GlobalMaxPooling1D()(conv4_title)

    

#     conc_title = Concatenate(axis=1)([conv2_title_maxp, conv2_title_avgp])

    

#     Title = Model(inputs=input_title, outputs = conc_title)

    

#     ######################################

#     combined = Concatenate(axis=1)([Title.output, Body.output, Answer.output, input_category])

    

# #     dense1 = Dense(1024)(combined)

# #     dense1 = Activation('relu')(dense1)

#     dense2 = Dense(128)(combined)

#     dense2 = LeakyReLU()(dense2)

#     output = Dense(30)(dense2)

#     output = Activation('sigmoid')(output)

    

#     model = Model(inputs=[Title.input, Body.input, Answer.input, input_category], outputs=output)

#     model.compile(loss=loss, optimizer = Adam(lr=1e-3), metrics=['accuracy'])

#     return model
def make_model(embedding_matrices=embedding_mat, embed_size=300, loss='binary_crossentropy'):

    input_title = Input(shape=(X_tr['treated_question_title'].apply(lambda x: x.split()).map(len).max(),), name='title')

    input_body = Input(shape=(maxlen,), name='body')

    input_answer = Input(shape=(maxlen,), name='answer')

    input_category = Input(shape=(len(cat_cols),),name='categorical')

    GRU_nums = 256

    filter_size=[3,4,5]

    num_filters=512

    

    allwords = Concatenate(axis=1)([input_title,input_body, input_category, input_answer])

    allwords = Embedding(length_to_keep, embed_size, weights=[embedding_matrices],trainable=False)(allwords)

    

    lstm1 = Bidirectional(LSTM(GRU_nums, return_sequences=True), merge_mode='sum')(allwords)

    lstm2 = Bidirectional(LSTM(GRU_nums, return_sequences=True), merge_mode='sum')(lstm1)

#     lstm3 = Bidirectional(LSTM(GRU_nums, return_sequences=True), merge_mode='sum')(lstm2)

    

    conc_in = Concatenate(axis=2)([lstm2, allwords])

    conv1 = Conv1D(kernel_size=filter_size[0], strides=1, filters=num_filters, padding='same', name='conv1')(conc_in)

    conv1 = LeakyReLU()(conv1)

    conv2 = Conv1D(kernel_size=filter_size[1], strides=1, filters=num_filters, padding='same', name='conv2')(conv1)

    conv2 = LeakyReLU()(conv2)

    

    conv_max = GlobalMaxPooling1D()(conv2)

    conv_avg = GlobalAveragePooling1D()(conv2)

    

    conc_out = Concatenate(axis=1)([conv_max, conv_avg])

    

    dense1 = Add()([Dense(num_filters*2)(conc_out),conc_out])

    dense1 = LeakyReLU()(dense1)

    dense2 = Add()([Dense(num_filters*2)(dense1),conc_out])

    dense2 = LeakyReLU()(dense2)

    output = Dense(30)(dense2)

    output = Activation('sigmoid')(output)

    

    ######################################

        

    model = Model(inputs=[input_title, input_body, input_answer, input_category], outputs=output)

    model.compile(loss=loss, optimizer = Adam(lr=1e-3), metrics=['accuracy'])

    return model
model = make_model()

model.summary()
class TrainingMonitor(BaseLogger):

    def __init__(self, figPath=None, jsonPath=None, startAt=0):

        super().__init__()

        self.figPath = figPath

        self.jsonPath = jsonPath

        self.startAt = startAt

        

    def on_train_begin(self,logs={}):

        self.H = {}

        

        if self.jsonPath is not None:

            if os.path.exists(self.jsonPath):

                self.H = json.loads(open(self.jsonPath).read())

                

                if self.startAt>0:

                    for k in self.H:

                        H[k] = self.H[k][:self.startAt]

    

    def on_epoch_end(self, epoch, logs={}):

        for k,v in logs.items():

            orig_list = self.H.get(k,[])

            orig_list.append(float(v))

            self.H[k] = orig_list

            

        if self.jsonPath is not None:

            with open(self.jsonPath, 'w') as f:

                f.write(json.dumps(self.H))

                

        

        if len(self.H)>1:

            num = np.arange(0, len(self.H['loss']))

            fig = plt.figure(figsize=(12,8))

            sns.lineplot(x=num, y=self.H['loss'], label = 'Train Loss')

            sns.lineplot(x=num, y=self.H['accuracy'], label = 'Train Accuracy')

            sns.lineplot(x=num, y=self.H['val_loss'], label = 'Val Loss')

            sns.lineplot(x=num, y=self.H['val_accuracy'], label = 'Val Accuracy')

            plt.title('Training Loss and Accuracy [Epoch {}]')

            plt.xlabel('Epochs')

            plt.ylabel('Accuracy/Loss')

            plt.legend(loc='best')

            plt.savefig(self.figPath)

#             plt.show()

            plt.close()
earlystopping = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=1, min_delta=0.01, restore_best_weights=True)

modelcheckpoint = ModelCheckpoint(filepath='/kaggle/working/best_model.hdf5', monitor='val_loss', mode='min',

                                  verbose=1, min_delta=0.04, save_best_only=True)

reduceLR = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1,factor=0.1, mode='min', min_lr=1e-6)

trainingmonitor = TrainingMonitor(figPath='/kaggle/working/training.png', jsonPath='/kaggle/working/training.json')
H = model.fit([train_question_title, train_question_body, train_answer, X_tr[cat_cols]],

              y_tr, 

              validation_data=([val_question_title, val_question_body, val_answer, X_val[cat_cols]],y_val),

              epochs=30,

              batch_size=128,

              callbacks = [trainingmonitor, modelcheckpoint, reduceLR, earlystopping],

              verbose=1

             )

#fgfvcvc cccc cccccccccccccvcvc
preds = model.predict([test_question_title, test_question_body, test_answer, X_test[cat_cols]],

                      batch_size=128, verbose=1

                     )
preds, preds.shape
subm = test = pd.read_csv('../input/google-quest-challenge/sample_submission.csv', header=0,encoding='utf-8')

print(subm.shape)

subm.head()
subm[targets] = preds
subm.head()
subm.to_csv("submission.csv", index = False)
X_tr['treated_question_title'].apply(lambda x: x.split()).map(len).max()#, X_train['treated_question_title'].apply(lambda x: x.split()).map(len).mean()
# del model

# gc.collect()
# os.remove('/kaggle/working/best_model.hdf5')