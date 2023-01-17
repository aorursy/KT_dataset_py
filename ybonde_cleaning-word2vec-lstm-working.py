# importing the dependencies

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm # progress bar

import copy # perform deep copyong rather than referencing in python

import multiprocessing # for threading of word2vec model process



# importing classes helpfull for text processing

import nltk # general NLP

import re # regular expressions

import gensim.models.word2vec as w2v # word2vec model



import matplotlib.pyplot as plt # data visualization

%matplotlib inline



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
data_train.head(10)
data_test.head(10)
col_names = data_train.columns.values[2:]

col_names = col_names.tolist()

col_names.append('None')

x = [sum(data_train[y]) for y in data_train.columns.values[2:]]

x.append(len(data_train) - sum(x))
plt.figure(figsize = (10, 10))

plt.bar(np.arange(len(x)),x)

plt.xticks(np.arange(len(x)), col_names)

plt.xlabel('Catagories')

plt.ylabel('Occurrence')
train_sentences = data_train['comment_text'].values.tolist()

test_sentences = data_test['comment_text'].values.tolist()

# making a list of total sentences

total_ = copy.deepcopy(train_sentences)

total_.extend(test_sentences)

print('[*]Training Sentences:', len(train_sentences))

print('[*]Test Sentences:', len(test_sentences))

print('[*]Total Sentences:', len(total_))



# converting the text to lower

for i in tqdm(range(len(total_))):

    total_[i] = str(total_[i]).lower()
'''

Won't be performing by this method rather will be using the crude way to do it

#initialize rawunicode, we'll add all text to this one big string

corpus_raw = u""

#for each sentence, read it, convert in utf 8 format, add it to the raw corpus

for i in tqdm(range(len(total_))):

    corpus_raw += str(total_[i])

print('[*]Corpus is now', len(corpus_raw), 'characters long')



# converting everything to small letter removing all the non caps

corpus_lower = corpus_raw.lower()



# we do no need a seperate tokenizer, sentence_to_wordlist function does it for us

# tokenization process will result in words

def tokenizer(sentences):

    temp = []

    for i in tqdm(range(len(sentences))):

        temp.append(sentences.split())

    return temp



# NLTKs tokenizer was giving me a lot of difficulties, so dropping it for now

# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

'''
# convert into list of words remove unecessary characters, split into words,

# no hyphens and other special characters, split into words

def sentence_to_wordlist(raw):

    clean = re.sub("[^a-zA-Z0-9]"," ", raw)

    words = clean.split()

    return words
# tokenising the lowered corpus

clean_lower = []

for i in tqdm(range(len(total_))):

    clean_lower.append(sentence_to_wordlist(total_[i]))
# tokens and its count

total_tokens_l = []

for s in clean_lower:

    total_tokens_l.extend(s)

unk_tokens_l = list(set(total_tokens_l))

print("[!]Total number of tokens:", len(total_tokens_l))

print("[!]Total number of unique tokens:", len(unk_tokens_l))
# while we convert each sentence into it's feature matrix, we need to have a consistency

# of size, else we will not be able to train the model efficiently. For that we need the

# length of largest sentence, and that of the smallest

maxlen = max([len(s) for s in clean_lower])

minlen = min([len(s) for s in clean_lower])

print(maxlen)

print(minlen)
# finding index where length is zero

index = [int(i) for i,s in enumerate(clean_lower) if len(s) == 0]

print("[*]No. of entries with 0 length:", len(index))
# as we can see that all those sentences exist in test data set

print('[*]Minimum index with length 0:',min(index))

print('[*]Length of training dataset:', len(train_sentences))



# so reducing the values of index by length of train_sentences

index_test = [i-len(train_sentences) for i in index]

# looking at those sentences with 0 length

# print(len(train_sentences) < index[0])

print(test_sentences[index_test[0]])

print(test_sentences[index_test[12]])

print(test_sentences[index_test[34]])
# we remove these indexes and in submission classify them as 0.5 for all catagories

clean_ = [c for i,c in enumerate(clean_lower) if i not in index]
print(clean_[10])
# hyper parameters of the word2vec model

num_features = 200 # dimensions of each word embedding

min_word_count = 1 # this is not advisable but since we need to extract

# feature vector for each word we need to do this

num_workers = multiprocessing.cpu_count() # number of threads running in parallel

context_size = 7 # context window length

downsampling = 1e-3 # downsampling for very frequent words

seed = 1 # seed for random number generator to make results reproducible
word2vec_ = thrones2vec = w2v.Word2Vec(

    sg = 1, seed = seed,

    workers = num_workers,

    size = num_features,

    min_count = min_word_count,

    window = context_size,

    sample = downsampling

)
# first we need to built the vocab

word2vec_.build_vocab(clean_)
# now we need to train the model

word2vec_.train(clean_, total_examples = word2vec_.corpus_count, epochs = word2vec_.iter)
word2vec_.wv.most_similar('male')
word2vec_.wv.most_similar('gay')
word2vec_.wv.most_similar('dick')
# how to get vector for each word

vec_ = word2vec_['male']

print('[*]Shape of vec_:', vec_.shape)
'''

 if not os.exists('trained'):

    os.makedirs('trained')

    

w2vector_.save(os.path.join('trained', 'w2vector_.w2v'))



w2vector_ = word2vec.Word2Vec.load(os.path.join('trained', 'w2vector_.w2v'))

'''
# adding 'PAD' to each sequence

print('[!]Adding \'PAD\' to each sequence...')

for i in tqdm(range(len(clean_))):

    sentence = clean_[i][::-1]

    for _ in range(maxlen - len(sentence)):

        sentence.append('PAD')

    clean_[i] = sentence[::-1]

print()



# defining 'PAD'

PAD = np.zeros(word2vec_['guy'].shape)
# first we make the training set

train_features = []

print('[!]Making training features...')

for i in tqdm(range(len(data_training))):

    sentence = clean_[i]

    temp = []

    for token in sentence:

        temp.append(word2vec_[token].tolist())

    train_features.append(temp)



# perform on local machine no need to waste kaggle resources

# train_data = np.array(train_features)

print()
# now we make the testing set

test_features = []

print('[!]Making training features...')

for i in tqdm(range(len(data_testing))):

    sentence = clean_[i+len(data_training)]

    temp = []

    for token in sentence:

        temp.append(word2vec_[token].tolist())

    test_features.append(temp)

    

# perform on local machine no need to waste kaggle resources

# test_data = np.array(testing_features)

print()
# saving the numpy arrays

print('[!]Saving training data file at:', PATH_SAVE_DATA_TRAIN, ' ...')

np.save(PATH_SAVE_DATA_TRAIN , train_data)



print('[!]Saving testing data file at:', PATH_SAVE_DATA_TEST, ' ...')

np.save(PATH_SAVE_DATA_TEST , test_data)