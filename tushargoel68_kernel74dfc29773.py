import pandas as pd

from gensim.models import KeyedVectors

import tarfile

from nltk import word_tokenize

import os

from string import punctuation

from nltk.corpus import stopwords

import numpy as np

import math

from tqdm.notebook import tqdm
# Helper function to read csv

def getKeywordLists(keywordFile, seperator):

    df = pd.read_csv(keywordFile,sep=seperator)

    return df
def extract(tar_file, path):

    opened_tar = tarfile.open(tar_file)

    if tarfile.is_tarfile(tar_file):

        opened_tar.extractall(path)

    else:

        print("the tar you entered is not a tar file")

        

extract("/kaggle/input/trained-word2vec-model/word2vec_model.tar.xz", "/kaggle/output/kaggle/working")
# punctuations

b = '!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~'



# load nlp model

#nlp = spacy.load("en_core_sci_lg")



# load word2vec model

model = KeyedVectors.load_word2vec_format("/kaggle/output/kaggle/working/word2vec_model.txt")

print('word2vec model loaded successfully')

word2vec_vocabulary = list(model.wv.vocab)



# load extended stopword list

new_file1 = open("/kaggle/input/extended-stopword-list/Extended_Stopwords.txt")

tfidf_stopwords = new_file1.readlines()

for i in range(len(tfidf_stopwords)):

    tfidf_stopwords[i] = tfidf_stopwords[i].replace("\n", "")

    

punctuation_list = list(punctuation)

punctuation_list.append('``')     

punctuation_list.append("''")     

punctuation_list.append("'s")

punctuation_list.append("n't")

new_stopwords = set(tfidf_stopwords + stopwords.words("english") + punctuation_list)

stop_words = set(stopwords.words("english"))

print('stopwords list successfully loaded')
# function to check whether a given word is a number or not

def is_number(word):

    try:

        word = word.translate(str.maketrans('','',b))

        float(word)

    except ValueError:

        return False

    return True
# function to calculate L2 norm

def l2_norm(a):

    return math.sqrt(np.dot(a,a))



# function to Calculate cosine similarity

def cosine_similarity(a,b):

    return np.dot(a,b) / (l2_norm(a)*l2_norm(b))
# Helper function to get unique words of phrases

def gettingphrasewords(phrases):

    phrase_words = []

    for phrase in phrases:

        w1 = word_tokenize(phrase)

        if len(w1) > 1:

            phrase_words.append(phrase)

        for w in w1:

            w = w.lower()

            if w not in new_stopwords and w not in phrase_words:

                if w in word2vec_vocabulary:

                    phrase_words.append(w)

    return phrase_words



## Helper function to get phrase embedding

def getphraseembedding(phrases):

    phrase_embedding = dict()

    for indicator in phrases:

        list1 = []

        ind_words = word_tokenize(indicator)

        for word in ind_words:

            word = word.lower()

            if word in indicator.lower() and word not in new_stopwords:

                if word in word2vec_vocabulary and not is_number(word):

                    list1.append(model[word])



        if indicator not in phrase_embedding:

            phrase_embedding[indicator] = np.mean(list1, axis =0)

        else:

            phrase_embedding[indicator] = 0 

    return phrase_embedding
def generate_list_non_repeating_words(non_dup_sent):

    list_of_words=[]

    for j in tqdm(range(len(non_dup_sent)), total=len(non_dup_sent)):

        words = word_tokenize(non_dup_sent[j])

        words = map(lambda x : x.lower(),words)

        list_of_words.extend(words)

    non_duplicate_list_of_words = list(set(list_of_words))       

    non_duplicate_list_of_words = filter(lambda x : x not in new_stopwords,non_duplicate_list_of_words)

    non_duplicate_list_of_words = filter(lambda x : x in word2vec_vocabulary,non_duplicate_list_of_words)

    non_duplicate_list_of_words = list(filter(lambda x : not is_number(x),non_duplicate_list_of_words))    

    return non_duplicate_list_of_words

            
def generating_word_similarity_dictionary(non_dup_sent, phrase_words, phrase_embedding):

    similarity_dict = {}

    words_list = generate_list_non_repeating_words(non_dup_sent)

    for phrase in phrase_words:

        for s_word in words_list:

            if (phrase, s_word) not in similarity_dict.keys():

                similarity_dict[(phrase, s_word)] = round(cosine_similarity(phrase_embedding[phrase],model[s_word]),5)

    return similarity_dict

    
print('Reading extracted sentences from filtered covid documents')

sent_df = getKeywordLists("/kaggle/input/covid-19-dataset-filtering-and-sentence-extraction/Extracted_sentences_from_filtered_covid_documents.csv", seperator='\t')

tot_sent = list(sent_df['Sentence'])

non_dup_sent = list(set(tot_sent))

print('Total sentences available after removing duplicates : ',len(non_dup_sent))

phrases = ['covid-19 risk factors', 'hypertension covid-19', 'diabetes covid-19', 'heart disease covid-19', 'smoking covid-19', 'pulmonary disease covid-19', 'cancer covid-19', 'risk factors for neonates and pregnant women', 'respiratory disease covid-19', 'co-infections risk covid-19', 'incubation period covid-19', 'reproductive number covid-19', 'serial interval covid-19']

phrase_words = gettingphrasewords(phrases)

print('total number of phrase_words avialable : ',len(phrase_words))

phrase_embedding = getphraseembedding(phrase_words)

print('phrase embedding generated successfully')

similarity_dict = generating_word_similarity_dictionary(non_dup_sent, phrase_words, phrase_embedding)

print('size of similarity dictionary is : ',len(similarity_dict))

np.save('phrase_word_similarity_dictionary.npy', similarity_dict)
from itertools import islice



def take(n, iterable):

    "Return first n items of the iterable as a list"

    return list(islice(iterable, n))

n_items = take(10, similarity_dict.items())

print('showing few elements of dictionary..')

n_items